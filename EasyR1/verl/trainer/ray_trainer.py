# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type, List

from tensordict import TensorDict

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from .config import PPOConfig
from collections import Counter
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def compute_rollout_accuracy_distribution(
    accuracy_list: list,
    n: int,
    step: int = 0,
    rank: int = 0,
    log_file: str = None,
) -> None:
    """
    统计每个问题的rollout正确率分布，并记录到jsonl文件。
    
    Args:
        accuracy_list: 每个回答的正确性列表 (0 或 1)
        n: 每个问题的rollout数量 (sampling_params.n)
        step: 当前训练step
        rank: 当前进程rank
        log_file: jsonl日志文件路径
    """
    if not accuracy_list or n <= 0:
        return
    
    total_samples = len(accuracy_list)
    if total_samples % n != 0:
        if rank == 0:
            print(f"[Step {step}] Warning: 样本数 {total_samples} 不能被 n={n} 整除，跳过统计")
        return
    
    num_questions = total_samples // n
    
    # 按问题分组，计算每个问题的正确率
    question_accuracies = []
    for q_idx in range(num_questions):
        start = q_idx * n
        end = start + n
        correct_count = sum(1 for acc in accuracy_list[start:end] if acc > 0)
        acc_ratio = correct_count / n
        question_accuracies.append(acc_ratio)
    
    # 统计各正确率区间的问题数量
    # 区间: 0% (全错), (0%, 25%], (25%, 50%], (50%, 75%], (75%, 100%), 100% (全对)
    distribution = {
        "all_wrong": 0,
        "0_to_125": 0,
        "125_to_25": 0,
        "25_to_375": 0,
        "375_to_50": 0,
        "50_to_625": 0,
        "625_to_75": 0,
        "75_to_875": 0,
        "875_to_100": 0,
        "all_correct": 0,
    }
    
    for acc in question_accuracies:
        if acc == 0.0:
            distribution["all_wrong"] += 1
        elif acc == 1.0:
            distribution["all_correct"] += 1
        elif acc <= 0.125:
            distribution["0_to_125"] += 1
        elif acc <= 0.25:
            distribution["125_to_25"] += 1
        elif acc <= 0.375:
            distribution["25_to_375"] += 1
        elif acc <= 0.50:
            distribution["375_to_50"] += 1
        elif acc <= 0.625:
            distribution["50_to_625"] += 1
        elif acc <= 0.75:
            distribution["625_to_75"] += 1
        elif acc <= 0.875:
            distribution["75_to_875"] += 1
        else:  # 0.875 < acc < 1.0
            distribution["875_to_100"] += 1
    
    # 计算平均正确率
    avg_accuracy = sum(question_accuracies) / len(question_accuracies) if question_accuracies else 0
    
    # 只在rank 0上打印和记录
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[Step {step}] Rollout正确率分布统计 (n={n}, 问题数={num_questions})")
        print(f"{'='*60}")
        labels = {
            "all_wrong": "0% (全错)",
            "0_to_125": "(0%, 12.5%]",
            "125_to_25": "(12.5%, 25%]",
            "25_to_375": "(25%, 37.5%]",
            "375_to_50": "(37.5%, 50%]",
            "50_to_625": "(50%, 62.5%]",
            "625_to_75": "(62.5%, 75%]",
            "75_to_875": "(75%, 87.5%]",
            "875_to_100": "(87.5%, 100%)",
            "all_correct": "100% (全对)",
        }
        for key, label in labels.items():
            count = distribution[key]
            pct = count / num_questions * 100 if num_questions > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {label:15s}: {count:4d} ({pct:5.1f}%) {bar}")
        print(f"{'='*60}")
        print(f"  平均正确率: {avg_accuracy*100:.1f}%")
        print(f"{'='*60}\n")
        
        # 写入jsonl文件
        if log_file:
            record = {
                "step": step,
                "n": n,
                "num_questions": num_questions,
                "avg_accuracy": avg_accuracy,
                "distribution": distribution,
                "pct_all_correct": distribution["all_correct"] / num_questions if num_questions > 0 else 0,
                "pct_all_wrong": distribution["all_wrong"] / num_questions if num_questions > 0 else 0,
                "pct_mixed": (num_questions - distribution["all_correct"] - distribution["all_wrong"]) / num_questions if num_questions > 0 else 0,
            }
            try:
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Warning] 写入rollout分布日志失败: {e}")


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: List[StatefulDataLoader],
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.save_val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.save_ckpt_metric = 0.0
        self.best_ckpt_metric = -1.0
        self.save_metric_scores: dict[str, float] = {"reward": 0.0}
        self.top_metric_ckpts: list[dict[str, float]] = []
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        self.max_response_length = config.data.max_response_length

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.save_val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.save_val_reward_score

        ckpt_metric = self.save_ckpt_metric
        if ckpt_metric > self.best_ckpt_metric:
            self.best_ckpt_metric = ckpt_metric
            self.best_global_step = self.global_step

        protected_steps = self._update_top_metric_ckpts(ckpt_metric)

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
            protected_steps=protected_steps,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "best_metric_score": round(self.best_ckpt_metric, 6),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
            "save_metric": self.config.trainer.save_metric,
            "top_metric_ckpts": self.top_metric_ckpts,
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _update_top_metric_ckpts(self, ckpt_metric: float) -> list[int]:
        if self.config.trainer.save_topk_best <= 0:
            self.top_metric_ckpts = []
            return []

        entry = {"step": self.global_step, "metric": round(float(ckpt_metric), 6)}
        self.top_metric_ckpts = [item for item in self.top_metric_ckpts if item["step"] != self.global_step]
        self.top_metric_ckpts.append(entry)
        self._trim_top_metric_ckpts()
        return [item["step"] for item in self.top_metric_ckpts]

    def _trim_top_metric_ckpts(self) -> None:
        if self.config.trainer.save_topk_best <= 0:
            self.top_metric_ckpts = []
            return

        self.top_metric_ckpts = sorted(
            self.top_metric_ckpts,
            key=lambda item: (-item["metric"], -item["step"]),
        )[: self.config.trainer.save_topk_best]

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
                self.best_ckpt_metric = tracker_info.get(
                    "best_metric_score", tracker_info.get("best_val_reward_score", 0.0)
                )
                tracker_metric_name = tracker_info.get("save_metric")
                if (
                    tracker_metric_name is not None
                    and tracker_metric_name != self.config.trainer.save_metric
                ):
                    print(
                        f"[checkpoint] Tracker metric `{tracker_metric_name}` differs from current "
                        f"`{self.config.trainer.save_metric}`. Protected checkpoints may not align."
                    )
                self.top_metric_ckpts = tracker_info.get("top_metric_ckpts", [])
                self._trim_top_metric_ckpts()
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        self.save_val_reward_score = 0.0
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        val_reward_metrics = None
        val_length_metrics = None
        print("Start validation...")
        val_reward_score = None
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for single_val_dataloader in self.val_dataloader:
            name = single_val_dataloader["name"]
            dataloader = single_val_dataloader["dataloader"]
            for batch_dict in dataloader:
                test_batch = DataProto.from_single_dict(batch_dict)
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
                repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
                test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
                test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
                test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
                test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps
                # test_gen_batch.meta_info["max_tokens"] = self.config.data.max_response_length
                test_gen_batch.meta_info["max_tokens"] = 8192
                # test_gen_batch.meta_info["stop"] = False

                test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
                test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

                # repeat to align with repeated responses in rollout
                test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

                # store generations
                input_ids = test_batch.batch["prompts"]
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                output_ids = test_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_inputs.extend(input_texts)
                sample_outputs.extend(output_texts)
                sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
                sample_scores.extend(scores)

                reward_tensor_lst.append(reward_tensor)
                for key, value in reward_metrics.items():
                    key = f"{name}/{key}"
                    reward_metrics_lst[key].extend(value)

                for key, value in compute_length_metrics(test_batch).items():
                    key = f"{name}/{key}"
                    length_metrics_lst[key].append(value)
            
            self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
            if val_reward_metrics is None:
                val_reward_metrics = {}
            if val_length_metrics is None:
                val_length_metrics = {}
            for key, value in reduce_metrics(reward_metrics_lst).items():
                val_reward_metrics[f"{key}_reward"] = value
            for key, value in reduce_metrics(length_metrics_lst).items():
                val_length_metrics[f"{key}"] = value

            self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
            if val_reward_score is None:
                val_reward_score = {}
            val_reward_score[f"val/{name}/reward_score"] = self.val_reward_score

            self.save_val_reward_score += self.val_reward_score

        self.save_val_reward_score /= len(self.val_dataloader)
        self.actor_rollout_ref_wg.release_rollout_engine()

        metric_averages = defaultdict(list)
        if val_reward_metrics is not None:
            for key, value in val_reward_metrics.items():
                if not key.endswith("_reward"):
                    continue
                metric_token = key.rsplit("/", 1)[-1]
                if metric_token.endswith("_reward"):
                    metric_name = metric_token[: -len("_reward")]
                else:
                    metric_name = metric_token
                metric_averages[metric_name].append(value)

        self.save_metric_scores = {"reward": self.save_val_reward_score}
        for metric_name, values in metric_averages.items():
            if len(values) == 0:
                continue
            self.save_metric_scores[metric_name] = float(np.mean(values))

        self.save_ckpt_metric = self.save_metric_scores.get(
            self.config.trainer.save_metric, self.save_val_reward_score
        )
        
        # self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        # val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        # val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {**val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)


    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "ground_truth"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )
            new_batch.non_tensor_batch["ground_truth"] = gen_batch.non_tensor_batch["ground_truth"]
            
            enable_concat = getattr(self.config.algorithm, "concat_responses", False)
            if enable_concat and self.config.worker.rollout.n < 2:
                raise ValueError(
                    f"concat_responses requires rollout.n >= 2, but got {self.config.worker.rollout.n}"
                )
            # 训练时限制 max_tokens 为 2048
            if enable_concat:
                gen_batch.meta_info["max_tokens"] = self.config.data.max_response_length
                gen_batch.meta_info["concat_responses"] = True
                gen_batch.meta_info["concat_max_response_length"] = self.max_response_length
            else:
                gen_batch.meta_info["max_tokens"] = self.config.data.max_response_length

            # generate a batch
            # TODO: use vllm to generate a batch
            if enable_concat:
                gen_batch_output = self.actor_rollout_ref_wg.generate_sequences_train(gen_batch)
            else:
                gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            if enable_concat:
                n = self.config.worker.rollout.n
                print(f"Response concatenation enabled: expect {n*2} samples per question.")

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences_train(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            # 注意：如果启用了response拼接，则不需要repeat，因为已经扩充到 n*2 了
            enable_concat = getattr(self.config.algorithm, "concat_responses", False)
            if enable_concat:
                # 扩充后已经是 original_size * n*2，需要repeat new_batch以对齐
                n = self.config.worker.rollout.n
                new_batch = new_batch.repeat(repeat_times=(n * 2), interleave=True)
                new_batch = new_batch.union(gen_batch_output)
            else:
                # 原始逻辑：repeat n 次然后union
                new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_result = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                if len(reward_result) == 2:
                    reward_tensor, reward_metrics = reward_result
                elif len(reward_result) == 3:
                    reward_tensor, selected_indices, reward_metrics = reward_result
                    if selected_indices is not None:
                        new_batch.reorder(selected_indices)
                else:
                    raise ValueError(f"compute_reward returned {len(reward_result)} values, expected 2 or 3")
                
                new_batch.batch["token_level_scores"] = reward_tensor
                
                # 在online_filtering模式下也统计rollout正确率分布
                if "accuracy" in reward_metrics:
                    n = self.config.worker.rollout.n
                    enable_concat = getattr(self.config.algorithm, "concat_responses", False)
                    if enable_concat:
                        n = n * 2
                    rollout_dist_log = os.path.join(
                        os.getcwd(), "rollout_accuracy_distribution.jsonl"
                    )
                    compute_rollout_accuracy_distribution(
                        accuracy_list=reward_metrics["accuracy"],
                        n=n,
                        step=self.global_step,
                        rank=int(os.getenv("RANK", "0")),
                        log_file=rollout_dist_log,
                    )
                
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            # 如果启用了response拼接，batch_size需要除以 n*2
            enable_concat = getattr(self.config.algorithm, "concat_responses", False)
            if enable_concat:
                n = self.config.worker.rollout.n
                current_batch_size = len(batch) // (n * 2)  # 扩充了 n*2 倍
            else:
                current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                # 如果启用了response拼接，需要乘以 n*2
                enable_concat = getattr(self.config.algorithm, "concat_responses", False)
                if enable_concat:
                    n = self.config.worker.rollout.n
                    return batch[: self.config.data.rollout_batch_size * (n * 2)]
                else:
                    return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def reward_preprocess(self, batch: DataProto) -> DataProto:
        """
        Preprocess the batch for reward computation.
        """
        
        return batch


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # self.reward_preprocess(batch)

                # 在balance之前计算reward用于统计正确率分布（balance会打乱数据顺序）
                if "token_level_scores" not in batch.batch:
                    with timer("reward_for_stats", timing_raw):
                        reward_result_for_stats = ray.get(self.reward_fn.compute_reward.remote(batch))
                        if len(reward_result_for_stats) == 2:
                            _, reward_metrics_for_stats = reward_result_for_stats
                        elif len(reward_result_for_stats) == 3:
                            _, _, reward_metrics_for_stats = reward_result_for_stats
                        else:
                            reward_metrics_for_stats = {}
                        
                        # 统计rollout正确率分布
                        if "accuracy" in reward_metrics_for_stats:
                            accuracy_list = reward_metrics_for_stats["accuracy"]
                            n = self.config.worker.rollout.n
                            enable_concat = getattr(self.config.algorithm, "concat_responses", False)
                            if enable_concat:
                                n = n * 2
                            rollout_dist_log = os.path.join(
                                os.getcwd(), "rollout_accuracy_distribution.jsonl"
                            )
                            compute_rollout_accuracy_distribution(
                                accuracy_list=accuracy_list,
                                n=n,
                                step=self.global_step,
                                rank=int(os.getenv("RANK", "0")),
                                log_file=rollout_dist_log,
                            )

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # 先计算reward和样本选择，再进行balance（样本选择逻辑在reward函数中）
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        # reward函数可以返回：
                        # 1. (reward_tensor, reward_metrics) - 不进行样本选择
                        # 2. (reward_tensor, selected_indices, reward_metrics) - 进行样本选择
                        reward_result = ray.get(self.reward_fn.compute_reward.remote(batch))
                        
                        if len(reward_result) == 2:
                            # 不进行样本选择
                            reward_tensor, reward_metrics_raw = reward_result
                        elif len(reward_result) == 3:
                            # 进行样本选择
                            reward_tensor, selected_indices, reward_metrics_raw = reward_result
                            # 根据选择的索引过滤batch
                            if selected_indices is not None:
                                batch.reorder(selected_indices)
                        else:
                            raise ValueError(f"compute_reward返回值格式错误: {len(reward_result)}个元素")
                        
                        batch.batch["token_level_scores"] = reward_tensor
                        
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics_raw).items()}
                        metrics.update(reward_metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
