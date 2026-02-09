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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        mask_for_reward = data.batch.get("reward_mask", data.batch["response_mask"])
        response_length = torch.sum(mask_for_reward, dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        # 优先使用reward_mask（完整response），如果不存在则使用response_mask
        mask_for_reward = data.batch.get("reward_mask", data.batch["response_mask"])
        response_length = torch.sum(mask_for_reward, dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class SelectiveBatchFunctionRewardManager(FunctionRewardManager):

    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict[str, list[float]]]:

        reward_inputs = []
        response_ids = data.batch["responses"]
        mask_for_reward = data.batch.get("reward_mask", data.batch["response_mask"])
        response_length = torch.sum(mask_for_reward, dim=-1)
        
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )

        group_ids = data.batch.get("group_ids", None)
        
        reward_result = self.reward_fn(reward_inputs, group_ids=group_ids)
        
        if isinstance(reward_result, tuple) and len(reward_result) == 2:
            scores, selected_indices = reward_result
        else:
            scores = reward_result
            selected_indices = None
        
        reward_metrics = defaultdict(list)
        
        if selected_indices is not None:
            selected_response_length = response_length[selected_indices]
            reward_tensor = torch.zeros((len(scores), data.batch["responses"].size(1)), dtype=torch.float32)
            
            for i, score in enumerate(scores):
                cur_response_length = int(selected_response_length[i].item())
                reward_tensor[i, cur_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)
        else:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            
            for i, score in enumerate(scores):
                cur_response_length = int(response_length[i].item())
                reward_tensor[i, cur_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

        return reward_tensor, selected_indices, reward_metrics
