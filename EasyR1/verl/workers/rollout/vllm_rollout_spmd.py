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

import json
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from mathruler.grader import extract_boxed_content, grade_answer
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": 8}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        
        # 训练时的采样参数：添加 stop string 需要设置 detokenize=True
        sampling_kwargs_train = sampling_kwargs.copy()
        # sampling_kwargs_train["stop"] = "<self-correction>"
        # sampling_kwargs_train["max_tokens"] = config.response_length//2
        # sampling_kwargs_train["detokenize"] = True  # stop strings 需要 detokenize=True
        default_sampling_params_train = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params_train, key):
                sampling_kwargs_train[key] = getattr(config, key)

        print(f"Sampling params train: {sampling_kwargs_train}.")
        self.sampling_params_train = SamplingParams(**sampling_kwargs_train)
        self.concat_response_max_length = getattr(config, "concat_response_max_length", 6144)
        env_debug_log_path = os.getenv("EASYR1_CONCAT_DEBUG_LOG")
        config_debug_log_path = getattr(config, "concat_debug_log_path", None)
        if config_debug_log_path == "":
            self.concat_debug_log_path = None
        elif config_debug_log_path is not None:
            self.concat_debug_log_path = config_debug_log_path
        elif env_debug_log_path == "":
            self.concat_debug_log_path = None
        else:
            self.concat_debug_log_path = env_debug_log_path or os.path.join(
                os.getcwd(), "concat_response_debug.jsonl"
            )


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
                if hasattr(self.sampling_params_train, key):
                    old_value = getattr(self.sampling_params_train, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params_train, key, value)
        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
            setattr(self.sampling_params_train, key, value)

    def _clean_response_text(self, text: str) -> str:
        patterns = ("\n\n<self-correction>", "\n<self-correction>", "<self-correction>")
        idxs = [text.find(p) for p in patterns if p in text]
        if idxs:
            cut_pos = min(idxs)
            text = text[:cut_pos].rstrip()
        if not text.strip():
            stripped = text.replace("<self-correction>", "")
            if stripped != "":
                return stripped.rstrip()
        return text

    @staticmethod
    def _normalize_ground_truth(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = value.item()
            else:
                value = value.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            return None
        return str(value)

    def _is_response_correct(self, response_text: str, ground_truth: Optional[str]) -> bool:
        if not ground_truth:
            return False
        try:
            answer = extract_boxed_content(response_text)
        except Exception:
            return False
        if not answer:
            return False
        try:
            return bool(grade_answer(answer, ground_truth))
        except Exception:
            return False

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        self.sampling_params.max_tokens = 8192
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            # 使用当前的 max_tokens 进行 padding（可能被 val_override_config 或训练时动态设置）
            current_max_tokens = self.sampling_params.max_tokens
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=current_max_tokens
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "reward_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def _write_concat_debug_log(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads or not self.concat_debug_log_path:
            return
        log_path = self.concat_debug_log_path
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fout:
                for payload in payloads:
                    fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - debug log should not break training
            if self.rank == 0:
                print(f"[concat_debug_log] Failed to write log to {log_path}: {exc}")

    def _truncate_after_last_boxed(self, text: str) -> str:
        boxed_start_idx = text.rfind("\\boxed{")
        if boxed_start_idx == -1:
            return text
        brace_count = 0
        boxed_end_idx = -1
        for i in range(boxed_start_idx, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    boxed_end_idx = i + 1
                    break
        if boxed_end_idx != -1:
            return text[:boxed_end_idx]
        return text

    def concat_response_ids(self, response_ids, n, ground_truths, target_responses_per_prompt: Optional[int] = None):
        min_required_tokens = 15
        short_indices = [idx for idx, tokens in enumerate(response_ids) if len(tokens) < min_required_tokens]
        if short_indices:
            replacement_pool = [tokens for tokens in response_ids if len(tokens) >= min_required_tokens]
            if replacement_pool:
                pool_size = len(replacement_pool)
                for replace_order, idx in enumerate(short_indices):
                    replacement_tokens = replacement_pool[replace_order % pool_size]
                    if isinstance(replacement_tokens, torch.Tensor):
                        replacement_tokens = replacement_tokens.detach().cpu().tolist()
                    else:
                        replacement_tokens = list(replacement_tokens)
                    response_ids[idx] = replacement_tokens
            else:
                pass
        if n <= 0:
            raise ValueError("n must be greater than 0 when concatenating responses.")
        total_samples = len(response_ids)
        if total_samples % n != 0:
            raise ValueError(
                f"response count {total_samples} is not divisible by n={n}, unable to group by prompt."
            )

        num_prompts = total_samples // n
        if target_responses_per_prompt is None:
            target_responses_per_prompt = max(1, n * 2)
        ground_truth_array = ground_truths
        if ground_truth_array is None:
            ground_truth_array = [None] * num_prompts
        elif isinstance(ground_truth_array, np.ndarray):
            ground_truth_array = ground_truth_array.tolist()
        ground_truth_list = [
            self._normalize_ground_truth(ground_truth_array[i]) if i < len(ground_truth_array) else None
            for i in range(num_prompts)
        ]
        eos_token_id = self.tokenizer.eos_token_id
        sep_token = self.tokenizer.encode(
            "\n\n<self-correction>\n</self-correction>\n\n", add_special_tokens=False, return_tensors="pt"
        )[0].to(dtype=torch.long)

        def strip_trailing_eos(tokens: torch.Tensor) -> torch.Tensor:
            if eos_token_id is None or tokens.numel() == 0:
                return tokens
            while tokens.numel() > 0 and tokens[-1].item() == eos_token_id:
                tokens = tokens[:-1]
            return tokens

        def ensure_trailing_eos(tokens: torch.Tensor) -> torch.Tensor:
            if eos_token_id is None:
                return tokens
            if tokens.numel() == 0 or tokens[-1].item() != eos_token_id:
                eos_tensor = tokens.new_tensor([eos_token_id])
                tokens = torch.cat([tokens, eos_tensor], dim=-1)
            return tokens

        max_concat_length = self.concat_response_max_length

        def clip_to_max_length(tokens: torch.Tensor) -> torch.Tensor:
            if max_concat_length is None or tokens.size(0) <= max_concat_length:
                return tokens
            return tokens[:max_concat_length]

        new_response_ids: list[list[int]] = []

        for prompt_idx in range(num_prompts):
            start = prompt_idx * n
            end = start + n
            ground_truth = ground_truth_list[prompt_idx] if prompt_idx < len(ground_truth_list) else None

            prompt_originals = []
            original_correct_count = 0
            for sample_idx in range(start, end):
                token_ids = response_ids[sample_idx]
                text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

                first_sc_end = text.find("</self-correction>")
                if first_sc_end != -1:
                    second_sc_start = text.find("<self-correction>", first_sc_end + len("</self-correction>"))
                    if second_sc_start != -1:
                        text = text[:second_sc_start]
                        new_tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
                        new_tokens = new_tokens.to(dtype=torch.long, device="cpu")
                        new_tokens = strip_trailing_eos(new_tokens)
                        new_tokens = ensure_trailing_eos(new_tokens)
                        response_ids[sample_idx] = new_tokens.tolist()

                prompt_originals.append(response_ids[sample_idx])

                orig_text = self.tokenizer.decode(response_ids[sample_idx], skip_special_tokens=False)
                if self._is_response_correct(orig_text, ground_truth):
                    original_correct_count += 1

            y1_parts = []  # list of {"tokens": Tensor, "is_correct": bool}
            y2_parts = []  # list of {"tokens": Tensor, "is_correct": bool}

            for sample_idx in range(start, end):
                token_ids = response_ids[sample_idx]
                full_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                has_sc = "<self-correction>" in full_text and "</self-correction>" in full_text

                if has_sc:
                    y1_text = full_text.split("<self-correction>")[0].strip()
                else:
                    y1_text = full_text.strip()

                if y1_text and "\\boxed{" in y1_text:
                    y1_text = self._truncate_after_last_boxed(y1_text)
                    y1_tok = self.tokenizer.encode(y1_text, add_special_tokens=False, return_tensors="pt")[0]
                    y1_tok = y1_tok.to(dtype=torch.long, device="cpu")
                    y1_tok = strip_trailing_eos(y1_tok)
                    y1_correct = self._is_response_correct(y1_text, ground_truth)
                    y1_parts.append({"tokens": y1_tok, "is_correct": y1_correct})

                if has_sc:
                    y2_text = full_text.split("</self-correction>")[-1].strip()
                    if y2_text and "\\boxed{" in y2_text:
                        y2_text = self._truncate_after_last_boxed(y2_text)
                        y2_tok = self.tokenizer.encode(y2_text, add_special_tokens=False, return_tensors="pt")[0]
                        y2_tok = y2_tok.to(dtype=torch.long, device="cpu")
                        y2_tok = strip_trailing_eos(y2_tok)
                        y2_correct = self._is_response_correct(y2_text, ground_truth)
                        y2_parts.append({"tokens": y2_tok, "is_correct": y2_correct})

            concat_groups: dict[str, list[list[int]]] = {"cc": [], "cw": [], "wc": [], "ww": []}

            for y1 in y1_parts:
                for y2 in y2_parts:
                    concat_token = torch.cat(
                        [
                            y1["tokens"],
                            sep_token.to(device=y1["tokens"].device, dtype=y1["tokens"].dtype),
                            y2["tokens"],
                        ],
                        dim=-1,
                    )
                    concat_token = strip_trailing_eos(concat_token)
                    concat_token = ensure_trailing_eos(concat_token)
                    if max_concat_length is not None and concat_token.size(0) > max_concat_length:
                        concat_token = clip_to_max_length(concat_token)

                    ctype = ("c" if y1["is_correct"] else "w") + ("c" if y2["is_correct"] else "w")
                    concat_groups[ctype].append(concat_token.tolist())

            k = original_correct_count
            target_extra = max(0, target_responses_per_prompt - n)
            selected_spliced: list[list[int]] = []

            half_nk = max(0, n - k) // 2

            wc_pool = list(concat_groups["wc"])
            random.shuffle(wc_pool)
            wc_quota = min(half_nk, target_extra, len(wc_pool))
            selected_spliced.extend(wc_pool[:wc_quota])

            cc_pool = list(concat_groups["cc"])
            random.shuffle(cc_pool)
            cc_quota = min(half_nk, target_extra - len(selected_spliced), len(cc_pool))
            selected_spliced.extend(cc_pool[:cc_quota])

            remaining = target_extra - len(selected_spliced)
            if remaining > 0:
                cw_pool = list(concat_groups["cw"])
                random.shuffle(cw_pool)
                ww_pool = list(concat_groups["ww"])
                random.shuffle(ww_pool)

                cw_to_take = min(1, len(cw_pool)) if remaining >= 1 else 0
                selected_spliced.extend(cw_pool[:cw_to_take])

                remaining = target_extra - len(selected_spliced)
                if remaining > 0:
                    ww_to_take = min(remaining, len(ww_pool))
                    selected_spliced.extend(ww_pool[:ww_to_take])

                remaining = target_extra - len(selected_spliced)
                if remaining > 0 and cw_to_take < len(cw_pool):
                    extra_cw = min(remaining, len(cw_pool) - cw_to_take)
                    selected_spliced.extend(cw_pool[cw_to_take:cw_to_take + extra_cw])

            remaining = target_extra - len(selected_spliced)
            if remaining > 0:
                leftover = wc_pool[wc_quota:] + cc_pool[cc_quota:]
                random.shuffle(leftover)
                to_take = min(remaining, len(leftover))
                selected_spliced.extend(leftover[:to_take])

            remaining = target_extra - len(selected_spliced)
            if remaining > 0:
                for fill_i in range(remaining):
                    selected_spliced.append(prompt_originals[fill_i % n])

            final_list = prompt_originals + selected_spliced[:target_extra]
            new_response_ids.extend(final_list)

        return new_response_ids

    @torch.no_grad()
    def generate_sequences_train(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        ground_truths = non_tensor_batch.get("ground_truth")
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        target_responses_per_prompt = max(1, self.sampling_params_train.n * 2)


        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params_train, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = self.concat_response_ids(
                response_ids,
                self.sampling_params_train.n,
                ground_truths,
                target_responses_per_prompt=target_responses_per_prompt,
            )

            current_max_tokens = self.concat_response_max_length

            if self.sampling_params_train.max_tokens == 4096:
                current_max_tokens = 4096
            
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=current_max_tokens
            ).to(input_ids.device)

            if target_responses_per_prompt > 1:
                batch_size = batch_size * target_responses_per_prompt
                input_ids = _repeat_interleave(input_ids, target_responses_per_prompt)
                attention_mask = _repeat_interleave(attention_mask, target_responses_per_prompt)
                position_ids = _repeat_interleave(position_ids, target_responses_per_prompt)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, target_responses_per_prompt)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        
        reward_mask = response_mask.clone()
        kl_mask = torch.zeros_like(response_mask)
        self_correction_end_tag = "</self-correction>"
        training_stage = getattr(self.config, "training_stage", 1)

        if training_stage == 2:
            gt_array = ground_truths
            if gt_array is None:
                gt_array = [None] * (batch_size // target_responses_per_prompt)
            elif isinstance(gt_array, np.ndarray):
                gt_array = gt_array.tolist()
            gt_list = [
                self._normalize_ground_truth(gt_array[i]) if i < len(gt_array) else None
                for i in range(len(gt_array))
            ]

        for i in range(batch_size):

            resp_ids = response_ids[i].tolist()
            response_text = self.tokenizer.decode(resp_ids, skip_special_tokens=False)

            if self_correction_end_tag not in response_text:
                continue

            should_mask = False

            if training_stage == 1:

                should_mask = True
            elif training_stage == 2:

                prompt_idx = i // target_responses_per_prompt
                gt = gt_list[prompt_idx] if prompt_idx < len(gt_list) else None

                y1_text = response_text.split("<self-correction>")[0].strip()
                y2_text = response_text.split("</self-correction>")[-1].strip()

                y1_correct = self._is_response_correct(y1_text, gt)
                y2_correct = self._is_response_correct(y2_text, gt)

                if y1_correct != y2_correct:
                    should_mask = True

            if should_mask:
                tag_end_pos = response_text.find(self_correction_end_tag) + len(self_correction_end_tag)

                while tag_end_pos < len(response_text) and response_text[tag_end_pos] == '\n':
                    tag_end_pos += 1

                y1_and_separator_text = response_text[:tag_end_pos]

                y1_tokens = self.tokenizer.encode(y1_and_separator_text, add_special_tokens=False)
                y1_token_count = len(y1_tokens)

                if y1_token_count > 0:
                    response_mask[i, :y1_token_count] = 0

            if training_stage == 1 and "<self-correction>" in response_text:
                y1_only_text = response_text.split("<self-correction>")[0]
                y1_only_tokens = self.tokenizer.encode(y1_only_text, add_special_tokens=False)
                y1_only_count = len(y1_only_tokens)
                if y1_only_count > 0:
                    valid_count = min(y1_only_count, kl_mask.size(1))
                    kl_mask[i, :valid_count] = reward_mask[i, :valid_count]

        attention_mask = torch.cat((attention_mask, reward_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "reward_mask": reward_mask,
                "kl_mask": kl_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)


        return data_proto