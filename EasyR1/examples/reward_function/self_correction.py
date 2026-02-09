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

import re
import math
import os
from typing import Any, Optional
import torch
import json
from mathruler.grader import extract_boxed_content, grade_answer



# 预编译正则表达式（避免每次调用时重新编译）
_FORMAT_PATTERN = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
_WHITESPACE_PATTERN = re.compile(r"\s*(<|>|/)\s*")
_SELF_CORRECTION_SPLIT_PATTERN = re.compile(r'</self-correction>')
_SELF_CORRECTION_CHECK_PATTERN = re.compile(r'</self-correction>')


def format_reward(response: str) -> float:
    """快速格式检查，使用预编译的正则"""
    think_start_count = response.count("<think>")
    think_end_count = response.count("</think>")
    if think_start_count != 1 or think_end_count != 1:
        return 0.0
    return 1.0 if _FORMAT_PATTERN.fullmatch(response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """准确率奖励"""
    if "</think>" in response:
        response = response.split("</think>")[-1]
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def self_correction_reward(acc_reward_1: float, acc_reward_2=-99.0, identical: bool = False) -> float:
    if acc_reward_2 != -99.0:
        if acc_reward_1 == 0.0 and acc_reward_2 ==1.0:
            reward = 1.0
        elif acc_reward_1 == 1.0 and acc_reward_2 ==0.0:
            reward = -0.25
        elif acc_reward_1 == 0.0 and acc_reward_2 ==0.0:
            reward = 0.0
        elif acc_reward_1 == 1.0 and acc_reward_2 ==1.0:
            reward = 0.75
        else:
            reward = 0.0
    else:
        reward = acc_reward_1

    if identical:
        reward = min(reward, 0.5)

    return reward

def compute_score(
    reward_inputs: list[dict[str, Any]], 
    format_weight: float = 0.1,
    group_ids: Optional[torch.Tensor] = None,
    select_top_k: int = 2,
    enable_selection: bool = True
) -> tuple[list[dict[str, float]], Optional[torch.Tensor]]:

    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    

    response_list = []
    selected_response_list = []
    for reward_input in reward_inputs:
        # 使用预编译的正则表达式，避免重复编译
        response_list.append(reward_input["response"])
        response = _WHITESPACE_PATTERN.sub(r"\1", reward_input["response"])
        ground_truth = reward_input["ground_truth"]
        
        # 快速检查是否包含self-correction标记
        has_self_correction = _SELF_CORRECTION_CHECK_PATTERN.search(response) is not None
        
        if has_self_correction:
            # 使用预编译的正则表达式分割
            parts = _SELF_CORRECTION_SPLIT_PATTERN.split(response, maxsplit=1)
            
            if len(parts) == 2:
                response_1, response_2 = parts[0].strip(), parts[1].strip()
            else:

                split_result = _SELF_CORRECTION_SPLIT_PATTERN.split(response, maxsplit=1)
                response_1 = split_result[0].strip()
                response_2 = split_result[1].strip() if len(split_result) > 1 else ""
            
            # 计算格式分数
            format_score_1 = format_reward(response_1)
            format_score_2 = format_reward(response_2)
            format_score = min(format_score_1, format_score_2)
            
            # 计算准确率分数
            acc_score_1 = accuracy_reward(response_1, ground_truth)
            acc_score_2 = accuracy_reward(response_2, ground_truth)


            identical_responses = response_1 == response_2
            self_correction_score = self_correction_reward(
                acc_score_1,
                acc_score_2,
                identical=identical_responses,
            )
            
            ans_1 = response_1.split("</think>")[-1].strip() if "</think>" in response_1 else response_1
            boxed_match = re.search(r"\\boxed\{(.*?)\}", ans_1, re.DOTALL)
            if boxed_match is None:
                self_correction_score = min(self_correction_score, 0.0)
            elif not boxed_match.group(1).strip():
                self_correction_score = min(self_correction_score, 0.0)

            if format_score_2 == 0.0:
                self_correction_score = min(self_correction_score, 0.5)
            
            # 计算总分
            overall_score = (1 - format_weight) * self_correction_score + format_weight * format_score
        else:
            self_correction_score = accuracy_reward(response, ground_truth) - 0.5

            format_score = format_reward(response)
            overall_score = (1 - format_weight) * self_correction_score + format_weight * format_score
            
    return overall_score
    