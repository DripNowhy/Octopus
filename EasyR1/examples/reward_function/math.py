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
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        original_response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        second_part = original_response
        # 初始化
        response_for_grade = original_response
        response_first = original_response


        # 如果包含 self-correction 标记，检查后半部分是否有 \boxed{}
        if "</self-correction>" in original_response:
            second_part = original_response.split("</self-correction>")[-1] if "<think>" in original_response.split("</self-correction>")[-1] else original_response
            # 使用正则表达式检查是否包含 \boxed{...} 格式
            if re.search(r'\\boxed\{.*?\}', second_part):
                response_for_grade = second_part
            # 否则使用完整的 response
        
        if "<self-correction>" in original_response:
            first_part = original_response.split("<self-correction>")[0]
            response_first = first_part  # 只使用修正后的部分


        format_score = format_reward(response_for_grade)
        accuracy_score = accuracy_reward(response_for_grade, reward_input["ground_truth"])
        accuracy_score_first = accuracy_reward(response_first, reward_input["ground_truth"])
        accuracy_score_second = accuracy_reward(second_part, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy_first": accuracy_score_first,
                "accuracy_second": accuracy_score_second,
                "accuracy": accuracy_score,
            }
        )

    return scores
