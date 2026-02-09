#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将过滤后的 jsonl（每行一个 dict，包含 question/image_path/correction 等字段）
转换成 LLaMA-Factory SFT 所需的 JSON 数组格式（参考 select_data.py 122-134 行）。

输出样本格式：
{
  "conversation": [
    {"from": "human", "value": "<image>{question}\\n\\n..."},
    {"from": "gpt", "value": "{final_answer}"}
  ],
  "images": ["{image_path}"]
}
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


HUMAN_SUFFIX = (
    "\n\nYou first think through your reasoning process as an internal monologue, enclosed within <think> </think> tags. "
    "Then, provide your final answer enclosed within \\boxed{}. If you believe the answer can be further enhanced, "
    "generate <self-correction> </self-correction> tags enclosed with no content, and regenerate a new reasoning process "
    "and a new answer from scratch after that. The new response should first think through your reasoning process as an "
    "internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \\boxed{}. "
    "All reasoning, answer steps must be included without omission."
)

SELF_CORRECTION_BLOCK = "\n\n<self-correction>\n</self-correction>\n\n"


def parse_json_line(line: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    line = line.strip()
    if not line:
        return None, "empty"
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            return None, "not_dict"
        return obj, None
    except Exception as e:
        return None, f"json_error:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser(description="Export filtered Qwen3-VL corrections to LLaMA-Factory JSON.")
    ap.add_argument("--input-jsonl", type=str, default="./dataset/sft_output/octopus_corrected_10k.jsonl")
    ap.add_argument("--output-json", type=str, default="./dataset/sft_output/octopus_corrected_10k_format.json")
    ap.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--question-key", type=str, default="question")
    ap.add_argument("--image-key", type=str, default="image_path")
    ap.add_argument("--answer-key", type=str, default="correction")
    ap.add_argument("--reference-key", type=str, default="original_reference")
    args = ap.parse_args()

    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(args.input_jsonl)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    data: List[Dict[str, Any]] = []
    total = 0
    bad = 0
    missing = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            obj, err = parse_json_line(line)
            if err is not None:
                bad += 1
                continue

            q = obj.get(args.question_key)
            img = obj.get(args.image_key)
            ans = obj.get(args.answer_key)
            ref = obj.get(args.reference_key)
            if not isinstance(q, str) or not isinstance(img, str) or not isinstance(ans, str) or not isinstance(ref, str):
                missing += 1
                continue

            # gpt value: 原参考回答 + 空 self-correction 标签 + 新回答
            final_answer = ref.strip() + SELF_CORRECTION_BLOCK + ans.strip()

            item = {
                "conversation": [
                    {
                        "from": "human",
                        "value": "<image>" + q + HUMAN_SUFFIX,
                    },
                    {
                        "from": "gpt",
                        "value": final_answer,
                    },
                ],
                "images": [img],
            }
            data.append(item)

    # Filter to 5000 samples first
    data = data[:5000]

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(data)

    with open(args.output_json, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, indent=4, ensure_ascii=False)

    print(
        json.dumps(
            {
                "input_jsonl": args.input_jsonl,
                "output_json": args.output_json,
                "total_lines": total,
                "exported": len(data),
                "bad_json_lines": bad,
                "missing_required_fields": missing,
                "shuffle": args.shuffle,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


