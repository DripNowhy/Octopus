import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# vLLM offline inference (for srun / easyr1)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

try:
    from PIL import Image
    from PIL.Image import Image as PILImage
except Exception as e:  # pragma: no cover
    raise RuntimeError("需要安装 pillow 才能加载 image_path。请在 easyr1 环境中 `pip install pillow`。") from e


# Defaults (can be overridden by CLI args)
DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_INPUT_FILE = "./dataset/sft_judge_output/final_judged_results.jsonl"
DEFAULT_OUTPUT_FILE = "./dataset/sft_output/octopus_corrected_10k.jsonl"


SYSTEM_PROMPT = (
    "You are an expert AI assistant. Your task is to generate a high-quality answer to the user's question. "
    "You will be provided with a reference answer and the ground truth. If the reference answer is incorrect, "
    "generate a correct one. If it is correct, generate an improved version. Do NOT mention or analyze the "
    "reference answer in your response. Just provide the direct answer."
)


def _process_image(
    image: Any, min_pixels: Optional[int], max_pixels: Optional[int]
) -> Optional["PILImage"]:
    """Load and optionally resize/convert an image.

    - image can be a file path (str), a PIL image, or bytes-like.
    - returns a PIL RGB image, or None if cannot be loaded.
    """
    try:
        if image is None:
            return None
        if isinstance(image, str):
            if not os.path.exists(image):
                return None
            img = Image.open(image)
        elif isinstance(image, (bytes, bytearray)):
            from io import BytesIO

            img = Image.open(BytesIO(image))
        elif hasattr(image, "read"):
            img = Image.open(image)
        else:
            # assume PIL image-like
            img = image

        img.load()  # avoid "Too many open files"

        if max_pixels is not None and (img.width * img.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (img.width * img.height))
            width, height = int(img.width * resize_factor), int(img.height * resize_factor)
            img = img.resize((width, height))

        if min_pixels is not None and (img.width * img.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (img.width * img.height))
            width, height = int(img.width * resize_factor), int(img.height * resize_factor)
            img = img.resize((width, height))

        if img.mode != "RGB":
            img = img.convert("RGB")

        return img
    except Exception:
        return None


def _build_prompt_text(question: str, ground_truth: str, reference_answer: str) -> str:
    return f"""Question: {question}

Ground Truth: {ground_truth}

Reference Answer:
{reference_answer}

Instruction:
Based on the provided information, generate a new, complete answer.
- Ensure the reasoning is correct and leads to the Ground Truth.
- If the Reference Answer is wrong, correct it implicitly by providing the right derivation.
- If the Reference Answer is correct, rewrite it to be clearer and more logical while keeping a similar length.
- You first think through the reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \\boxed{{}}.
- Do not generate thinking process that are too short.
- Do not mention the Reference Answer in your response."""


def _count_existing_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _chunked(seq: List[Dict[str, Any]], chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield i, seq[i : i + chunk_size]


def main():
    parser = argparse.ArgumentParser(description="Offline vLLM inference for Qwen3-VL (batch=512).")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("TENSOR_PARALLEL_SIZE", "2")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")))
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("MAX_MODEL_LEN", "32768")))
    # Python>=3.9 supports BooleanOptionalAction
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Load JSONL file
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    print(f"Loaded {len(data)} items from {args.input_file}")

    start_idx = _count_existing_lines(args.output_file)
    if start_idx > 0:
        print(f"Found existing output with {start_idx} lines. Resuming from index {start_idx}...")
    data_to_process = data[start_idx:]
    if not data_to_process:
        print("All items processed.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    # vLLM versions differ; keep init robust.
    try:
        llm = LLM(
            model=args.model_path,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
            # multimodal: follow EasyR1 settings
            disable_mm_preprocessor_cache=True,
        )
    except TypeError:
        llm = LLM(
            model=args.model_path,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
        )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        # We want text output directly.
        detokenize=True,
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    mode = "a" if start_idx > 0 else "w"
    total_batches = math.ceil(len(data_to_process) / args.batch_size)
    with open(args.output_file, mode, encoding="utf-8") as out_f, tqdm(
        total=total_batches, desc="Processing", unit="batch"
    ) as pbar:
        for _, batch in _chunked(data_to_process, args.batch_size):
            vllm_inputs = []
            metas = []
            images_loaded = 0

            for item in batch:
                question = item["question"]
                image_path = item.get("image_path")
                ground_truth = item.get("ground_truth", "")
                reference_answer = item.get("model_answer", "")

                prompt_text = _build_prompt_text(question, ground_truth, reference_answer)

                user_content = []
                img = _process_image(image_path, args.min_pixels, args.max_pixels)
                if img is not None:
                    user_content.append({"type": "image"})
                    images_loaded += 1
                user_content.append({"type": "text", "text": prompt_text})

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                # Tokenize with chat template (Qwen3-VL template inserts vision tokens automatically)
                prompt_token_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )

                vllm_inp: Dict[str, Any] = {"prompt_token_ids": prompt_token_ids}
                if img is not None:
                    # vLLM multimodal expects list of images for "image"
                    vllm_inp["multi_modal_data"] = {"image": [img]}

                vllm_inputs.append(vllm_inp)
                metas.append((question, image_path, ground_truth, reference_answer))

            try:
                outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
                # outputs are in the same order as inputs
                for out, (question, image_path, ground_truth, reference_answer) in zip(outputs, metas):
                    text = ""
                    try:
                        if out.outputs:
                            text = out.outputs[0].text
                    except Exception:
                        text = ""

                    result = {
                        "question": question,
                        "image_path": image_path,
                        "ground_truth": ground_truth,
                        "original_reference": reference_answer,
                        "correction": text,
                        "model": os.path.basename(args.model_path.rstrip("/")),
                        "status": "success",
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e:
                # If batch fails, fall back to per-sample to salvage progress.
                for vllm_inp, (question, image_path, ground_truth, reference_answer) in zip(vllm_inputs, metas):
                    try:
                        out = llm.generate([vllm_inp], sampling_params=sampling_params)[0]
                        text = out.outputs[0].text if out.outputs else ""
                        result = {
                            "question": question,
                            "image_path": image_path,
                            "ground_truth": ground_truth,
                            "original_reference": reference_answer,
                            "correction": text,
                            "model": os.path.basename(args.model_path.rstrip("/")),
                            "status": "success",
                        }
                    except Exception as e2:
                        result = {
                            "question": question,
                            "image_path": image_path,
                            "ground_truth": ground_truth,
                            "original_reference": reference_answer,
                            "error": f"{type(e2).__name__}: {str(e2)}",
                            "model": os.path.basename(args.model_path.rstrip("/")),
                            "status": "failed",
                        }
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            finally:
                pbar.update(1)


if __name__ == "__main__":
    main()

