import os
import json
import math
import multiprocessing
import re
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# 设置启动方法为 spawn，这对 CUDA 多进程是必须的
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

PROMPT_TEMPLATE = """You are provided with a question and two responses. You need to judge whether this two responses are consistent with each other.

Question: {question}
Response 1: {response1}
Response 2: {response2}

If the two responses are consistent with each other, output "Yes".
If the two responses are not consistent with each other, output "No".

Please make sure your output is only "Yes" or "No".
"""

def check_format(text):
    """
    检查文本是否同时包含 <think>...</think> 标签和 \\boxed{...} 结构
    """
    if not text:
        return False
    
    has_think = bool(re.search(r'<think>.*?</think>', text, re.DOTALL))
    has_boxed = "\\boxed{" in text
    
    return has_think and has_boxed

def load_data(data_path):
    dataset = []
    print(f"Loading data from {data_path}...")
    with open(data_path, "r") as f:
        for line_idx, line in enumerate(f):
            try:
                item = json.loads(line)
                dataset.append(item)
            except Exception as e:
                print(f"Error parsing line {line_idx}: {e}")
                continue
    print(f"Loaded {len(dataset)} items.")
    return dataset

def run_judge(gpu_id, data_chunk, output_file):
    """
    Worker function to run judgment on a specific GPU.
    """
    # 设置当前进程可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process on GPU {gpu_id} started. Processing {len(data_chunk)} items.")

    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    
    # 初始化 vLLM
    try:
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            max_model_len=4096, # Judge 不需要特别长的 context
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 5},
            dtype="bfloat16",
        )
    except Exception as e:
        print(f"GPU {gpu_id}: Failed to initialize vLLM: {e}")
        return

    # 初始化 Processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0, # Judge 最好用 greedy
        max_tokens=20
    )

    BATCH_SIZE = 128
    results = []
    
    for i in range(0, len(data_chunk), BATCH_SIZE):
        batch_items = data_chunk[i : i + BATCH_SIZE]
        batch_inputs = []
        valid_indices = []

        print(f"GPU {gpu_id}: Preparing batch {i // BATCH_SIZE + 1} ({len(batch_items)} items)...")

        for idx, item in enumerate(batch_items):
            # 1. Python Format Check
            model_output = item.get("model_output", "")
            is_format_correct = check_format(model_output)
            item["is_format_correct"] = is_format_correct
            
            # 2. Prepare LLM Judge
            question = item.get("question", "")
            gt_answer = item.get("gt_answer", "")
            extracted_answer = item.get("extracted_answer", "")
            
            # 如果提取为空，可能不需要 judge，直接认为不一致？
            # 这里还是让模型 judge 一下，或者你可以加逻辑优化
            
            prompt_content = PROMPT_TEMPLATE.format(question=question, response1=gt_answer, response2=extracted_answer)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_content},
                    ],
                }
            ]
            
            try:
                prompt_text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # 纯文本输入，不需要 image
                batch_inputs.append({
                    "prompt": prompt_text,
                })
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"GPU {gpu_id}: Error preparing prompt for item {item.get('id', 'unknown')}: {e}")
                # 即使出错，也要保留 format check 结果？
                # 这里简单处理，如果出错就不 judge consistency 了
                item["judge_consistency"] = "Error"
                results.append(item) # 仍然保存
                continue

        if not batch_inputs:
            # 如果整个 batch 都没有有效输入（比如只有 format check），把处理过的 item 加进去
            # 上面的循环逻辑有点问题，如果没有 batch_inputs，说明 valid_indices 为空
            # 但我们已经对 batch_items 做了 format check，需要保存
            # 下面的 outputs 循环只处理 valid 的
            # 简单的做法：先全部做 format check，然后收集需要 LLM 的
            pass

        # 运行 LLM
        if batch_inputs:
            try:
                outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
                
                for j, output in enumerate(outputs):
                    original_idx = valid_indices[j]
                    judge_result = output.outputs[0].text.strip()
                    
                    # 移除可能的标点
                    if "yes" in judge_result.lower():
                        judge_result = "Yes"
                    elif "no" in judge_result.lower():
                        judge_result = "No"
                    else:
                        judge_result = "Error"
                        
                    batch_items[original_idx]["judge_consistency"] = judge_result
            
            except Exception as e:
                print(f"GPU {gpu_id}: Error during generation: {e}")

        # 将 batch_items (此时已包含 format_correct 和 judge_consistency) 加入结果
        results.extend(batch_items)
        
        # 定期写入
        if len(results) >= 500:
            mode = "a" if os.path.exists(output_file) else "w"
            with open(output_file, mode) as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            results = []

    # 保存剩余
    if results:
        mode = "a" if os.path.exists(output_file) else "w"
        print(f"GPU {gpu_id}: Saving remaining {len(results)} results to {output_file}")
        with open(output_file, mode) as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

def main():
    # 输入文件（生成结果）
    input_file = "./dataset/sft_output/final_merged_results.jsonl"
    # 输出文件（Judge 结果）
    output_dir = "./dataset/sft_judge_output"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return
        
    full_dataset = load_data(input_file)
    total_items = len(full_dataset)
    
    num_gpus = 8
    chunk_size = math.ceil(total_items / num_gpus)
    
    processes = []
    print(f"Starting {num_gpus} processes for judging {total_items} items...")

    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_items)
        
        if start_idx >= total_items:
            break
            
        data_chunk = full_dataset[start_idx:end_idx]
        output_file = os.path.join(output_dir, f"judged_part_{i}.jsonl")
        
        if os.path.exists(output_file):
            os.remove(output_file)
            
        p = multiprocessing.Process(
            target=run_judge,
            args=(i, data_chunk, output_file)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    print("All processes completed.")
    
    # 合并
    final_output = os.path.join(output_dir, "final_judged_results.jsonl")
    print(f"Merging results to {final_output}...")
    with open(final_output, "w") as outfile:
        for i in range(num_gpus):
            part_file = os.path.join(output_dir, f"judged_part_{i}.jsonl")
            if os.path.exists(part_file):
                with open(part_file, "r") as infile:
                    for line in infile:
                        outfile.write(line)
    print("Done!")

if __name__ == "__main__":
    main()
