import os
import json
import math
import multiprocessing
import re
from PIL import Image
from sympy.logic import false
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

PROMPT_TEMPLATE = """\nYou first think through the reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \\boxed{}.
"""

def extract_answer(text):
    """
    提取文本中最后一个 \\boxed{...} 里的内容。
    支持嵌套的花括号。
    """
    if not text:
        return ""
    
    start_marker = "\\boxed{"
    start_idx = text.rfind(start_marker)
    
    if start_idx == -1:
        return ""
    
    content_start = start_idx + len(start_marker)
    balance = 1
    content = []
    
    for i in range(content_start, len(text)):
        char = text[i]
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
            
        if balance == 0:
            return "".join(content)
        
        content.append(char)
        
    return ""

def load_data(data_path):
    dataset = []
    image_list = ["ai2d", "chartqa", "CLEVR_v1.0", "geoqa+", "gqa", "sqa", "textvqa", "ocr_vqa", "vg", "web", "wikiart"]
    print(f"Loading data from {data_path}...")
    with open(data_path, "r") as f:
        for line_idx, line in enumerate(f):
            try:
                item = json.loads(line)
                image_path = os.path.join("./dataset/LLaVA-CoT-100k", item["image"])
                image_exist = False
                for image_name in image_list:
                    if image_name in image_path:
                        image_exist = True
                        break
                if not image_exist:
                    continue
                if "conversations" in item and len(item["conversations"]) >= 2:
                    question = item["conversations"][0]["value"]
                    gt_answer = item["conversations"][1]["value"].split("<CONCLUSION>")[-1].split("</CONCLUSION>")[0].strip()
                    
                    dataset.append({
                        "id": line_idx,
                        "image_path": image_path,
                        "question": question,
                        "answer": gt_answer
                    })
            except Exception as e:
                print(f"Error parsing line {line_idx}: {e}")
                continue
    print(f"Loaded {len(dataset)} items.")
    return dataset

def run_inference(gpu_id, data_chunk, output_file):
    """
    Worker function to run inference on a specific GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process on GPU {gpu_id} started. Processing {len(data_chunk)} items.")

    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    
    try:
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            max_model_len=8192 * 2, # 根据显存调整
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 5},
            dtype="bfloat16",
        )
    except Exception as e:
        print(f"GPU {gpu_id}: Failed to initialize vLLM: {e}")
        return

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        n=4,
        temperature=0.6,
        top_p=0.9,
        top_k=20,
        max_tokens=2048
    )

    BATCH_SIZE = 128
    results = []
    
    # 按 batch 处理数据
    for i in range(0, len(data_chunk), BATCH_SIZE):
        batch_items = data_chunk[i : i + BATCH_SIZE]
        batch_inputs = []
        valid_items_indices = []
        
        print(f"GPU {gpu_id}: Preparing batch {i // BATCH_SIZE + 1} ({len(batch_items)} items)...")

        for idx, item in enumerate(batch_items):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": item["image_path"]},
                            {"type": "text", "text": item["question"] + PROMPT_TEMPLATE},
                        ],
                    }
                ]
                
                prompt_text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # 准备 multi_modal_data
                try:
                    image = Image.open(item["image_path"]).convert("RGB")
                except Exception as e:
                    print(f"GPU {gpu_id}: Error loading image {item['image_path']}: {e}")
                    continue

                inputs = {
                    "prompt": prompt_text,
                    "multi_modal_data": {
                        "image": image
                    },
                }
                
                batch_inputs.append(inputs)
                valid_items_indices.append(idx)

            except Exception as e:
                print(f"GPU {gpu_id}: Error preparing item {item.get('id', 'unknown')}: {e}")
                continue
        
        if not batch_inputs:
            continue

        try:
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            
            for j, output in enumerate(outputs):
                original_item_idx = valid_items_indices[j]
                original_item = batch_items[original_item_idx]
                
                for k, completion in enumerate(output.outputs):
                    generated_text = completion.text
                    extracted_val = extract_answer(generated_text)
                    
                    result_item = {
                        "id": original_item["id"],
                        "sample_index": k,
                        "image": original_item["image_path"],
                        "question": original_item["question"],
                        "gt_answer": original_item["answer"],
                        "model_output": generated_text,
                        "extracted_answer": extracted_val
                    }
                    results.append(result_item)
            
            print(f"GPU {gpu_id}: Processed {len(results)}/{len(data_chunk)}")
            
            if len(results) >= 500:
                mode = "a" if os.path.exists(output_file) else "w"
                with open(output_file, mode) as f:
                    for res in results:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                results = []

        except Exception as e:
            print(f"GPU {gpu_id}: Error during batch generation: {e}")
            continue

    if results:
        mode = "a" if os.path.exists(output_file) else "w"
        print(f"GPU {gpu_id}: Saving remaining {len(results)} results to {output_file}")
        with open(output_file, mode) as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

def main():
    data_path = "./dataset/LLaVA-CoT-100k/train.jsonl"
    output_dir = "./dataset/sft_output"
    os.makedirs(output_dir, exist_ok=True)
    
    full_dataset = load_data(data_path)
    total_items = len(full_dataset)
    
    if total_items == 0:
        print("No data loaded. Exiting.")
        return

    num_gpus = 8
    chunk_size = math.ceil(total_items / num_gpus)
    
    processes = []
    
    print(f"Starting {num_gpus} processes for {total_items} items (approx {chunk_size} per GPU)...")

    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_items)
        
        if start_idx >= total_items:
            break
            
        data_chunk = full_dataset[start_idx:end_idx]
        output_file = os.path.join(output_dir, f"result_part_{i}.jsonl")
        
        if os.path.exists(output_file):
            os.remove(output_file)
        
        p = multiprocessing.Process(
            target=run_inference,
            args=(i, data_chunk, output_file)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    print("All processes completed.")
    
    final_output = os.path.join(output_dir, "final_merged_results.jsonl")
    print(f"Merging results to {final_output}...")
    with open(final_output, "w") as outfile:
        for i in range(num_gpus):
            part_file = os.path.join(output_dir, f"result_part_{i}.jsonl")
            if os.path.exists(part_file):
                with open(part_file, "r") as infile:
                    for line in infile:
                        outfile.write(line)
    print("Done!")

if __name__ == "__main__":
    main()
