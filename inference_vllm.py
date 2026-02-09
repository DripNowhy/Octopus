from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image

prompt_suffix = """\n\nYou first think through your reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \\boxed{}. If you believe the answer can be further enhanced, generate <self-correction> </self-correction> tags enclosed with no content, and regenerate a new reasoning process and a new answer from scratch after that. The new response should first think through your reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \\boxed{}. All reasoning, answer steps must be included without omission."""

MODEL_PATH = "Tuwhy/Octopus-8B"

def main():
    # Initialize model
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        seed=1,
        max_model_len=8192 * 8,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        max_pixels=1280*28*28,
        min_pixels=256*28*28
    )

    # Single case
    prompt = "The accuracy gap between the Octopus-8B and the Qwen3-8B-VL-Thinking model is?"
    image_path = "./assets/head.png"

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        max_tokens=8192*2
    )

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt + prompt_suffix}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Prepare input
    inputs = {
        "prompt": text_prompt,
        "multi_modal_data": {
            "image": image
        }
    }

    # Generate
    outputs = llm.generate([inputs], sampling_params=sampling_params)

    # Print result
    generated_text = outputs[0].outputs[0].text

    print("Generated response:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)

if __name__ == '__main__':
    main()
