from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

prompt_suffix = """\n\nYou first think through your reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \\boxed{}. If you believe the answer can be further enhanced, generate <self-correction> </self-correction> tags enclosed with no content, and regenerate a new reasoning process and a new answer from scratch after that. The new response should first think through your reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \\boxed{}. All reasoning, answer steps must be included without omission."""

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Tuwhy/Octopus-8B", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Tuwhy/Octopus-8B")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./assets/head.png",
            },
            {"type": "text", "text": "The accuracy gap between the Octopus-8B and the Qwen3-8B-VL-Thinking model is?" + prompt_suffix},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=8192*2)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
