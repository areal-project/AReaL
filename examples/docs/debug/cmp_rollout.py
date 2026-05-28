import base64
from io import BytesIO

import datasets
import requests
import torch
from PIL.Image import Image as ImageObject
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def image2base64(images: list[ImageObject] | ImageObject) -> list[str] | str:
    if isinstance(images, ImageObject):
        images = [images]

    byte_images = []
    for image in images:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            buffer.seek(0)
            byte_image = base64.b64encode(buffer.read()).decode("utf-8")
            byte_images.append(byte_image)

    return byte_images


model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_id, dtype=torch.bfloat16
).to("cuda")
model.eval()

processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

dataset = datasets.load_dataset("BUAADreamer/clevr_count_70k", split="train")


t_count = 0
s_count = 0

for idx, sample in enumerate(dataset):
    answer = int(sample["answer"])
    image = sample["images"][0]

    # Apply a chat template
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Solve the following question: count the number of items in the image and provide the final answer in [ ] format, ensuring that only the number is inside the brackets without any additional text or explanations.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "How many items are there in the image?"},
            ],
        },
    ]

    # Using the same inputs for both implementations
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[-1]

    # Run the generation with Transformers locally on GPUs
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        generation = generation[0][input_len:]

    hf_results = processor.decode(generation, skip_special_tokens=True)

    # Send the request to the inference server
    response = requests.post(
        "http://127.0.0.1:30000/generate",
        json={
            "input_ids": input_ids[0].tolist(),
            "image_data": image2base64(image),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
            },
        },
    )
    sglang_results = processor.decode(
        response.json()["output_ids"], skip_special_tokens=True
    )

    if f"[{answer}]" == hf_results:
        t_count += 1
    if f"[{answer}]" == sglang_results:
        s_count += 1

print(f"Transformers accuracy: {t_count / len(dataset) * 100:.2f}%")
print(f"SGLang       accuracy: {s_count / len(dataset) * 100:.2f}%")
