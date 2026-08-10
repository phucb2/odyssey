"""Demo: caption / VQA with CLIP + Q-Former + Qwen3 VLM.

Run: ``uv run python -m odyssey.experiments.minigpt.vlm_demo --image path/to.jpg``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from odyssey.experiments.minigpt.vlm import MiniGPT, MiniGPTConfig

DEFAULT_PROMPT = "Describe this image in one sentence."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniGPT VLM demo")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="User question or instruction")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional Q-Former + projector checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg = MiniGPTConfig()
    model = MiniGPT.from_pretrained(cfg, device=device)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_trainable_state_dict(state)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name)
    image = Image.open(args.image).convert("RGB")
    pixel_values = model.vision.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generated_ids = model.generate(
        pixel_values,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
    )

    prompt_len = input_ids.shape[1]
    output_ids = generated_ids[0, prompt_len:].tolist()
    print(tokenizer.decode(output_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
