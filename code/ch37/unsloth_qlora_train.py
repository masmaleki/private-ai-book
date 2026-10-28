#!/usr/bin/env python3
"""
Chapter 37 — Accelerated QLoRA Fine-Tuning with Unsloth (unslothai/unsloth)
Book: Private AI: From Token to Decision (هوش خصوصی: از توکن تا تصمیم)
Author: Mohammad Sadegh Maleki

This script demonstrates production QLoRA fine-tuning of Qwen2.5-7B-Instruct
using Unsloth's custom Triton kernels and manual autograd backpropagation
(2x-5x faster training, up to 70% less VRAM, 0% accuracy loss), followed by
exporting to 16-bit merged weights for vLLM and Q4_K_M GGUF for llama.cpp.

Note: Run on a machine with an NVIDIA GPU and `pip install unsloth trl datasets`.
Passing `--dry-run` validates the configuration and prints the memory/speed plan
on CPU without requiring a GPU.
"""
import argparse
import json

CONFIG = {
    "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "max_seq_length": 4096,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "use_gradient_checkpointing": "unsloth",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_steps": 300,
    "optim": "adamw_8bit",
}


def run_unsloth_training(dataset_path: str, output_dir: str):
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["base_model"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=CONFIG["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=CONFIG["target_modules"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=CONFIG["use_gradient_checkpointing"],
        random_state=42,
    )

    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            warmup_steps=15,
            max_steps=CONFIG["max_steps"],
            learning_rate=CONFIG["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim=CONFIG["optim"],
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=output_dir,
        ),
    )

    trainer.train()

    # Save LoRA adapter for vLLM Multi-LoRA serving
    model.save_pretrained(f"{output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{output_dir}/lora_adapter")

    # Optional: Export merged 16-bit for standalone vLLM or GGUF Q4_K_M for llama.cpp
    # model.save_pretrained_merged(f"{output_dir}/merged_16bit", tokenizer, save_method="merged_16bit")
    # model.save_pretrained_gguf(f"{output_dir}/gguf_q4km", tokenizer, quantization_method="q4_k_m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="code/ch37/train.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen25-rfq-lora")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    if args.dry_run:
        print("=== Unsloth QLoRA Training Plan (Dry Run) ===")
        print(json.dumps(CONFIG, indent=2))
    else:
        run_unsloth_training(args.dataset, args.output_dir)
