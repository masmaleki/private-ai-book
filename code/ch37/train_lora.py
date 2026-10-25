"""train_lora.py - LoRA / QLoRA fine-tuning with PEFT + TRL (SFTTrainer).

CPU demo : python train_lora.py --model HuggingFaceTB/SmolLM2-135M-Instruct
GPU QLoRA: python train_lora.py --model <7B-instruct> --qlora --epochs 2
Data: train.jsonl / eval.jsonl in prompt/completion chat format (make_dataset.py).
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
p.add_argument("--out", default="out/adapter")
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--qlora", action="store_true", help="4-bit base (needs CUDA)")
args = p.parse_args()

torch.manual_seed(42)
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# --- 1) load the frozen base model (optionally 4-bit for QLoRA) ---
model_kwargs = {}
if args.qlora:
    from transformers import BitsAndBytesConfig
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4
        bnb_4bit_use_double_quant=True,     # also quantize the scales
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_kwargs["device_map"] = "auto"
model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

# --- 2) LoRA config: which layers get a low-rank side branch ---
lora = LoraConfig(
    r=args.rank,
    lora_alpha=args.alpha,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# --- 3) data ---
ds = load_dataset("json", data_files={"train": "train.jsonl",
                                      "eval": "eval.jsonl"})

# --- 4) training config ---
cfg = SFTConfig(
    output_dir="out/run",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=2,
    logging_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    max_length=256,
    completion_only_loss=True,   # loss only on the assistant answer
    bf16=args.qlora,             # CPU demo stays in fp32
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                     eval_dataset=ds["eval"], processing_class=tok,
                     peft_config=lora)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model(args.out)   # saves ONLY the adapter (a few MB)
tok.save_pretrained(args.out)
print("adapter saved to", args.out)
