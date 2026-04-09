import os
import argparse
import pickle
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType
from optimizers import MaskedAdamW

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized HF Trainer script (LoRA & Full SFT)")

    # Core input/output paths
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Training mode switch (LoRA vs Full SFT)
    parser.add_argument("--finetune_mode", type=str, default="full", choices=["lora", "full"], help="Choose fine-tuning mode")
    
    # Core training hyperparameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--data_limit", type=int, default=-1)
    parser.add_argument("--eval_ratio", type=float, default=0.01)

    # LoRA parameters (used only when finetune_mode="lora")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimizer and mask configuration
    parser.add_argument("--optimizer", type=str, default="masked_adamw", choices=["adamw", "masked_adamw"])
    parser.add_argument("--mask_source", type=str, default="file", choices=["none", "file", "random"])
    parser.add_argument("--mask_file_path", type=str)
    parser.add_argument("--mask_random_file_path", type=str)
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--random_select_ratio", type=float, default=0.1)
    parser.add_argument("--force_train_bias_norm", action="store_true", default=True)

    # Precision configuration
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    if args.mask_source == "file" and not args.mask_file_path:
        parser.error("--mask_file_path is required when --mask_source file is used")
    if args.mask_source == "random" and not args.mask_random_file_path:
        parser.error("--mask_random_file_path is required when --mask_source random is used")

    return args


def print_trainable_ratio(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable / total:.2%})")


def guess_text(example):
    """Best-effort parsing across common supervised fine-tuning formats."""
    if not isinstance(example, dict):
        return ""

    if "messages" in example and isinstance(example["messages"], list):
        return "\n".join([f"{str(m.get('role', 'user')).capitalize()}: {str(m.get('content', '')).strip()}" for m in example["messages"] if str(m.get('content', '')).strip()])
    
    if "conversations" in example and isinstance(example["conversations"], list):
        return "\n".join([f"{str(m.get('from', m.get('role', 'user'))).capitalize()}: {str(m.get('value', m.get('content', ''))).strip()}" for m in example["conversations"] if str(m.get('value', m.get('content', ''))).strip()])

    if "instruction" in example and "output" in example:
        ins, inp, out = str(example.get("instruction", "")).strip(), str(example.get("input", "")).strip(), str(example.get("output", "")).strip()
        return f"Instruction: {ins}\nInput: {inp}\nAnswer: {out}" if inp else f"Instruction: {ins}\nAnswer: {out}"

    if "prompt" in example and "response" in example:
        return f"User: {str(example['prompt']).strip()}\nAssistant: {str(example['response']).strip()}"

    if "question" in example and "answer" in example:
        return f"Question: {str(example['question']).strip()}\nAnswer: {str(example['answer']).strip()}"

    return str(example.get("text", ""))


def build_tokenize_fn(tokenizer, max_length):
    def _tokenize(example):
        text = guess_text(example) or ""
        tokens = tokenizer(text, truncation=True, max_length=max_length, padding=False)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    return _tokenize


def generate_masks(model, args):
    """Build gradient masks for MaskedAdamW."""
    print(f"========== Mask Config ==========\nSource: {args.mask_source} | Invert: {args.invert_mask} | Train Bias/Norm: {args.force_train_bias_norm}\n===============================")
    
    gradient_masks = {}
    device = next(model.parameters()).device
    targets_by_name = {}

    if args.mask_source in ["file", "random"]:
        path = args.mask_file_path if args.mask_source == "file" else args.mask_random_file_path
        if not path or not os.path.exists(path):
            raise ValueError(f"Invalid mask file path: {path}")
        with open(path, "rb") as f:
            targets_by_name = pickle.load(f)
        print(f"Loaded targets: {len(targets_by_name)} layers")

    total_params_count = 0
    trainable_elements = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        numel = param.numel()
        total_params_count += numel
        
        if args.mask_source == "none":
            base_mask = torch.ones_like(param.data, device=device)
        else:
            base_mask = torch.zeros_like(param.data, device=device)

        if args.mask_source == "file" and name in targets_by_name:
            target_indices_1d = targets_by_name[name]
            if target_indices_1d is not None and len(target_indices_1d) > 0:
                unraveled = np.unravel_index(target_indices_1d, param.data.shape)
                idx_tensors = tuple(torch.as_tensor(idx, device=device, dtype=torch.long) for idx in unraveled)
                base_mask[idx_tensors] = 1.0

        elif args.mask_source == "random":
            num_selected = int(numel * args.random_select_ratio)
            if num_selected > 0:
                mask_flat = torch.zeros(numel, device=device, dtype=param.dtype)
                current_selected = 0
                
                if name in targets_by_name and targets_by_name[name] is not None:
                    candidate = torch.as_tensor(targets_by_name[name], device=device, dtype=torch.long)
                    candidate = candidate[(candidate >= 0) & (candidate < numel)]
                    if candidate.numel() > 0:
                        k = min(num_selected, candidate.numel())
                        chosen = candidate[torch.randperm(candidate.numel(), device=device)[:k]]
                        mask_flat[chosen] = 1.0
                        current_selected = k

                remaining = num_selected - current_selected
                if remaining > 0:
                    available = torch.where(mask_flat == 0)[0]
                    if available.numel() > 0:
                        k = min(remaining, available.numel())
                        chosen2 = available[torch.randperm(available.numel(), device=device)[:k]]
                        mask_flat[chosen2] = 1.0
                
                base_mask = mask_flat.view_as(param.data)

        final_mask = 1.0 - base_mask if args.invert_mask else base_mask
        if args.force_train_bias_norm and any(kw in name.lower() for kw in ["bias", "norm", "ln_"]):
            final_mask = torch.ones_like(param.data, device=device)

        gradient_masks[name] = final_mask
        trainable_elements += torch.sum(final_mask).item()

    print(f"Mask done. Total: {total_params_count:,}, Trainable elems: {int(trainable_elements):,} ({trainable_elements/max(1, total_params_count):.2%})")
    return gradient_masks


def main():
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 cannot be enabled simultaneously")

    # 1) Basic setup
    set_seed(args.seed)

    # Build output save path
    model_name_short = os.path.basename(os.path.normpath(args.model_path))
    method_str = f"lora_r{args.lora_rank}" if args.finetune_mode == "lora" else "full"
    time_str = datetime.now().strftime("%m%d_%H%M")
    save_path = os.path.join(args.output_dir, f"{model_name_short}_{method_str}_{args.optimizer}_{time_str}")
    os.makedirs(save_path, exist_ok=True)
    print(f"Training output dir: {save_path}")
    print(f"Training Mode: {'LoRA' if args.finetune_mode == 'lora' else 'Full Fine-Tuning (SFT)'}")

    # 2) Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    # 3) Switch between LoRA mode and full SFT mode
    if args.finetune_mode == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear", 
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print("LoRA enabled with 'all-linear' targeting.")
    else:
        print("Running Full Fine-Tuning (SFT).")

    model.enable_input_require_grads()
    print_trainable_ratio(model)

    # 4) Dataset preparation (use num_proc=1 to avoid fork-related OOM/crash)
    ds = hf_load_dataset("json", data_files=args.dataset_path, split="train")
    if args.data_limit > 0 and len(ds) > args.data_limit:
        ds = ds.shuffle(seed=args.seed).select(range(args.data_limit))

    if args.eval_ratio > 0 and len(ds) > 10:
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    tokenize_fn = build_tokenize_fn(tokenizer, args.max_length)
    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names, num_proc=1) 
    if eval_ds is not None:
        eval_ds = eval_ds.map(tokenize_fn, remove_columns=eval_ds.column_names, num_proc=1)

    # 5) Optimizer and Trainer
    optimizer = None
    if args.optimizer == "masked_adamw":
        masks = generate_masks(model, args) if args.mask_source != "none" else None
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_map = {p: n for n, p in model.named_parameters()}
        optimizer = MaskedAdamW(trainable_params, lr=args.learning_rate, masks=masks, param_map=param_map)

    train_args = TrainingArguments(
        output_dir=save_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps if eval_ds is not None else None,
        save_total_limit=1,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        dataloader_num_workers=0,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
    )

    trainer.train()
    
    # 6) Save model (LoRA and full SFT paths differ)
    if args.finetune_mode == "lora":
        trainer.model.save_pretrained(save_path)
    else:
        trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Training finished. Model saved to: {save_path}")

if __name__ == "__main__":
    main()