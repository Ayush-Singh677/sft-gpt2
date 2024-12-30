import os
import random
import torch
import numpy as np
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
)
from trl import SFTTrainer
from huggingface_hub import login
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(42)

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    login(args.hf_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(mode="disabled")

    train_dataset = load_dataset(args.dataset_name)['train']
    random_numbers = random.sample(range(len(train_dataset)), 500)
    eval_dataset = train_dataset.select(random_numbers)

    tokenizer = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="adamw_hf",
        save_strategy="no",
        logging_steps=50,
        learning_rate=5e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to=None,
        evaluation_strategy="steps",
        eval_steps=200,
        do_eval=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=512,
        dataset_text_field="text",
        processing_class=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()
    model.push_to_hub(f"{args.user_name}/gpt-sft")
    tokenizer.push_to_hub(f"{args.user_name}/gpt-sft")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model and push to Hugging Face Hub")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to fine-tune")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face API key")
    parser.add_argument("--user_name", type=str, required=True, help="Hugging Face username")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    args = parser.parse_args()

    main(args)
