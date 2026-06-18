"""Model creation, training helpers, and inference utilities."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

from .evaluation import compute_rouge


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str, device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device is None:
        device = get_device()
    model = model.to(device)
    return model, tokenizer


def apply_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list[str]] = None,
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    model.enable_input_require_grads()
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)


def tokenize_dataset(
    df,
    tokenizer,
    prompt_col: str = "prompt",
    answer_col: str = "output",
    max_input: int = 256,
    max_target: int = 512,
    is_test: bool = False,
):
    def tokenize_fn(examples):
        inputs = tokenizer(
            examples[prompt_col],
            max_length=max_input,
            truncation=True,
            padding=False,
        )
        if not is_test:
            targets = tokenizer(
                text_target=examples[answer_col],
                max_length=max_target,
                truncation=True,
                padding=False,
            )
            inputs["labels"] = [
                [(t if t != tokenizer.pad_token_id else -100) for t in seq]
                for seq in targets["input_ids"]
            ]
        return inputs

    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset.map(tokenize_fn, batched=True, remove_columns=hf_dataset.column_names)


def build_training_args(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    fp16: bool = False,
    bf16: bool = False,
    eval_strategy: str = "epoch",
    save_strategy: str = "no",
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "rougeL",
    predict_with_generate: bool = True,
    generation_max_length: int = 512,
    generation_num_beams: int = 4,
    seed: int = 42,
    **kwargs,
):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        bf16=bf16,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        predict_with_generate=predict_with_generate,
        generation_max_length=generation_max_length,
        generation_num_beams=generation_num_beams,
        report_to="none",
        seed=seed,
        **kwargs,
    )


def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100 in labels (padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        scores = compute_rouge(decoded_preds, decoded_labels)
        return {
            "rouge1": scores["rouge1_f1"],
            "rougeL": scores["rougeL_f1"],
        }
    return compute_metrics


def generate_predictions(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 16,
    max_input_length: int = 256,
    max_new_tokens: int = 512,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    device=None,
) -> list[str]:
    if device is None:
        device = get_device()
    model.eval()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    predictions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend([p.strip() for p in decoded])
        if (i // batch_size) % 10 == 0:
            print(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return predictions
