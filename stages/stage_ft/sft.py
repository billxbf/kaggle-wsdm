from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import random
import numpy as np
import argparse
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_from_disk
from peft import PeftModel, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, log_loss
from accelerate import PartialState
from huggingface_hub import login
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_STRING = PartialState().process_index

os.chdir("/group-volume/binfeng/wsdm/stage_ft")

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules="all-linear",
    bias="none",
    lora_dropout=0.05,
    task_type=TaskType.SEQ_CLS,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)


def set_seeds(seed):
    """Set seeds for reproducibility """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc}


def train(args):

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,  # Change this to your project name
        name=args.run_name,        # Will be added to args
        config=vars(args)          # Track hyperparameters
    )

    print(f'Loading dataset from {args.dataset_path} ...')
    dataset = load_from_disk(args.dataset_path)
    trainset = dataset[args.train_split]
    valset = dataset[args.val_split] if args.val_split else None
    if args.debug == True:
        print("DEBUG MODE!", args.debug)
        trainset = trainset.select(range(1000))
        valset = valset.select(range(10))

    print(f'Loading model and tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = "right"

    if args.use_qlora:
        print("USING QLORA!")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=2,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
        if args.resume_from_checkpoint:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
            model = PeftModel.from_pretrained(
                model,
                args.resume_from_checkpoint,
                is_trainable=True,
            )
        else:
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        for name, param in model.named_parameters():
            if "score" in name:
                print("Found score layer, setting to trainable.", name)
                param.requires_grad = True
        model.print_trainable_parameters()
    else:
        print("Full Finetune!")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    print(f'Setting up trainer ...')
    training_args = TrainingArguments(
        # optimizing
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.bs_per_device,
        gradient_accumulation_steps=args.bs // args.bs_per_device,
        weight_decay=args.wd,
        # logging
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=0.1,
        logging_steps=0.01,
        output_dir=args.model_save_path,
        # misc
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        save_only_model=args.save_only_model,
        report_to="wandb",
        run_name=args.run_name,
        label_names=['labels'],
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=valset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print(f'Start training ...')
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)


def main():
    parser = argparse.ArgumentParser()
    # Add wandb related arguments
    parser.add_argument("--run_name", required=False, type=str,
                        default=None, help="Name for the wandb run")
    parser.add_argument("--wandb_project", required=False, type=str,
                        default="default", help="WandB project name")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--val_split", required=False, type=str, default=None)
    parser.add_argument("--epoch", required=False, type=int, default=1)
    parser.add_argument("--lr", required=False, type=float, default=5e-6)
    parser.add_argument("--bs", required=False, type=int, default=64)
    parser.add_argument("--bs_per_device", required=False, type=int, default=4)
    parser.add_argument("--wd", required=False, type=float, default=0.0)
    parser.add_argument("--resume_from_checkpoint",
                        required=False, type=str, default=None)
    parser.add_argument("--debug", required=False, type=bool, default=False)
    parser.add_argument("--use_lora", required=False, type=bool, default=False)
    parser.add_argument("--use_qlora", required=False,
                        type=bool, default=False)
    parser.add_argument('--lora_path', type=str,
                        default='none', help="lora path")
    parser.add_argument("--save_only_model", required=False,
                        type=bool, default=False)
    parser.add_argument("--seed", required=False, type=int, default=666)
    args = parser.parse_args()

    set_seeds(args.seed)
    train(args)


if __name__ == "__main__":
    main()
