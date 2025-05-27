from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import random
import ast
import numpy as np
import argparse
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, log_loss
from accelerate import PartialState
from huggingface_hub import login
import torch.nn.functional as F
import wandb


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_STRING = PartialState().process_index

os.chdir("/group-volume/binfeng/wsdm/stage_final")


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DistillationTrainer(Trainer):
    def __init__(self, loss_weights, *args, T=1.0,  **kwargs):
        super().__init__(*args, **kwargs)
        self.T = T  # Temperature for distillation
        self.lw = loss_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        plabel_1 = inputs.pop("logits_qwencd_cali")
        plabel_2 = inputs.pop("logits_qwen32_cali")

        outputs = model(**inputs)
        logits = outputs.logits

        loss = distillation_loss(
            logits, labels, plabel_1, plabel_2, T=self.T, loss_weights=self.lw)

        if return_outputs:
            return loss, outputs
        return loss


class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        logits1 = torch.tensor([f.pop("logits_qwencd_cali") for f in features])
        logits2 = torch.tensor([f.pop("logits_qwen32_cali") for f in features])

        batch = super().__call__(features)
        batch["logits_qwencd_cali"] = logits1
        batch["logits_qwen32_cali"] = logits2
        return batch


def set_seeds(seed):
    """Set seeds for reproducibility """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_latest_checkpoint(directory):
    checkpoints = [d for d in os.listdir(
        directory) if d.startswith('checkpoint-')]
    if not checkpoints:
        return directory
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    return os.path.join(directory, checkpoints[-1])


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc}


def distillation_loss(logits, labels, plab1, plab2, T=1.0, loss_weights=[0.2, 0.2, 0.2, 0.1]):
    # Cross-entropy loss with ground truth labels
    loss_ce = F.cross_entropy(logits, labels)
    focal_loss_fn = FocalLoss(alpha=1, gamma=2, reduction='mean')
    loss_focal = focal_loss_fn(logits, labels)
    loss_kl1 = F.kl_div(
        F.log_softmax(logits / T, dim=1),
        F.softmax(plab1 / T, dim=1),
        reduction="batchmean"
    )
    loss_kl2 = F.kl_div(
        F.log_softmax(logits / T, dim=1),
        F.softmax(plab2 / T, dim=1),
        reduction="batchmean"
    )
    cos_loss1 = F.cosine_embedding_loss(
        F.softmax(logits / T, dim=1),
        F.softmax(plab1 / T, dim=1),
        torch.ones(logits.size(0)).to(logits.device)
    )
    cos_loss2 = F.cosine_embedding_loss(
        F.softmax(logits / T, dim=1),
        F.softmax(plab2 / T, dim=1),
        torch.ones(logits.size(0)).to(logits.device)
    )

    loss = loss_weights[0] * loss_ce + loss_weights[1] * loss_focal + loss_weights[2] * \
        (loss_kl1 + cos_loss1) + loss_weights[3] * (loss_kl2 + cos_loss2)
    return loss


def train(args):

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )

    args.loss_weights = ast.literal_eval(args.loss_weights)

    print(f'Loading dataset from {args.dataset_path} ...')
    dataset = load_from_disk(args.dataset_path)
    trainset = dataset[args.train_split]
    valset = dataset[args.val_split] if args.val_split else None

    print(f'Loading model and tokenizer ...')
    if os.path.exists(args.tokenizer_path):
        args.tokenizer_path = get_latest_checkpoint(args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = "right"

    print("Full Finetune!")
    if os.path.exists(args.model_path):
        args.model_path = get_latest_checkpoint(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f'Setting up trainer ...')
    training_args = TrainingArguments(
        # optimizing
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
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
        remove_unused_columns=False,
    )

    trainer = DistillationTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=valset,
        compute_metrics=compute_metrics,
        data_collator=CustomDataCollator(tokenizer=tokenizer),
        loss_weights=args.loss_weights,
        T=1.0,
    )

    print(f'Start training ...')
    trainer.train()


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
    parser.add_argument("--loss_weights", required=False,
                        type=str, default="[0.2,0.2,0.2,0.1]")
    parser.add_argument("--save_only_model", required=False,
                        type=bool, default=False)
    parser.add_argument("--seed", required=False, type=int, default=666)
    args = parser.parse_args()

    set_seeds(args.seed)
    train(args)


if __name__ == "__main__":
    main()
