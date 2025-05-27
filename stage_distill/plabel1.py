from huggingface_hub import login
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaModel, AutoModelForSequenceClassification
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import transformers
import torch
from datasets import Dataset, DatasetDict
from accelerate import PartialState
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import warnings
import random
from time import time
import re
import gc
from threading import Thread
import os
os.chdir("/group-volume/binfeng/wsdm/stage_distill")

tqdm.pandas()


MODEL_PATH = "/group-volume/binfeng/wsdm/ckpt/qwencd_ft/checkpoint-1476"
SAVE_NAME = "qwencd32b"
USE_DEVICES = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

MAX_LENGTH = 2000
MAX_PROMPT_LENGTH = 400
BATCH_SIZE = 4

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def split_col(col, idx):
    return col[idx]


def tokenize(tokenizer, texts):
    res = []
    for text in texts:
        input_ids = tokenizer(text)['input_ids']
        input_ids.append(tokenizer.eos_token_id)
        res.append(input_ids)
    return res


def format_text(tokenizer, prompt, response_a, response_b, max_len=2000, max_prompt_len=400, reverse=False, bidirect=False):

    enc_prompt, enc_response_a, enc_response_b = tokenizer.encode(
        prompt), tokenizer.encode(response_a), tokenizer.encode(response_b)
    max_len = max_len - 50  # leave space for special tokens
    if len(enc_prompt) + len(enc_response_a) + len(enc_response_b) > max_len:
        if len(enc_prompt) > max_prompt_len:
            enc_prompt = enc_prompt[:max_prompt_len] + \
                tokenizer.encode(" ... (truncated)")
        prompt_len, response_a_len, response_b_len = len(
            enc_prompt), len(enc_response_a), len(enc_response_b)
        # dynamic truncation to balance the length of responses
        trunc_a, trunc_b = 0, 0
        while prompt_len + response_a_len + response_b_len > max_len:
            if response_a_len > response_b_len:
                enc_response_a = enc_response_a[:-1]
                response_a_len -= 1
                trunc_a += 1
            else:
                enc_response_b = enc_response_b[:-1]
                response_b_len -= 1
                trunc_b += 1
        prompt, response_a, response_b = tokenizer.decode(enc_prompt), tokenizer.decode(
            enc_response_a), tokenizer.decode(enc_response_b)
        if trunc_a:
            response_a = response_a + f" ... (truncated {trunc_a} tokens)"
        if trunc_b:
            response_b = response_b + f" ... (truncated {trunc_b} tokens)"

    prompt_format = "<|User Prompt|>\n{prompt}\n\n<|Response A|>\n{response_a}\n\n<|Response B|>\n{response_b}\n\n<|Which response do you prefer?|>\n"
    if bidirect:
        return [prompt_format.format(prompt=prompt, response_a=response_a, response_b=response_b),
                prompt_format.format(prompt=prompt, response_a=response_b, response_b=response_a)]

    if not reverse:
        return prompt_format.format(prompt=prompt, response_a=response_a, response_b=response_b)
    else:
        return prompt_format.format(prompt=prompt, response_a=response_b, response_b=response_a)


def format_label(winner, reverse=False, bidirect=False):
    if bidirect:
        return [int(0) if winner == "model_a" else int(1),
                int(1) if winner == "model_a" else int(0)]
    if not reverse:
        return int(0) if winner == "model_a" else int(1)
    else:
        return int(1) if winner == "model_a" else int(0)


def process_df(data):
    data["tmp"] = data.apply(lambda x: format_text(tokenizer, x.prompt, x.response_a, x.response_b,
                                                   max_len=MAX_LENGTH, max_prompt_len=MAX_PROMPT_LENGTH,
                                                   bidirect=True), axis=1)
    data["text"] = data.apply(lambda x: split_col(x.tmp, 0), axis=1)
    data["text_reverse"] = data.apply(lambda x: split_col(x.tmp, 1), axis=1)
    data['text_len'] = data['text'].apply(lambda x: len(x.split(' ')))
    data["input_ids"] = tokenize(tokenizer, data["text"])
    data["input_ids_reverse"] = tokenize(tokenizer, data["text_reverse"])
    data = data.sort_values("text_len", ascending=False)
    data = data.drop(["tmp", "text_len"], axis=1)
    data = data.reset_index(drop=True)
    return data


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.input_ids = df["input_ids"].tolist()
        self.input_ids_reverse = df["input_ids_reverse"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return both normal and reversed input_ids
        if isinstance(idx, int):
            return {
                "input_ids": self.input_ids[idx],
                "input_ids_reverse": self.input_ids_reverse[idx]
            }
        else:
            return {
                "input_ids": [self.input_ids[i] for i in idx],
                "input_ids_reverse": [self.input_ids_reverse[i] for i in idx]
            }


def load_model(device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    input_ids_reverse = [b["input_ids_reverse"] for b in batch]

    # Pad both normal and reversed sequences
    normal_batch = pad_without_fast_tokenizer_warning(
        tokenizer,
        {"input_ids": input_ids},
        padding="longest",
        pad_to_multiple_of=None,
        return_tensors="pt",
    )

    reverse_batch = pad_without_fast_tokenizer_warning(
        tokenizer,
        {"input_ids": input_ids_reverse},
        padding="longest",
        pad_to_multiple_of=None,
        return_tensors="pt",
    )

    return {
        "normal": normal_batch,
        "reverse": reverse_batch
    }


def inference_parallel(df, use_devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']):
    models = []
    for device in use_devices:
        print(f"Loading model on {device}")
        models.append(load_model(device))

    df = df.reset_index(drop=True)
    df['fold'] = [i % len(use_devices) for i in range(len(df))]

    results = []

    def run_inference_thread(sub_df, model, device):
        dataset = CustomDataset(sub_df, tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1
        )

        all_logits_normal = []
        all_logits_reverse = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Device {device}"):
                # Process normal order
                normal_batch = {k: v.to(device)
                                for k, v in batch["normal"].items()}
                outputs_normal = model(**normal_batch)
                logits_normal = outputs_normal.logits.float()
                all_logits_normal.append(logits_normal.cpu().numpy())

                # Process reverse order
                reverse_batch = {k: v.to(device)
                                 for k, v in batch["reverse"].items()}
                outputs_reverse = model(**reverse_batch)
                logits_reverse = outputs_reverse.logits.float()
                all_logits_reverse.append(logits_reverse.cpu().numpy())

        all_logits_normal = np.concatenate(all_logits_normal, axis=0)
        all_logits_reverse = np.concatenate(all_logits_reverse, axis=0)

        # For reverse order, we need to swap the logits before averaging
        swapped_logits_reverse = np.column_stack([
            all_logits_reverse[:, 1],  # swap columns
            all_logits_reverse[:, 0]
        ])

        # Average the logits
        averaged_logits = (all_logits_normal + swapped_logits_reverse) / 2

        sub_df['logits_model_a'] = averaged_logits[:, 0]
        sub_df['logits_model_b'] = averaged_logits[:, 1]

        results.append(sub_df)

    threads = []
    for idx, device in enumerate(use_devices):
        sub_df = df[df['fold'] == idx].copy()
        thread = Thread(target=run_inference_thread,
                        args=(sub_df, models[idx], device))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    final_df = pd.concat(results, axis=0)
    final_df = final_df.sort_index()
    return final_df


print("processing data ...")
# ppt = pd.read_csv("/user-volume/bx/ppt127k.csv")
kaggle = pd.read_csv("/user-volume/bx/kaggle48k.csv")
# lmsys = pd.read_parquet("/user-volume/bx/lmsys61k.parquet")
# hfopen = pd.read_parquet(
#     "/group-volume/binfeng/wsdm/stage_qft/data/hfopen26k_unlabel.parquet")


# print("inferencing hfopen set ... ")
# hfopen = process_df(hfopen)
# res_hfopen = inference_parallel(hfopen, use_devices=USE_DEVICES)
# res_hfopen.to_parquet(
#     f"/group-volume/binfeng/wsdm/stage_qft/plabels/{SAVE_NAME}_plabel_hfopen.parquet")


# print("inferencing lmsys set ... ")
# lmsys = process_df(lmsys)
# res_lmsys = inference_parallel(lmsys, use_devices=USE_DEVICES)
# res_lmsys.to_parquet(
#     f"/group-volume/binfeng/wsdm/stage_qft/plabels/{SAVE_NAME}_plabel_lmsys.parquet")


print("inferencing kaggle set ... ")
kaggle = process_df(kaggle)
res_kaggle = inference_parallel(kaggle, use_devices=USE_DEVICES)
res_kaggle.to_parquet(
    f"/group-volume/binfeng/wsdm/stage_qft/plabels/{SAVE_NAME}_plabel_kaggle.parquet")
# print("inferencing ppt set ... ")
# res_ppt = inference_parallel(ppt, use_devices=USE_DEVICES)
# res_ppt.to_csv(
#     f"/group-volume/binfeng/wsdm/stage_distill/plabels/{SAVE_NAME}_plabel_ppt.csv", index=False)
