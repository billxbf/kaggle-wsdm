import torch
from transformers import GPTQConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, Gemma2ForCausalLM
import logging
import pandas as pd
import pickle
from datasets import load_from_disk
import os
os.chdir("/group-volume/binfeng/wsdm/stage_qft")

MODEL_PATH = "/group-volume/binfeng/wsdm/ckpt/qwen14b_dare_dslerp"
SAVE_PATH = "/group-volume/binfeng/wsdm/ckpt/q4/qwen14b_dare_dslerp_gptq_q4"
MAX_PROMPT_LENGTH = 400


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


data = pd.read_parquet(
    "/group-volume/binfeng/wsdm/stage_merge/qwen_ft_text.parquet").sample(n=32, random_state=2024)
data = data["text"].to_list()
gptq_config = GPTQConfig(bits=4, dataset=data, model_seqlen=3000, group_size=128,
                         use_cuda_fp16=True,
                         tokenizer=tokenizer)

merged_quantized_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    # num_labels=2,
    device_map='auto',
    quantization_config=gptq_config,
)


merged_quantized_model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
