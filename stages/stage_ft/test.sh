# pip install --upgrade pip
pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p4.yaml" sft.py \
    --wandb_project=wsdm_evo \
    --run_name=qwen32b_ft \
    --tokenizer_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_ppt/checkpoint-1973 \
    --model_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_ppt/checkpoint-1973 \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_qwen32b_ft \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_ft \
    --train_split=kaggle48k_train \
    --val_split=kaggle48k_val \
    --epoch=1 \
    --lr=2e-5 \
    --bs=8 \
    --wd=1e-4 \
    --bs_per_device=1 \
    --save_only_model=True \
    --seed=32

