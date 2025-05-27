# pip install --upgrade pip
pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" sft.py \
    --wandb_project=wsdm_evo \
    --run_name=gemma27b_ft \
    --tokenizer_path=/group-volume/binfeng/wsdm/ckpt/gemma27b_lbs/checkpoint-493 \
    --model_path=/group-volume/binfeng/wsdm/ckpt/gemma27b_lbs/checkpoint-493 \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_gemma27b_ft \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/gemma27b_ft \
    --train_split=kaggle48k_train \
    --val_split=kaggle48k_val \
    --epoch=1 \
    --lr=2e-5 \
    --bs=8 \
    --wd=0. \
    --bs_per_device=2 \
    --save_only_model=True \
    --seed=32

