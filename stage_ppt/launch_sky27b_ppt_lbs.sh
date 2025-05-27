# pip install --upgrade pip
pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" sft.py \
    --wandb_project=wsdm_evo \
    --run_name=sky27b_ppt_lbs \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/sky27b \
    --model_path=Skywork/Skywork-Reward-Gemma-2-27B-v0.2 \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_sky27b_ppt \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/sky27b_lbs \
    --train_split=ppt127k_train \
    --val_split=ppt127k_val \
    --epoch=1 \
    --lr=1e-5 \
    --bs=64 \
    --wd=1e-4 \
    --bs_per_device=2 \
    --save_only_model=True \
    --seed=69
