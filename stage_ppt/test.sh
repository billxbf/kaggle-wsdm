# pip install --upgrade pip
pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/fsdp_config_p4.yaml" sft.py \
    --wandb_project=wsdm_evo \
    --run_name=test \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/qwen32b \
    --model_path=Qwen/Qwen2.5-32B-Instruct \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_qwen32b_ppt \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_ppt \
    --train_split=ppt127k_train \
    --val_split=ppt127k_val \
    --epoch=1 \
    --lr=1e-5 \
    --bs=4 \
    --wd=1e-3 \
    --bs_per_device=1 \
    --save_only_model=True

