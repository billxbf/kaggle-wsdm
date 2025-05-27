# pip install --upgrade pip
pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p4.yaml" distill.py \
    --wandb_project=wsdm_distill \
    --run_name=phi4_ft \
    --tokenizer_path=/group-volume/binfeng/wsdm/ckpt/phi4_ppt/checkpoint-529 \
    --model_path=/group-volume/binfeng/wsdm/ckpt/phi4_ppt/checkpoint-529 \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_phi4_distill \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_ft \
    --train_split=ft48k_train \
    --val_split=ft48k_val \
    --epoch=1 \
    --lr=1e-5 \
    --bs=8 \
    --wd=0. \
    --bs_per_device=4 \
    --save_only_model=True \
    --seed=32
