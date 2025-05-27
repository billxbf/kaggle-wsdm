pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

# accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p4.yaml" distill.py \
#     --wandb_project=wsdm_final \
#     --run_name=gemma9_ppt \
#     --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/gemma9b \
#     --model_path=google/gemma-2-9b-it \
#     --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_gemma9b_final \
#     --model_save_path=/group-volume/binfeng/wsdm/ckpt/gemma9b_ppt \
#     --train_split=ppt_train \
#     --val_split=ppt_val \
#     --epoch=1 \
#     --lr=2e-5 \
#     --bs=32 \
#     --wd=1e-4 \
#     --bs_per_device=4 \
#     --loss_weights="[0.1,0.2,0.2,0.15]" \
#     --save_only_model=True \
#     --seed=32


accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p4.yaml" distill.py \
    --wandb_project=wsdm_final \
    --run_name=gemma9_ft_v2 \
    --tokenizer_path=/group-volume/binfeng/wsdm/ckpt/gemma9b_ppt \
    --model_path=/group-volume/binfeng/wsdm/ckpt/gemma9b_ppt \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_gemma9b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/gemma9b_ft_v2 \
    --train_split=ft_train \
    --val_split=ft_val \
    --epoch=1 \
    --lr=1e-5 \
    --bs=8 \
    --wd=0. \
    --bs_per_device=4 \
    --loss_weights="[0.1,0.4,0.15,0.1]" \
    --save_only_model=True \
    --seed=32
