pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

# accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p4.yaml" distill.py \
#     --wandb_project=wsdm_final \
#     --run_name=qwencd14_ppt \
#     --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/qwencd14b \
#     --model_path=Qwen/Qwen2.5-Coder-14B-Instruct \
#     --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_qwencd14b_final \
#     --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwencd14b_ppt \
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
    --run_name=qwencd14_ft_v3 \
    --tokenizer_path=/group-volume/binfeng/wsdm/ckpt/qwencd14b_ppt \
    --model_path=/group-volume/binfeng/wsdm/ckpt/qwencd14b_ppt \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_qwencd14b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwencd14b_ft_v3 \
    --train_split=ft_train \
    --val_split=ft_val \
    --epoch=1 \
    --lr=1e-5 \
    --bs=8 \
    --wd=0. \
    --bs_per_device=4 \
    --loss_weights="[0.1,0.4,0.2,0.05]" \
    --save_only_model=True \
    --seed=32
