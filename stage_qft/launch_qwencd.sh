pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

# accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
#     --wandb_project=wsdm_qft \
#     --run_name=qwencd_ppt \
#     --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/qwencd \
#     --model_path=Qwen/Qwen2.5-Coder-32B-Instruct \
#     --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_qwencd_final \
#     --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwencd_ppt \
#     --train_split=ppt_train \
#     --val_split=ppt_val \
#     --epoch=1 \
#     --lr=2e-5 \
#     --bs=16 \
#     --wd=0. \
#     --bs_per_device=2 \
#     --loss_weights="[0.5,0.5,0.0,0.0]" \
#     --save_only_model=True \
#     --seed=32

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_qft \
    --run_name=qwencd_ft \
    --tokenizer_path=/group-volume/binfeng/wsdm/ckpt/qwencd_ppt \
    --model_path=/group-volume/binfeng/wsdm/ckpt/qwencd_ppt \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_qwencd_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwencd_ft \
    --train_split=ft_train \
    --val_split=ft_val \
    --epoch=1 \
    --lr=1e-5 \
    --bs=32 \
    --wd=0. \
    --bs_per_device=4 \
    --loss_weights="[0.2,0.4,0.1,0.1]" \
    --save_only_model=True \
    --seed=32


