pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

# accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p4.yaml" distill.py \
#     --wandb_project=wsdm_final \
#     --run_name=phi4_ppt \
#     --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/phi4 \
#     --model_path=microsoft/phi-4 \
#     --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_phi4_final \
#     --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_ppt \
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
    --run_name=phi4_ft_v2 \
    --tokenizer_path=/group-volume/binfeng/wsdm/ckpt/phi4_ppt \
    --model_path=/group-volume/binfeng/wsdm/ckpt/phi4_ppt \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_phi4_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_ft_v2 \
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
