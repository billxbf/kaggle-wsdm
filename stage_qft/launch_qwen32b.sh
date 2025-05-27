pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb


accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=qwen32b_soft \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/qwen32b \
    --model_path=/group-volume/binfeng/hf_models/Qwen2.5-32B-Instruct/ \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_qwen32b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft \
    --train_split=soft_train \
    --val_split=soft_val \
    --epoch=1 \
    --lr=2e-5 \
    --bs=32 \
    --wd=0. \
    --bs_per_device=4 \
    --loss_weights="[0.0,0.0,0.5,0.5]" \
    --save_only_model=True \
    --seed=42



accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=qwen32b_soft_ft_fold0 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/qwen32b \
    --model_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_qwen32b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft_ft/fold0 \
    --train_split=ft_train_fold0 \
    --val_split=ft_val_fold0 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.25,0.25,0.25,0.25]" \
    --save_only_model=True \
    --seed=1




accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=qwen32b_soft_ft_fold1 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/qwen32b \
    --model_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_qwen32b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft_ft/fold1 \
    --train_split=ft_train_fold1 \
    --val_split=ft_val_fold1 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.25,0.25,0.25,0.25]" \
    --save_only_model=True \
    --seed=2





accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=qwen32b_soft_ft_fold2 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/qwen32b \
    --model_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_qwen32b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft_ft/fold2 \
    --train_split=ft_train_fold2 \
    --val_split=ft_val_fold2 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.25,0.25,0.25,0.25]" \
    --save_only_model=True \
    --seed=3




accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=qwen32b_soft_ft_fold3 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/qwen32b \
    --model_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_qwen32b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft_ft/fold3 \
    --train_split=ft_train_fold3 \
    --val_split=ft_val_fold3 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.25,0.25,0.25,0.25]" \
    --save_only_model=True \
    --seed=4


    

accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=qwen32b_soft_ft_fold4 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/qwen32b \
    --model_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_qwen32b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/qwen32b_soft_ft/fold4 \
    --train_split=ft_train_fold4 \
    --val_split=ft_val_fold4 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.25,0.25,0.25,0.25]" \
    --save_only_model=True \
    --seed=5