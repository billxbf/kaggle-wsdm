pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

# accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
#     --wandb_project=wsdm_final \
#     --run_name=phi4_soft \
#     --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/phi4 \
#     --model_path=billxbf/wsdm-phi4-ppt \
#     --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_phi4 \
#     --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft \
#     --train_split=soft_train \
#     --val_split=soft_val \
#     --epoch=1 \
#     --lr=1e-5 \
#     --bs=32 \
#     --wd=0. \
#     --bs_per_device=4 \
#     --loss_weights="[0.0,0.0,0.5,0.5]" \
#     --save_only_model=True \
#     --seed=42


accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=phi4_soft_ft_r1_fold0 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/phi4 \
    --model_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_phi4 \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft_ft_r1/fold0 \
    --train_split=ft_train_fold0 \
    --val_split=ft_val_fold0 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.15,0.15,0.2,0.2,0.3]" \
    --save_only_model=True \
    --seed=10




accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=phi4_soft_ft_r1_fold1 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/phi4 \
    --model_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_phi4 \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft_ft_r1/fold1 \
    --train_split=ft_train_fold1 \
    --val_split=ft_val_fold1 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.15,0.15,0.2,0.2,0.3]" \
    --save_only_model=True \
    --seed=20





accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=phi4_soft_ft_r1_fold2 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/phi4 \
    --model_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_phi4 \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft_ft_r1/fold2 \
    --train_split=ft_train_fold2 \
    --val_split=ft_val_fold2 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.15,0.15,0.2,0.2,0.3]" \
    --save_only_model=True \
    --seed=30




accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=phi4_soft_ft_r1_fold3 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/phi4 \
    --model_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_phi4 \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft_ft_r1/fold3 \
    --train_split=ft_train_fold3 \
    --val_split=ft_val_fold3 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.15,0.15,0.2,0.2,0.3]" \
    --save_only_model=True \
    --seed=40


    

accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=phi4_soft_ft_r1_fold4 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/phi4 \
    --model_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_phi4 \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_soft_ft_r1/fold4 \
    --train_split=ft_train_fold4 \
    --val_split=ft_val_fold4 \
    --epoch=1 \
    --lr=8e-6 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.15,0.15,0.2,0.2,0.3]" \
    --save_only_model=True \
    --seed=50