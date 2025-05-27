pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_qlora_p4.yaml" distill_lora.py \
    --wandb_project=wsdm_qft_final \
    --run_name=gemma9b_soft \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/gemma9b \
    --model_path=billxbf/wsdm-gemma-ppt \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_gemma9b \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/gemma9b_soft \
    --train_split=soft_train \
    --val_split=soft_val \
    --epoch=1 \
    --lr=1e-4 \
    --bs=16 \
    --wd=0. \
    --bs_per_device=4 \
    --loss_weights="[0.0,0.0,0.25,0.25]" \
    --save_only_model=True \
    --use_lora \
    --seed=32

accelerate launch --config-file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_qlora_p4.yaml" distill_lora.py \
    --wandb_project=wsdm_qft_final \
    --run_name=gemma9b_soft_qft \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/gemma9b \
    --model_path=billxbf/wsdm-gemma-ppt \
    --lora_weights=/group-volume/binfeng/wsdm/ckpt/gemma9b_soft \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_gemma9b \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/gemma9b_soft_qft_v2 \
    --train_split=ft_train \
    --val_split=ft_val \
    --epoch=1 \
    --lr=2e-5 \
    --bs=8 \
    --wd=0. \
    --bs_per_device=2 \
    --loss_weights="[0.0,0.2,0.2,0.2]" \
    --save_only_model=True \
    --use_lora \
    --seed=32
