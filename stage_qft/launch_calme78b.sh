pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_qlora_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_qft \
    --run_name=calme78_ppt_v4 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/calme78b \
    --model_path=/group-volume/binfeng/hf_models/calme3.2-78b/ \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_calme78b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/calme78b_ppt \
    --train_split=ppt_train \
    --val_split=ppt_val \
    --epoch=1 \
    --lr=2e-4 \
    --bs=32 \
    --wd=0. \
    --bs_per_device=4 \
    --loss_weights="[0.5,0.5,0.0,0.0]" \
    --save_only_model=True \
    --use_lora \
    --seed=42

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_qlora_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_qft \
    --run_name=calme78_ft_v4 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/calme78b \
    --model_path=/group-volume/binfeng/hf_models/calme3.2-78b/ \
    --lora_weights=/group-volume/binfeng/wsdm/ckpt/calme78b_ppt \
    --dataset_path=/group-volume/binfeng/wsdm/stage_qft/dataset/tokenized_calme78b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/calme78b_ft \
    --train_split=ft_train \
    --val_split=ft_val \
    --epoch=2 \
    --lr=1e-4 \
    --bs=8 \
    --wd=0. \
    --bs_per_device=1 \
    --loss_weights="[0.5,0.5,0.0,0.0]" \
    --save_only_model=True \
    --use_lora \
    --seed=42
