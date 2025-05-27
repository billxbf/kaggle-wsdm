pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_qlora_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=calme78_ppt_v2 \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/calme78b \
    --model_path=/group-volume/binfeng/hf_models/calme3.2-78b/ \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_calme78b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/calme78b_ppt_v2 \
    --train_split=ppt_train \
    --val_split=ppt_val \
    --epoch=1 \
    --lr=1e-3 \
    --bs=32 \
    --wd=1e-4 \
    --bs_per_device=16 \
    --loss_weights="[0.6,0.4,0.0,0.0]" \
    --save_only_model=True \
    --use_lora \
    --seed=32

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_qlora_p8.yaml" distill_lora.py \
    --wandb_project=wsdm_final \
    --run_name=calme78_ft_v2 \
    --tokenizer_path=/group-volume/binfeng/wsdm/ckpt/calme78b_ppt_v2 \
    --model_path=/group-volume/binfeng/hf_models/calme3.2-78b/ \
    --lora_weights=/group-volume/binfeng/wsdm/ckpt/calme78b_ppt_v2 \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_calme78b_final \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/calme78b_ft_v2 \
    --train_split=ft_train \
    --val_split=ft_val \
    --epoch=2 \
    --lr=8e-4 \
    --bs=8 \
    --wd=0. \
    --bs_per_device=8 \
    --loss_weights="[0.4,0.4,0.05,0.05]" \
    --save_only_model=True \
    --use_lora \
    --seed=32
