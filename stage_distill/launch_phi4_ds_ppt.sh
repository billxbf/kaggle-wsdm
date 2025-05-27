# pip install --upgrade pip
pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

accelerate launch --config_file "/group-volume/binfeng/wsdm/stage_ppt/config/deepspeed_z3_p8.yaml" distill.py \
    --wandb_project=wsdm_distill \
    --run_name=phi4_ppt \
    --tokenizer_path=/group-volume/binfeng/wsdm/tokenizer/phi4 \
    --model_path=microsoft/phi-4 \
    --dataset_path=/group-volume/binfeng/wsdm/data/tokenized_phi4_distill \
    --model_save_path=/group-volume/binfeng/wsdm/ckpt/phi4_ppt \
    --train_split=ppt135k_train \
    --val_split=ppt135k_val \
    --epoch=1 \
    --lr=2e-5 \
    --bs=32 \
    --wd=0. \
    --bs_per_device=4 \
    --save_only_model=True \
    --seed=32



