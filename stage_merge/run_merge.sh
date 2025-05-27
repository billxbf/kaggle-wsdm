pip install --upgrade transformers bitsandbytes accelerate peft scikit-learn deepspeed wandb

cd mergekit
pip install -e .
cd ..

python merge2.py