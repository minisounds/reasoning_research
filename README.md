# Reasoning Research - Jason Zhang 
Generalized Reasoning Research - Training Sparse Autoencoders in Pursuit of Generalized Reasoning Control Vector - Stanford SERI Summer Cohort 2024 - Advised by Scott Viteri

# Install Dependencies
pip install transformers openai datasets scikit-learn

# Authenticate with HuggingFace
huggingface-cli login

# Commit & Push to Runpod VM
git config --global user.email "MY_NAME@example.com"

git config --global user.name "FIRST_NAME LAST_NAME"

# Git LFS
apt update
apt install git-lfs
git lfs install

# Run Finetune Command
python finetune_llama.py --data_path "data/processed_gsm8k_dataset.json" --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --batch_size 1 --learning_rate 0.00005 --num_epochs 3 --warmup_steps 100


