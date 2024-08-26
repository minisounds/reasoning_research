# Reasoning Research - Jason Zhang 
Generalized Reasoning Research - Training Sparse Autoencoders in Pursuit of Generalized Reasoning Control Vector - Stanford SERI Summer Cohort 2024 - Advised by Scott Viteri

# Install Dependencies
pip install transformers openai datasets scikit-learn

# Install Big Bench
pip install "bigbench @ https://storage.googleapis.com/public_research_data/bigbench/bigbench-0.0.1.tar.gz"

# Authenticate with HuggingFace
huggingface-cli login

# Commit & Push to Runpod VM
git config --global user.email "jzscuba@gmail.com"
git config --global user.name "Jason"

# Git LFS
apt update
apt install git-lfs
git lfs install
git lfs pull

# Run Finetune Command
python finetune_llama.py --data_path "data/processed_gsm8k_dataset.json" --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --batch_size 1 --learning_rate 0.00005 --num_epochs 3 --warmup_steps 100