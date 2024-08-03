import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    GPT2LMHeadModel,
)
from datasets import load_dataset
from sklearn.decomposition import PCA
import numpy as np
from steering_utils import get_activations, get_contrasted_activations, generate_steered_response_w_vector, device
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

dataset = load_dataset("gsm8k", "main", split="test")

all_activations = []
layer = 19
coeff = 4 # think: how to get best? 
max_seq_length = 512

# TODO: load dataset from FSM8k
dataset = dataset[:150]
for question in tqdm(dataset["question"], desc="Processing Questions: "):
    activations = get_contrasted_activations(model, tokenizer, layer, coeff, question)
    
    # Move activations to CPU and convert to NumPy
    activations_cpu = activations.cpu().numpy()
    
    # Flatten activation tensor
    flattened = activations_cpu.reshape(activations_cpu.shape[0], -1)
    
    if flattened.shape[1] < max_seq_length * model.config.hidden_size:
        padding = np.zeros((flattened.shape[0], max_seq_length * model.config.hidden_size - flattened.shape[1]))
        flattened = np.hstack([flattened, padding])
    
    all_activations.append(flattened)

# stack activations
stacked_activations = np.vstack(all_activations)

# Run PCA
pca = PCA(n_components=10)
pca_result = pca.fit_transform(stacked_activations)

# Analyze results
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("Explained variance ratio:", explained_variance_ratio)
print("Cumulative explained variance ratio:", cumulative_variance_ratio)

# The first principal component (potential "steering vector")
steering_vector = pca.components_[0]
question = dataset['question'][0]
ex_response = generate_steered_response_w_vector(model, tokenizer, question, steering_vector)