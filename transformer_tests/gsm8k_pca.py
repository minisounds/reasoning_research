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
from steering_utils import get_contrasted_pooled_activations, get_pooled_activations, generate_steered_response_w_vector, generate_baseline_response, device
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

dataset = load_dataset("gsm8k", "main", split="test")

layer = 19
coeff = 4 # think: how to get best? 
max_seq_length = 512

# TODO: load dataset from GSM8k
dataset = dataset[:100]
activations = []
for question in tqdm(dataset["question"], desc="Processing Questions: "):
    pooled_activation = get_contrasted_pooled_activations(model, tokenizer, layer, coeff, question)
    activations.append(pooled_activation.cpu().numpy())

# stack hidden states
stacked_states = np.vstack(activations)

# Run PCA
pca = PCA(n_components=5)
pca_result = pca.fit_transform(stacked_states)

# Analyze results
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("Explained variance ratio:", explained_variance_ratio)
print("Cumulative explained variance ratio:", cumulative_variance_ratio)

# The first principal component (potential "steering vector")
steering_vector = pca.components_[0]
question = dataset['question'][8]
answer = dataset['answer'][8]

# TODO: create a steering function that generates steered response with hidden state 
pos = 1 # token where we inject
ex_response = generate_steered_response_w_vector(model, tokenizer, layer, question, steering_vector, pos)
baseline = generate_baseline_response(model, tokenizer, question)
print(f"steered response: \n {ex_response} \n")
print(f"baseline response: \n {baseline}")
print(f"answer: \n {answer}")