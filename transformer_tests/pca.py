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

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
set_seed(42)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False
    
dataset = load_dataset("gsm8k", "main", split="test")


max_seq_length = 512

dataset = dataset[:100] #TODO: PLAY WITH THIS 
layer_range = [17, 21] #TODO: PLAY WITH THIS
activations = []

# Initialize an empty PyTorch tensor
concatenated_activation = torch.empty(0).to(device)

for question in tqdm(dataset["question"], desc="Processing Questions: "):
    concatenated_activation = torch.empty(0).to(device)
    for layer in range(layer_range[0], layer_range[1], 1):
        pooled_activation = get_contrasted_pooled_activations(model, tokenizer, layer, question, seed=42)
        concatenated_activation = torch.cat([concatenated_activation, pooled_activation], dim=-1)
    activations.append(concatenated_activation.cpu().numpy())

# stack hidden states
stacked_states = np.vstack(activations)

# Run PCA
pca = PCA(n_components=10)
pca_result = pca.fit_transform(stacked_states)

# Analyze results
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("Explained variance ratio:", explained_variance_ratio)
print("Cumulative explained variance ratio:", cumulative_variance_ratio)

# The first principal component (potential "steering venctor")
steering_vector = pca.components_[0]
steering_vectors = [steering_vector[i:i+4096] for i in range(0, len(steering_vector), 4096)]
question = "Three friends, Alice, Bob, and Charlie, are sitting in a row. Alice is not sitting next to Bob. Bob is sitting to the right of Charlie. Who is sitting in the middle?"
# question = dataset['question'][8]
answer = dataset['answer'][8]

# HYPERPARAMETERS
pos = -1 # token where we inject (currently the last token)
coeff = 50 # think: how to get best? 

ex_response = generate_steered_response_w_vector(model, tokenizer, layer_range, question, steering_vectors, coeff, pos, seed=42)
baseline = generate_baseline_response(model, tokenizer, question, seed=42)
print(f"steered response: \n {ex_response} \n")
print(f"baseline response: \n {baseline}")
print(f"answer: \n {answer}")