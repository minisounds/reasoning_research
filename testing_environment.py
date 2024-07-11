# Load model directly
import h5py
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import torch
from tqdm import tqdm

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token 
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
config.use_cache = False 
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Dictionary to store activations
# TODO: use 3rd party service to store activations
activations = {}

# Hook function to save activations
def save_activation(name):
    def hook(model, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach()  # TODO: Assuming the first element is the main output
        else:
            activations[name] = output.detach()
    return hook

# Register hooks for layers you want to capture
layers_to_capture = [15] # capture latter half of layers (look at which ones spec)
for layer_num in layers_to_capture:
    model.model.layers[layer_num].register_forward_hook(save_activation(f'layer_{layer_num}'))

# Function to process a batch of texts and save activations
def process_batch(texts, file, batch_size=32):
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        for layer_name, activation in activations.items():
            if isinstance(activation, torch.Tensor):
                activation_np = activation.cpu().numpy()
            else:
                print(f"Unexpected activation type for {layer_name}: {type(activation)}")
                continue

            dataset = file.require_dataset(
                f"{layer_name}/batch_{i//batch_size}", 
                shape=(len(batch),) + activation_np.shape[1:],
                dtype='float32',
                compression="gzip",
                compression_opts=9,
            )
            dataset[:] = activation_np

# def process_batch(texts, file, batch_size=32):
#     for i in tqdm(range(0, len(texts), batch_size)):
#         batch = texts[i:i+batch_size]
#         inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         for layer_name, activation in activations.items():
#             dataset = file.require_dataset(
#                 f"{layer_name}/batch_{i//batch_size}", 
#                 shape=(len(batch),) + activation.shape[1:],
#                 dtype='float32',
#                 compression="gzip",
#                 compression_opts=9,
#             )
#             dataset[:] = activation.cpu().numpy()

# Prepare input
# TODO: make this flexible to dataset (modeLing.json)
texts = [
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog.",
    # ... add more texts here
]

# Save activations to HDF5 file
with h5py.File('llama_activations.h5', 'w') as f:
    process_batch(texts, f)

print("Activations saved to llama_activations.h5")