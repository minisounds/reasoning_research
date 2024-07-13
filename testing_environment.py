# Load model directly
import h5py
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import torch
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token 
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
config.use_cache = False 
    
def get_steering_vectors():
    # Dictionary to store activations
    # TODO: use 3rd party service to store activations
    activations = {}

    # Hook function to save activations
    def save_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()  # Assuming the first element is the main output
            else:
                activations[name] = output.detach()
        return hook

    # Register hooks for layers you want to capture
    layer_idx = 15
    layers_to_capture = [layer_idx] # capture latter half of layers (look at which ones spec)
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
        return activations[f"layer_{layer_idx}"][1] - activations[f"layer_{layer_idx}"][0]
    
    texts = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        # ... add more texts here
    ]

    # Save activations to HDF5 file
    with h5py.File('llama_activations.h5', 'w') as f:
        steering_vectors = process_batch(texts, f)

    print("Activations saved to llama_activations.h5")
    return steering_vectors
    
    
#def steered_forward(model, layer_number, steering_vectors, prompt):
#    model.model.layers[layer_number]

steering_vectors = get_steering_vectors()
def add_steering_vectors_hook(module, input, output):
    if module == model.model.layers[15]:
        return output[0] + steering_vectors, output[1]
    return output

pre = model(tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"])
model.model.layers[15].register_forward_hook(add_steering_vectors_hook)
post = model(tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"])

assert pre != post

# Access the logits from the output tuple
logits = post[0]

# Check the shape of the logits
print(logits.shape)