# Load model directly
import h5py
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import torch
from tqdm import tqdm
from benchmarks.addition_benchmark import generate_addition_problem # benchmark #1 - add 3 numbers together 

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token 
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
config.use_cache = False 
    

def get_steering_vector(texts, model, tokenizer, layer_idx=15):
    if len(texts) != 2:
        raise ValueError("This function requires exactly two texts to compute the steering vector difference.")
    
    activations = []

    # Function to save activations
    def save_activation(model, input, output):
        activations.append(output[0].detach())  # Assuming output is a tuple and we need the first element

    # Register hook for the specified layer
    handle = model.model.layers[layer_idx].register_forward_hook(save_activation)

    # Process both texts and capture activations
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Perform forward pass to trigger hooks
        with torch.no_grad():
            _ = model(**inputs)
            
    # Remove the hook after processing both texts
    handle.remove()

    # Ensure both activation tensors have the same sequence length
    max_seq_length = max(activations[0].shape[1], activations[1].shape[1])
    padded_activations = [torch.nn.functional.pad(act, (0, 0, 0, max_seq_length - act.shape[1])) for act in activations]

    steering_vector = padded_activations[1] - padded_activations[0]

    return steering_vector

# Dataset 
def generate_dataset(): 
    num1, num2, num3, answer = generate_addition_problem()
    w_cot = f"Answer the following problem by thinking step by step: {num1} + {num2} + {num3} = "
    wo_cot = f"Answer the following problem by just providing the answer: {num1} + {num2} + {num3} = "
    return w_cot, wo_cot, answer

w_cot, wo_cot, answer = generate_dataset()
prompts = [w_cot, wo_cot]

steering_vector = get_steering_vector(prompts, model, tokenizer)

def add_steering_vectors_hook(module, input, output):
    global steering_vector  # Ensure this is accessible
    
    # Get the current sequence length
    current_seq_length = output[0].shape[1]
    
    # Trim or pad the steering vector to match the current sequence length
    if steering_vector.shape[1] > current_seq_length:
        adjusted_steering_vector = steering_vector[:, :current_seq_length, :]
    else:
        adjusted_steering_vector = torch.nn.functional.pad(
            steering_vector, 
            (0, 0, 0, current_seq_length - steering_vector.shape[1], 0, 0)
        )
    
    # Add the adjusted steering vector to the output
    return output[0] + adjusted_steering_vector, output[1]

def test_steering():
    n1, n2, n3, answer = generate_addition_problem() # insert new prompt here (can be loop in future)
    pre = tokenizer.decode(model.generate(tokenizer(f"Solve the following problem: {n1} + {n2} + {n3} = ", return_tensors="pt")["input_ids"], max_new_tokens=50)[0])
    print(f"pre answer: {pre}")
    model.model.layers[15].register_forward_hook(add_steering_vectors_hook)
    post = tokenizer.decode(model.generate(tokenizer(f"Solve the following problem: {n1} + {n2} + {n3} = ", return_tensors="pt")["input_ids"], max_new_tokens=50)[0])
    print(f"post answer: {post}")
    print(f"answer: {answer}")

test_steering()