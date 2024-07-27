# Load model directly
import os
import uuid
import json
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    GPT2LMHeadModel,
)
import torch
from evaluate_response import grade_response
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

w_cot_prompt = "<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI Assistant who answers questions step by step.<|eot_id|>"
wo_cot_prompt = "<|start_header_id|>system<|end_header_id|>\nYou are an AI Assistant who answers questions immediately without elaboration.<|eot_id|>"

def get_steering_vector(model, tokenizer, layer_idx, coeff):
    activations = []
    # Function to save activations
    def save_activation(model, input, output):
        activations.append(
            output[0].detach()
        )  # Assuming output is a tuple and we need the first element

    # Register hook for the specified layer
    if isinstance(model, LlamaForCausalLM):
        handle = model.model.layers[layer_idx].register_forward_hook(save_activation)
    elif isinstance(model, GPT2LMHeadModel):
        handle = model.transformer.h[layer_idx].register_forward_hook(save_activation)
    else:
        raise ValueError("Unsupported model type")
    
    # Process both texts and capture activations
    w_cot = tokenizer(w_cot_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    wo_cot = tokenizer(wo_cot_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    
    w_cot.to(device)
    wo_cot.to(device)

    # Perform forward pass to trigger hooks
    with torch.no_grad():
        _ = model(**w_cot)
        _ = model(**wo_cot)

    # Remove the hook after processing both texts
    handle.remove()

    # Ensure both activation tensors have the same sequence length
    max_seq_length = max(activations[0].shape[1], activations[1].shape[1])
    padded_activations = [
        torch.nn.functional.pad(act, (0, 0, 0, max_seq_length - act.shape[1]))
        for act in activations
    ]

    steering_vector = padded_activations[1] - padded_activations[0]
    steering_vector = coeff * steering_vector
    return steering_vector

def add_steering_vectors_hook(steering_vector):
    def hook(module, input, output):
        current_seq_length = output[0].shape[1]
        if steering_vector.shape[1] > current_seq_length:
            adjusted_steering_vector = steering_vector[:, :current_seq_length, :]
        else:
            adjusted_steering_vector = torch.nn.functional.pad(
                steering_vector,
                (0, 0, 0, current_seq_length - steering_vector.shape[1], 0, 0),
            )
        return output[0] + adjusted_steering_vector, output[1]
    return hook

def generate_response(model, inputs, **kwargs):
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=250,
            **sampling_kwargs,
            **kwargs
        )
    return output

def post_steering(model, tokenizer, layer, steering_vector):
    question = "Three friends, Alice, Bob, and Charlie, are sitting in a row. Alice is not sitting next to Bob. Bob is sitting to the right of Charlie. Who is sitting in the middle?"
    full_prompt = f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if isinstance(model, LlamaForCausalLM):
        handle = model.model.layers[layer].register_forward_hook(add_steering_vectors_hook(steering_vector))
    else:
        raise ValueError("Unsupported model type")
    
    post_responses = []
    with torch.no_grad():
        for _ in range(5): 
            output = generate_response(model, inputs)
            post_responses.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    handle.remove()
    
    post_scores = [int(grade_response(post, question)) for post in post_responses]
    avg_post = sum(post_scores) / len(post_scores)
    
    return avg_post, post_responses, post_scores

def grid_search(model, tokenizer, layer_range, coeff_range):
    question = "Three friends, Alice, Bob, and Charlie, are sitting in a row. Alice is not sitting next to Bob. Bob is sitting to the right of Charlie. Who is sitting in the middle?"
    results = []
    best_combination = {"layer": -1, "coefficient": -1, "avg_score": -1}
    
    for layer in tqdm(range(layer_range), desc="Layers"):
        for coeff in tqdm(range(3, coeff_range), desc="Coefficients", leave=False):
            steering_vector = get_steering_vector(model, tokenizer, layer, coeff)
            avg_post, post_responses, post_scores = post_steering(model, tokenizer, layer, steering_vector)
            
            result = {
                "layer": layer,
                "coefficient": coeff,
                "avg_score": avg_post,
                "scores": post_scores,
                "best_response": max(post_responses, key=lambda x: grade_response(x, question))
            }
            results.append(result)
            
            if avg_post > best_combination["avg_score"]:
                best_combination = {"layer": layer, "coefficient": coeff, "avg_score": avg_post}
    
    result_id = str(uuid.uuid4())
    final_result = {
        "metadata": {
            "result_id": result_id,
            "layer_range": layer_range,
            "coeff_range": coeff_range,
            "model": model.config.name_or_path
        },
        "results": results,
        "best_combination": best_combination
    }
    
    with open("../res/scores.json", "w") as f:
        json.dump(final_result, f, indent=4)
    
    return result_id

# Usage
layer_range = 32  # Adjust based on your model's architecture
coeff_range = 15  # Adjust based on your desired range
result_id = grid_search(model, tokenizer, layer_range, coeff_range)