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
from steering_utils import generate_last_token_steered_response, device
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

def grid_search(model, tokenizer, layer_range, coeff_range):
    question = "Three friends, Alice, Bob, and Charlie, are sitting in a row. Alice is not sitting next to Bob. Bob is sitting to the right of Charlie. Who is sitting in the middle?"
    results = []
    best_combination = {"layer": -1, "coefficient": -1, "avg_score": -1}
    
    for layer in tqdm(range(layer_range), desc="Layers"):
        for coeff in tqdm(range(3, coeff_range), desc="Coefficients", leave=False):
            post_responses = []
            for i in range(3):
                response = generate_last_token_steered_response(model, tokenizer, question, layer, coeff)
                post_responses.append(response)
            
            post_scores = [int(grade_response(post, question)) for post in post_responses]
            avg_post = sum(post_scores) / len(post_scores)

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
layer_range = 5  # Adjust based on your model's architecture
coeff_range = 4 # Adjust based on your desired range
result_id = grid_search(model, tokenizer, layer_range, coeff_range)
