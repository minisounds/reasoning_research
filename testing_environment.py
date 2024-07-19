# Load model directly
import h5py
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    GPT2LMHeadModel,
)
import torch
from tqdm import tqdm
from benchmarks.addition_benchmark import (
    generate_addition_problem,
)  # benchmark #1 - add 3 numbers together

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = model.to(device)  # Move model to GPU

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
config.use_cache = False

# model = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("")
# tokenizer.pad_token = tokenizer.eos_token
# config = model.config
# config.use_cache = False
LAYER = 20
INJ_COEF = 15
w_cot_prompt = f"Think step by step."
wo_cot_prompt = f"Answer immediately."

tokenLen = lambda tokens: len(tokens["input_ids"][0])

# returns new input_ids with padding of space character AND updates attention mask
def pad_right(tensor, length):
    space_token = tokenizer.encode(" ", add_special_tokens=False)[0]
    space_tokens_tensor = torch.tensor([space_token] * (length - tokenLen(tensor)))
    padded_tokens = torch.cat((tensor["input_ids"][0], space_tokens_tensor), dim=0)
    attention_mask = torch.cat((tensor["attention_mask"][0], torch.zeros_like(space_tokens_tensor)), dim=0)
    return padded_tokens, attention_mask

# TODO: removed "texts" put positive and negative strings directly in here
def get_steering_vector(model, tokenizer, layer_idx=LAYER):
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
    w_cot = tokenizer(w_cot_prompt, return_tensors="pt", padding=False, truncation=True, max_length=512, return_attention_mask=True)
    wo_cot = tokenizer(wo_cot_prompt, return_tensors="pt", padding=False, truncation=True, max_length=512, return_attention_mask=True)
    
    # manual padding 
    l = max(tokenLen(w_cot), tokenLen(wo_cot))
    padded_cot = pad_right(w_cot, l)
    padded_wo_cot = pad_right(wo_cot, l)
    w_cot["input_ids"][0], w_cot["attention_mask"][0] = padded_cot[0].unsqueeze(0), padded_cot[1].unsqueeze(0)
    wo_cot["input_ids"][0], wo_cot["attention_mask"][0] = padded_wo_cot[0].unsqueeze(0), padded_wo_cot[1].unsqueeze(0)
    
    w_cot.to(device)
    wo_cot.to(device)

    # Perform forward pass to trigger hooks
    with torch.no_grad():
        _ = model(**w_cot)
        _ = model(**wo_cot)

    # Remove the hook after processing both texts
    handle.remove()
    # TODO: make sure that save_activation hook is deleted after use

    # Ensure both activation tensors have the same sequence length
    max_seq_length = max(activations[0].shape[1], activations[1].shape[1])
    padded_activations = [
        torch.nn.functional.pad(act, (0, 0, 0, max_seq_length - act.shape[1]))
        for act in activations
    ]

    steering_vector = padded_activations[1] - padded_activations[0]
    steering_vector = INJ_COEF * steering_vector
    return steering_vector

# Dataset
steering_vector = get_steering_vector(model, tokenizer)

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
            (0, 0, 0, current_seq_length - steering_vector.shape[1], 0, 0),
        )

    # Add the adjusted steering vector to the output
    return output[0] + adjusted_steering_vector, output[1]


def test_steering(num_responses=3):
    # question, answer = generate_addition_problem()  # insert new prompt here (can be loop in future)
    question = "Three friends, Alice, Bob, and Charlie, are sitting in a row. Alice is not sitting next to Bob. Bob is sitting to the right of Charlie. Who is sitting in the middle?"
    answer = "Bob is sitting in the middle"
    # Generate tokens before steering
    inputs = tokenizer(
        f"Solve the following problem: {question}",
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to GPU 
    
    # Generate multiple responses before steering
    pre_responses = []
    with torch.no_grad():
        for _ in range(num_responses):
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=400,
            )
            pre_responses.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    print("Pre-steering responses:")
    for i, pre in enumerate(pre_responses, 1):
        print(f"Pre answer {i}: {pre}")
    
    if isinstance(model, LlamaForCausalLM):
        print("in llama rn")
        model.model.layers[LAYER].register_forward_hook(add_steering_vectors_hook)
    elif isinstance(model, GPT2LMHeadModel):
        model.transformer.h[LAYER].register_forward_hook(add_steering_vectors_hook)
    else:
        raise ValueError("Unsupported model type")
    
    inputs = tokenizer(
        f"Solve the following problem: {question}",
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to GPU 
    
    # Generate multiple responses after steering
    post_responses = []
    with torch.no_grad():
        for _ in range(num_responses):
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=300,
            )
            post_responses.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    print("Post-steering responses:")
    for i, post in enumerate(post_responses, 1):
        print(f"Post answer {i}: {post}")
    
    # print(f"answer: {answer}")

test_steering()
