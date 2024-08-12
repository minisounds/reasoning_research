import torch
from transformers import LlamaForCausalLM, GPT2LMHeadModel
import numpy as np
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampling_kwargs = dict(temperature=1.0, top_p=0.3)

eight_shot = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n"
w_cot_prompt = "<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI Assistant who answers questions step by step.<|eot_id|>"
wo_cot_prompt = "<|start_header_id|>system<|end_header_id|>\nYou are an AI Assistant who answers questions immediately without elaboration.<|eot_id|>"

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
def get_contrasted_pooled_activations(model, tokenizer, layer, question, seed=None):
    if seed is not None: 
        torch.manual_seed(seed)
        
    activations = []
    def extract_activation(model, input, output):
        activations.append(
            output[0].detach()
        )
    
    hook = model.model.layers[layer].register_forward_hook(extract_activation)
    
    w_cot = w_cot_prompt+f"\n<|start_header_id|>user<|end_header_id|>{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    wo_cot = wo_cot_prompt+f"\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    
    cot_input_ids = tokenizer(w_cot, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    wo_cot_input_ids = tokenizer(wo_cot, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    
    cot_input_ids.to(device)
    wo_cot_input_ids.to(device)
    
    with torch.no_grad():
        _ = model(**cot_input_ids)
        _ = model(**wo_cot_input_ids)
    
    hook.remove()
    
    # pool activations for equal activations
    pool_cot = average_pooling(activations[0])
    pool_wo_cot = average_pooling(activations[1])
    
    return pool_cot-pool_wo_cot

def get_pooled_activations(model, tokenizer, layer, question, seed=None):
    if seed is not None: 
        torch.manual_seed(seed)

    activations = []
    def extract_activation(model, input, output):
        activations.append(
            output[0].detach()
        )
    
    hook = model.model.layers[layer].register_forward_hook(extract_activation)
    
    w_cot = w_cot_prompt+f"\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    w_cot_input_ids = tokenizer(w_cot, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    w_cot_input_ids.to(device)

    wo_cot = wo_cot_prompt+f"\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    wo_cot_input_ids = tokenizer(wo_cot, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    wo_cot_input_ids.to(device)
    
    with torch.no_grad():
        _ = model(**w_cot_input_ids)
        _ = model(**wo_cot_input_ids)
    
    hook.remove()
    
    # pool activations for equal activations
    pool_w_cot = average_pooling(activations[0])
    pool_wo_cot = average_pooling(activations[1])
    
    return pool_w_cot, pool_wo_cot

def average_pooling(activations):
    pooled_states = torch.mean(activations, dim=1)
    return pooled_states

def get_hidden_state(model, tokenizer, layer, question): 
    w_cot = w_cot_prompt+f"\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    # wo_cot = wo_cot_prompt+f"\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    
    cot_input_ids = tokenizer(w_cot, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    # wo_cot_input_ids = tokenizer(wo_cot, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    
    cot_input_ids.to(device)
    # wo_cot_input_ids.to(device)
    
    with torch.no_grad():
        output = model(**cot_input_ids, output_hidden_states=True)
        hidden_state = output.hidden_states
        # hidden_state = output.hidden_states
        # _ = model(**wo_cot_input_ids)
    
    format_state = average_pooling(hidden_state[layer])
    
    return format_state

def get_activations(model, tokenizer, layer, coeff, question):
    activations = []
    def extract_activation(model, input, output):
        activations.append(
            output[0].detach()
        )
    
    hook = model.model.layers[layer].register_forward_hook(extract_activation)
    
    w_cot = w_cot_prompt+f"\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    
    cot_input_ids = tokenizer(w_cot, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    cot_input_ids.to(device)
    
    with torch.no_grad():
        _ = model(**cot_input_ids)
    
    hook.remove()
    
    return coeff * activations[0]

def add_steering_vectors_hook(steering_vector, coeff, pos):
    steering_vector = torch.tensor(steering_vector).to(device)
    def hook(model, input, output):
        if output[0].shape[1] > 2:
            for p in pos: 
                output[0][:, p, :] += coeff*steering_vector # add to the last seq
        return output[0], output[1]
    return hook
    
def generate_steered_response_w_vector(model, tokenizer, layer, question, steering_vector, coeff, pos, seed=None):
    if seed is not None: 
        torch.manual_seed(seed)
    full_prompt = f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # for i, layer in enumerate(range(layer_range[0], layer_range[1], 1)): 
    handle = model.model.layers[layer].register_forward_hook(add_steering_vectors_hook(steering_vector, coeff, pos))
    
    with torch.no_grad():
        output = generate_response(model, inputs)
    
    handle.remove()
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def add_steering_vectors_hook_batch(steering_vector, coeff, pos):
    steering_vector = torch.tensor(steering_vector).to(device)
    def hook(model, input, output):
        if output[0].shape[1] > 2:
            for p in pos:
                output[0][:, p, :] += coeff * steering_vector  # Add batch dimension
        return output
    return hook

def generate_steered_responses_batch(model, tokenizer, layer, questions, steering_vector, coeff, pos, batch_size, seed=None):
    if seed is not None: 
        torch.manual_seed(seed)
        
    full_prompts = [f"<|start_header_id|>user<|end_header_id|>\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in questions]
        
    inputs = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    handle = model.model.layers[layer].register_forward_hook(add_steering_vectors_hook_batch(steering_vector, coeff, pos))
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=250,
            **sampling_kwargs
        )
    
    handle.remove()
    
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return responses
        
def generate_baseline_responses_batch(model, tokenizer, questions, seed=None, batch_size=16):
    if seed is not None:
        torch.manual_seed(seed)
    
    all_responses = []
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        
        full_prompts = [f"<|start_header_id|>user<|end_header_id|>\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in batch_questions]
        
        inputs = tokenizer(
            full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=250,
                **sampling_kwargs
            )
        
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_responses.extend(batch_responses)
    
    return all_responses


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

def write_to_json(json_path, responses):
    if os.path.exists(json_path):
        # File exists, load existing data
        with open(json_path, 'r') as f:
            existing_data = json.load(f)
        # Append new responses
        existing_data.extend(responses)
    else:
        # File doesn't exist, use new responses
        existing_data = responses

    # Write updated data back to file
    with open(json_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    
def generate_steered_response(model, tokenizer, question, layer, coeff):
    # Generate steered vector
    steering_vector = get_steering_vector(model, tokenizer, layer, coeff)
    
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
    
    with torch.no_grad():
        output = generate_response(model, inputs)
    
    handle.remove()
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_baseline_response(model, tokenizer, question, seed=None):
    if seed is not None: 
        torch.manual_seed(seed)
        
    full_prompt = f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    inputs = tokenizer(
        full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()} 

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def generate_multi_layer_steered_response(model, tokenizer, question, layer_range, coeff):
    # Generate steering vectors for multiple layers
    steering_vectors = {}
    for layer in layer_range:
        steering_vectors[layer] = get_steering_vector(model, tokenizer, layer, coeff)
    
    full_prompt = f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    handles = []
    if isinstance(model, LlamaForCausalLM):
        for layer, steering_vector in steering_vectors.items():
            handle = model.model.layers[layer].register_forward_hook(add_steering_vectors_hook(steering_vector))
            handles.append(handle)
    else:
        raise ValueError("Unsupported model type")
    
    with torch.no_grad():
        output = generate_response(model, inputs)
    
    # Remove all hooks
    for handle in handles:
        handle.remove()
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Helper function to get layer ranges
def get_layer_ranges(model, num_sections=3):
    if isinstance(model, LlamaForCausalLM):
        total_layers = len(model.model.layers)
    else:
        raise ValueError("Unsupported model type")
    
    section_size = total_layers // num_sections
    ranges = []
    for i in range(num_sections):
        start = i * section_size
        end = start + section_size if i < num_sections - 1 else total_layers
        ranges.append(range(start, end))
    return ranges


def get_last_token_steering_vector(model, tokenizer, layer_idx, coeff):
    activations = []

    def save_activation(model, input, output):
        # Save only the last token's activation
        activations.append(output[0][:, -1, :].detach())

    if isinstance(model, LlamaForCausalLM):
        handle = model.model.layers[layer_idx].register_forward_hook(save_activation)
    elif isinstance(model, GPT2LMHeadModel):
        handle = model.transformer.h[layer_idx].register_forward_hook(save_activation)
    else:
        raise ValueError("Unsupported model type")
    
    w_cot = tokenizer(w_cot_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    wo_cot = tokenizer(wo_cot_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    
    w_cot.to(device)
    wo_cot.to(device)

    with torch.no_grad():
        _ = model(**w_cot)
        _ = model(**wo_cot)

    handle.remove()

    steering_vector = activations[1] - activations[0]
    steering_vector = coeff * steering_vector
    return steering_vector

def add_last_token_steering_vectors_hook(steering_vector):
    def hook(module, input, output):
        # Apply the steering vector only to the last token
        output[0][:, -1, :] += steering_vector
        return output
    return hook

def generate_last_token_steered_response(model, tokenizer, question, layer, coeff):
    # Generate steered vector
    steering_vector = get_last_token_steering_vector(model, tokenizer, layer, coeff)
    
    full_prompt = f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if isinstance(model, LlamaForCausalLM):
        handle = model.model.layers[layer].register_forward_hook(add_last_token_steering_vectors_hook(steering_vector))
    else:
        raise ValueError("Unsupported model type")
    
    with torch.no_grad():
        output = generate_response(model, inputs)
    
    handle.remove()
    
    return tokenizer.decode(output[0], skip_special_tokens=True)