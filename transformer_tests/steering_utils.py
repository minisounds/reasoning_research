import torch
from transformers import LlamaForCausalLM, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampling_kwargs = dict(temperature=1.0, top_p=0.3)

w_cot_prompt = "<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI Assistant who answers questions step by step.<|eot_id|>"
wo_cot_prompt = "<|start_header_id|>system<|end_header_id|>\nYou are an AI Assistant who answers questions immediately without elaboration.<|eot_id|>"

def get_contrasted_pooled_activations(model, tokenizer, layer, question, seed=None):
    if seed is not None: 
        torch.manual_seed(seed)
        
    activations = []
    def extract_activation(model, input, output):
        activations.append(
            output[0].detach()
        )
    
    hook = model.model.layers[layer].register_forward_hook(extract_activation)
    
    w_cot = w_cot_prompt+f"\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
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

def get_pooled_activations(model, tokenizer, layer, question):
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
    
    # pool activations for equal activations
    pool_cot = average_pooling(activations[0])
    
    return pool_cot

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
            output[0][:, pos, :] += coeff*steering_vector # add to the last seq
        return output[0], output[1]
    return hook
    
def generate_steered_response_w_vector(model, tokenizer, layer_range, question, steering_vectors, coeff, pos, seed=None):
    if seed is not None: 
        torch.manual_seed(seed)
    full_prompt = f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    for i, layer in enumerate(range(layer_range[0], layer_range[1], 1)): 
        steering_vector = steering_vectors[i]
        handle = model.model.layers[layer].register_forward_hook(add_steering_vectors_hook(steering_vector, coeff, pos))
    
    with torch.no_grad():
        output = generate_response(model, inputs)
    
    handle.remove()
    
    return tokenizer.decode(output[0], skip_special_tokens=True)



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