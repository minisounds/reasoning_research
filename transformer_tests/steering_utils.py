import torch
from transformers import LlamaForCausalLM, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampling_kwargs = dict(temperature=1.0, top_p=0.3)

w_cot_prompt = "<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI Assistant who answers questions step by step.<|eot_id|>"
wo_cot_prompt = "<|start_header_id|>system<|end_header_id|>\nYou are an AI Assistant who answers questions immediately without elaboration.<|eot_id|>"

def get_steering_vector(model, tokenizer, layer_idx, coeff):
    activations = []

    def save_activation(model, input, output):
        activations.append(
            output[0].detach()
        )

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

    