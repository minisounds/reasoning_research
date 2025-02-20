import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
import numpy as np
import os
import json
from contextlib import ExitStack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampling_kwargs = dict(temperature=1.0, top_p=0.3)

# System Setup Functions: Setup Seed and Model and Tokenizer
def set_seed(seed):
    """
    Args:
        seed (int): the seed value for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def setup_model_and_tokenizer(model_name, seed, device):
    """
    Set up the model, tokenizer and configuration for a specified seed.

    Args:
        model_name (str): The name of the pre-trained model.
        seed (int): The seed value for reproducibility.
        device (torch.device): The device to move the model to (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: The model, tokenizer, and config.
    """
    set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = None
    if model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        config = LlamaConfig.from_pretrained(model_name)
        config.use_cache = False

    return model, tokenizer, config

def average_pooling(activations):
    pooled_states = torch.mean(activations, dim=1)
    return pooled_states

def get_pooled_activations(model, tokenizer, layer, question):
    """
    Extracts and average pools layer activations of a LM when prompted with CoT vs. without CoT.
   
    Args:
       model (PreTrainedModel): The language model to extract activations from (Llama or Mistral)
       tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
       layer (int): Index of the model layer to extract activations from.
       question (str): The input question to process.
       
    Returns:
        averaged_cot (torch.Tensor): Average pooled activations across token position when prompted with CoT. 
        averaged_wo_cot (torch.Tensor): Average pooled activations across token position when prompted without CoT. 
    """
    activations = []
    # Define hook function that captures the layer's output tensor and detaches it
    def extract_activation(model, input, output):
        activations.append(
            output[0].detach()
        )
    
    # Attaches the hook to the specified layer in model, ensuring layer activation gets captured and appended to activations list
    hook = model.model.layers[layer].register_forward_hook(extract_activation)
    
    w_cot = ""
    wo_cot = ""
    
    # Check model type and set the appropriate prompt format
    model_type = model.config.model_type
    if model_type == "llama":
        w_cot = f"<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI Assistant who answers questions step by step.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>Answer the following question thinking step by step: \n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        wo_cot = f"<|start_header_id|>system<|end_header_id|>\nYou are an AI Assistant who answers questions immediately without elaboration.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>Answer the following question immediately, without any elaboration: \n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif model_type == "mistral": 
        w_cot = f"<s>[INST] You are a helpful AI Assistant who answers questions step by step. [/INST]\n[INST]Answer the following thinking step by step: {question} [/INST]\n"
        wo_cot = f"<s>[INST] You are an AI Assistant who answers immediately without elaboration. [/INST]\n[INST] Answer the following question immediately, without any elaboration: {question} [/INST]\n"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    w_cot_input_ids = tokenizer(w_cot, return_tensors="pt", padding=True, truncation=True, max_length=700, return_attention_mask=True)
    w_cot_input_ids.to(device)

    wo_cot_input_ids = tokenizer(wo_cot, return_tensors="pt", padding=True, truncation=True, max_length=700, return_attention_mask=True)
    wo_cot_input_ids.to(device)
    
    with torch.no_grad():
        _ = model(**w_cot_input_ids)
        _ = model(**wo_cot_input_ids)
    
    hook.remove()
    
    # average pool activations across token position
    averaged_cot = average_pooling(activations[0])
    averaged_wo_cot = average_pooling(activations[1])
    
    return averaged_cot, averaged_wo_cot

def get_mean_mass_steering_vector(w_cot_activations, wo_cot_activations): 
    """
    Calculate the steering vector by computing the mean difference between 
    list of activations with and without chain-of-thought (CoT) reasoning.

    Args:
        w_cot_activations (list of torch.Tensor): List of activations with CoT reasoning.
        wo_cot_activations (list of torch.Tensor): List of activations without CoT reasoning.

    Returns:
        numpy.ndarray: The steering vector obtained by subtracting the mean activations without CoT from the mean activations with CoT.
    """
    # Move pytorch activation tensors to CPU and convert to numpy arrays for vectorized operation
    w_cot_activations = [w_cot_vector.cpu().numpy() for w_cot_vector in w_cot_activations]
    wo_cot_activations = [wo_cot_vector.cpu().numpy() for wo_cot_vector in wo_cot_activations]
    
    mean_w_cot = np.mean(w_cot_activations, axis=0)
    mean_wo_cot = np.mean(wo_cot_activations, axis=0)
    
    steering_vector = mean_w_cot - mean_wo_cot
    return steering_vector


def add_steering_vectors_hook_batch(steering_vector, coeff, pos):
    """
    Creates a hook function that adds scaled steering vectors to specific positions in model activations.
    
    Args:
        steering_vector (array-like): Vector to add to model activations.
        coeff (float): Scaling coefficient for the steering vector.
        pos (list): Token positions where steering should be applied.
        
    Returns:
        function: Hook function that modifies model layer outputs.
    """
    steering_vector = torch.tensor(steering_vector).to(device)
    def hook(model, input, output):
        # Inject Alongside Prompt: Inject only when output contains more than 2 tokens. 
        # Flip to < 2 if you want to inject at Token Time.
        if output[0].shape[1] > 2:
            # Inject vector into all positions in pos list
            for p in pos:
                # Inject SV into token position p for all batches [batch_size, seq_len, hidden_dim]
                output[0][:, p, :] += coeff * steering_vector 
        return output
    return hook

def generate_steered_responses_batch(model, tokenizer, layer, questions, steering_vector, coeff, pos, seed=None):
    """
    Generates responses using activation steering to influence model output.
    
    Args:
        model (PreTrainedModel): Language model to generate responses.
        tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
        layer (int): Index of the model layer to apply steering.
        questions (list): List of input questions/prompts.
        steering_vector (array-like): Vector to steer activations.
        coeff (float): Scaling coefficient for the steering vector.
        pos (list): Token positions where steering should be applied.
        batch_size (int): Number of responses to generate in parallel.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        list: Processed model responses with steering applied.
    """
    if seed is not None: 
        torch.manual_seed(seed)
    
    model_type = model.config.model_type
    full_prompts = []
    if model_type == "llama":
        full_prompts = [f"<|start_header_id|>user<|end_header_id|>\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in questions]
    elif model_type == "mistral":
        full_prompts = [f"<s>[INST]\n{q}[\INST]" for q in questions]
    else: 
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Convert prompts into token IDs and move to device
    inputs = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    hook = model.model.layers[layer].register_forward_hook(add_steering_vectors_hook_batch(steering_vector, coeff, pos))
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=700,
            do_sample=True,
            **sampling_kwargs
        )
    
    hook.remove()
    
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    final_responses = []
    for response in responses:
        final_responses.append(parse_message(response))

    return final_responses

def parse_message(text):
    # start = text.find("[\\INST]") + 7
    start = text.find("assistant\n\n") + len("assistant\n\n")
    
    # Extract the user message
    user_message = text[start:].strip()
    
    return user_message

def generate_baseline_responses_batch(model, tokenizer, questions):
    """
    Generates baseline responses from a language model for a batch of questions.
    
    Args:
        model (PreTrainedModel): Language model to generate responses.
        tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
        questions (list): List of input questions/prompts.
        
    Returns:
        list: Processed model responses with step-by-step reasoning.
    """
    model_type = model.config.model_type
    
    full_prompts = []
    if model_type == "llama":
        full_prompts = [f"<|start_header_id|>user<|end_header_id|>Answer the following question thinking step by step: \n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in questions]
    elif model_type == "mistral":
        full_prompts = [f"<s>[INST]Answer the following question thinking step by step: [\INST]\n[INST]\n {q}[\INST]" for q in questions]
    else: 
        raise ValueError(f"Unsupported model type: {model_type}")
    
    inputs = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
        
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=700,
            do_sample=True,
            **sampling_kwargs
        )
    
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    final_responses = []
    for response in responses:
        final_responses.append(parse_message(response))
        
    return final_responses

def add_steering_vectors_hook(steering_vector, coeff, pos):
    steering_vector = torch.tensor(steering_vector).to(device)
    def hook(model, input, output):
        if output[0].shape[1] < 2:
            for p in pos: 
                output[0][:, p, :] += coeff*steering_vector # add to the last seq
        return output[0], output[1]
    return hook
    
def generate_steered_response_w_vector(model, tokenizer, layer, question, steering_vector, coeff, pos, seed=None):
    if seed is not None: 
        torch.manual_seed(seed)
    full_prompt = f"<s>[INST]\n{question}[\INST]"
    
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