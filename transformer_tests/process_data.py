import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from steering_utils import get_pooled_activations, generate_baseline_response, device, set_seed
from evaluate_response import find_answer

# MODEL & TOKENIZER & SEED SET UP: 
set_seed(42)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = model.to(device)  # Move model to GPU
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# config.use_cache = False

# HELPER FUNC:
def get_mean_mass_steering_vector(w_cot_activations, wo_cot_activations): 
    w_cot_activations = [w_cot_vector.cpu().numpy() for w_cot_vector in w_cot_activations]
    wo_cot_activations = [wo_cot_vector.cpu().numpy() for wo_cot_vector in wo_cot_activations]
    
    mean_w_cot = np.mean(w_cot_activations, axis=0)
    mean_wo_cot = np.mean(wo_cot_activations, axis=0)
    
    steering_vector = mean_w_cot - mean_wo_cot
    return steering_vector
    
# DATASETS
def process_data(layer): 
    w_cot_activations = []
    wo_cot_activations = []
    
    # Big Bench
    configs = ['auto_debugging', 'bbq_lite_json', 'code_line_description', 'conceptual_combinations', 'conlang_translation', 'emoji_movie', 'formal_fallacies_syllogisms_negation', 'hindu_knowledge', 'known_unknowns', 'language_identification', 'linguistics_puzzles', 'logic_grid_puzzle', 'logical_deduction', 'misconceptions_russian', 'novel_concepts', 'operators', 'parsinlu_reading_comprehension', 'play_dialog_same_or_different', 'repeat_copy_logic', 'strange_stories', 'strategyqa', 'symbol_interpretation', 'vitaminc_fact_verification', 'winowhy']
    for config in tqdm(configs, desc="Evaluating BigBench"): 
        # bblite_data = load_dataset("bigbench", config, trust_remote_code=True)
        bblite_data = load_dataset("bigbench", config, split="train", streaming=True, trust_remote_code=True)
        questions = bblite_data.take(3)
        # questions = bblite_data['train']['inputs'][:3]
        for q in questions: 
            # TODO: Make sure get_pooled_activations() works
            w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, q['inputs'], seed=42)
            # w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, q, seed=42)
            w_cot_activations.append(w_cot_vector)
            wo_cot_activations.append(wo_cot_vector)

    print("Big Bench Done")
    # MMLU - Pull 2 Questions from Each Subsection (from dev) - total of 125 questions
    mmlu_data = load_dataset("cais/mmlu", "all")
    count = 0
    for i in tqdm(range(len(mmlu_data['dev'])), desc="Evaluating MMLU"):
        # iterate twice, then add 3 to index
        if count < 2: 
            w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, mmlu_data['dev']['question'][i], seed=42)
            w_cot_activations.append(w_cot_vector)
            wo_cot_activations.append(wo_cot_vector)
            count += 1
        else: 
            count = 0
            i += 3

    print("mmlu training completed")

    # GSM8K - Pull First 100 Examples from Training (Easy), Then Medium (4,500 until 4,600)
    gsm8k_data = load_dataset("gsm8k", "main", split="train")
    for i in tqdm(range(4500, 4600), desc="Evaluating GSM8K: Easy"):
        w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, gsm8k_data['question'][i], seed=42)
        w_cot_activations.append(w_cot_vector)
        wo_cot_activations.append(wo_cot_vector)

    # Medium Level GSM8K Questions
    for i in tqdm(range(4500, 4600), desc="Evaluating GSM8K: Medium"):
        w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, gsm8k_data['question'][i], seed=42)
        w_cot_activations.append(w_cot_vector)
        wo_cot_activations.append(wo_cot_vector)

    print("gsm8k training completed")
    
    return w_cot_activations, wo_cot_activations

for layer in range(10, 32): 
    w_cot_activations, wo_cot_activations = process_data(layer)
    # CREATE STEERING VECTOR 
    steering_vector = get_mean_mass_steering_vector(w_cot_activations, wo_cot_activations)
    # SAVE STEERING VECTOR 
    np.save(f"steering_vector_layer_{layer}.npy", steering_vector) 
    print("steering vector process completed")

print
