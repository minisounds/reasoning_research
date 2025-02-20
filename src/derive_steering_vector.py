import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from steering_utils import get_pooled_activations, get_mean_mass_steering_vector, device, setup_model_and_tokenizer

# MODEL & TOKENIZER & SEED SET UP: 
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # "meta-llama/Meta-Llama-3-8B-Instruct"
SEED=43
model, tokenizer, config = setup_model_and_tokenizer(MODEL_NAME, seed=SEED, device=device)
    
# DATASET PROCESSING
def process_data(layer): 
    """ 
    Process data to obtain activations with and without chain-of-thought (CoT) reasoning 
    for a specified layer of the model.

    Args:
        layer (int): The layer of the model for which to obtain activations.

    Returns:
        tuple: Two lists containing activations with CoT reasoning and without CoT reasoning.
    """
    w_cot_activations = []
    wo_cot_activations = []
    
    # Big Bench - Load 3 Questions from Each BBLite Topic - total of 75 questions
    topics = ['auto_debugging', 'bbq_lite_json', 'code_line_description', 'conceptual_combinations', 'conlang_translation', 'emoji_movie', 'formal_fallacies_syllogisms_negation', 'hindu_knowledge', 'known_unknowns', 'language_identification', 'linguistics_puzzles', 'logic_grid_puzzle', 'logical_deduction', 'misconceptions_russian', 'novel_concepts', 'operators', 'parsinlu_reading_comprehension', 'play_dialog_same_or_different', 'repeat_copy_logic', 'strange_stories', 'strategyqa', 'symbol_interpretation', 'vitaminc_fact_verification', 'winowhy']
    for topics in tqdm(topics, desc="Evaluating BigBench"): 
        bblite_data = load_dataset("bigbench", config, split="train", streaming=True, trust_remote_code=True)
        questions = bblite_data.take(3)
        for q in questions: 
            w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, q['inputs'], seed=SEED)
            w_cot_activations.append(w_cot_vector)
            wo_cot_activations.append(wo_cot_vector)

    print("Big Bench Done")
    
    # MMLU - Pull 2 Questions from Each Subsection in "dev" Split - total of 125 questions
    mmlu_data = load_dataset("cais/mmlu", "all", split="dev")
    question_count = 0
    for idx in tqdm(range(len(mmlu_data)), desc="Evaluating MMLU"):
        if question_count < 2:
            w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, mmlu_data['question'][idx], seed=SEED)
            w_cot_activations.append(w_cot_vector)
            wo_cot_activations.append(wo_cot_vector)
            question_count += 1
        else:
            question_count = 0
            idx += 3

    print("MMLU training completed")

    # GSM8K - Pull First 100 Examples from Training (Easy), Then Medium (4,500 until 4,600)
    gsm8k_data = load_dataset("gsm8k", "main", split="train")
    for i in tqdm(range(0, 101), desc="Evaluating GSM8K: Easy"):
        w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, gsm8k_data['question'][i], seed=SEED)
        w_cot_activations.append(w_cot_vector)
        wo_cot_activations.append(wo_cot_vector)

    # Medium Level GSM8K Questions
    for i in tqdm(range(4500, 4601), desc="Evaluating GSM8K: Medium"):
        w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, gsm8k_data['question'][i], seed=SEED)
        w_cot_activations.append(w_cot_vector)
        wo_cot_activations.append(wo_cot_vector)

    print("GSM8K training completed")
    
    return w_cot_activations, wo_cot_activations

# STEERING VECTOR GENERATION
def generate_and_save_steering_vectors(start_layer, end_layer, save_dir):
    """
    Generate and save steering vectors for specified layers of the model.

    Args:
        start_layer (int): The starting layer for which to generate steering vectors.
        end_layer (int): The ending layer for which to generate steering vectors.
        save_dir (str): The directory where the steering vectors will be saved.
    """
    for layer in range(start_layer, end_layer + 1): 
        # PROCESS LAYER
        w_cot_activations, wo_cot_activations = process_data(layer)
        # CREATE STEERING VECTOR 
        steering_vector = get_mean_mass_steering_vector(w_cot_activations, wo_cot_activations)
        # SAVE STEERING VECTOR 
        np.save(f"{save_dir}/steering_vector_layer_{layer}.npy", steering_vector)

# Define parameters for steering vector derivation
start_layer = 11
end_layer = 32
save_dir = "steering_vectors/steering_vectors_llama"

# Generate and save steering vectors
generate_and_save_steering_vectors(start_layer, end_layer, save_dir)