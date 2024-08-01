import json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm

training_data = load_dataset("gsm8k", "main", split="train")

def generate_response(model, tokenizer, prompt, use_cot, max_length = 512):
    inputs = tokenizer(prompt, return_tensors="pt", max_length = max_length, truncation=True) 
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7 if use_cot else 0.3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def create_gsm8k_dataset(output_file, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
        
    full_prompt = f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    # positive and negative response
    processed_data = []
    for item in training_data:
        cot_system_prompt = "You are a helpful AI Assistant who answers questions step by step."
        wo_cot_system_prompt = "You are an AI Assistant who answers questions immediately without elaboration."
        question = item['question']
        
        pos_prompt=f"<|start_header_id|>system<|end_header_id|>\n\n{cot_system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        pos_response=generate_response(model, tokenizer, pos_prompt, use_cot=True)
        
        neg_prompt=f"<|start_header_id|>system<|end_header_id|>\n\n{wo_cot_system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        neg_response=generate_response(model, tokenizer, neg_prompt, use_cot=False)
        
        processed_data.append({
            'question': question,
            'positive_example': pos_response,
            'negative_example': neg_response
        })
        
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
create_gsm8k_dataset('data/processed_gsm8k_dataset.json')
        
        

    
    