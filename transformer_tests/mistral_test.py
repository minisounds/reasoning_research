# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

question = "What is the meaning of life?"
prompt = f"<s>[INST]{question}[/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
 
generated_ids = model.generate(**inputs, max_new_tokens=300, do_sample=True)

# decode with mistral tokenizer
# result = tokenizer.decode(generated_ids[0].tolist())
result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(result)
