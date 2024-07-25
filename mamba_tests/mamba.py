# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import MambaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")
print("hi")