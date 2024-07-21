import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from benchmarks.addition_benchmark import generate_addition_problem

# Set up the device
print(torch.cuda.is_available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

def get_baseline(question):
    inputs = tokenizer(
        question, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Example usage
def test():
    q, ans = generate_addition_problem()
    # question = f"Solve the following problem: {q} = "
    question = "What is the capital of France?"
    # question = "What is the answer to this riddle: Three friends, Alice, Bob, and Charlie, are sitting in a row. Alice is not sitting next to Bob. Bob is sitting to the right of Charlie. Who is sitting in the middle?"
    answer = get_baseline(question)
    print(f"Question: {question}")
    print(f"Model Answer: {answer}")
    print(f"Actual Answer: B")

test()
