import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

prompt = "The capital of France is"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))