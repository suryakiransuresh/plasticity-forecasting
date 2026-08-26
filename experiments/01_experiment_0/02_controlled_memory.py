import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

MODEL_NAME = "gpt2"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

tokenizer.pad_token = tokenizer.eos_token

protected_examples = [
    ("The capital of Velora is", " Nymara"),
    ("The capital of Torvane is", " Eloria"),
    ("The capital of Caldris is", " Vensar"),
    ("The capital of Merovia is", " Talune"),
    ("The capital of Zoravia is", " Pelith"),
]

def answer_loss(model, prompt, answer):
    full_text = prompt + answer

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"].to(device)

    full_ids = tokenizer(
        full_text,
        return_tensors="pt"
    )["input_ids"].to(device)

    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            labels=labels
        )

    return outputs.loss.item()


def evaluate_examples(model, examples):
    results = []

    for prompt, answer in examples:
        loss = answer_loss(model, prompt, answer)
        results.append((prompt, answer, loss))

    return results

# -------------------------
# BEFORE TRAINING
# -------------------------

model.eval()

before_results = evaluate_examples(
    model,
    protected_examples
)

print("\n=== BEFORE CONTROLLED TRAINING ===")

for prompt, answer, loss in before_results:
    print(f"{prompt}{answer}")
    print(f"  Loss: {loss:.4f}")

# -------------------------
# CONTROLLED TRAINING
# -------------------------

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.0
)

EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0.0

    for prompt, answer in protected_examples:
        full_text = prompt + answer

        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt"
        )["input_ids"].to(device)

        full_ids = tokenizer(
            full_text,
            return_tensors="pt"
        )["input_ids"].to(device)

        labels = full_ids.clone()

        # Train only on answer tokens
        labels[:, :prompt_ids.shape[1]] = -100

        optimizer.zero_grad()

        outputs = model(
            input_ids=full_ids,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(protected_examples)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Avg train loss: {avg_train_loss:.4f}"
    )

    # -------------------------
# AFTER CONTROLLED TRAINING
# -------------------------

model.eval()

after_results = evaluate_examples(
    model,
    protected_examples
)

print("\n=== AFTER CONTROLLED TRAINING ===")

for before, after in zip(
    before_results,
    after_results
):
    prompt, answer, loss_before = before
    _, _, loss_after = after

    improvement = loss_before - loss_after

    print(f"\n{prompt}{answer}")
    print(f"  Before:     {loss_before:.4f}")
    print(f"  After:      {loss_after:.4f}")
    print(f"  Improvement:{improvement:+.4f}")

import os

checkpoint_dir = "results/checkpoints/controlled_memory"

os.makedirs(checkpoint_dir, exist_ok=True)

model.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)

print(f"\nSaved controlled-memory checkpoint to: {checkpoint_dir}")