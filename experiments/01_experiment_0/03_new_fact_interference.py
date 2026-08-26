import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# REPRODUCIBILITY
# -------------------------

SEED = 42

torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


# -------------------------
# DEVICE
# -------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")


# -------------------------
# LOAD CONTROLLED MEMORY
# -------------------------

CHECKPOINT = "results/checkpoints/controlled_memory"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT
).to(device)

tokenizer.pad_token = tokenizer.eos_token

model.eval()


# -------------------------
# OLD PROTECTED MEMORIES
# -------------------------

protected_examples = [
    ("The capital of Velora is", " Nymara"),
    ("The capital of Torvane is", " Eloria"),
    ("The capital of Caldris is", " Vensar"),
    ("The capital of Merovia is", " Talune"),
    ("The capital of Zoravia is", " Pelith"),
]


# -------------------------
# NEW EXPERIENCE
# -------------------------

novel_prompt = "The capital of Ravelle is"
novel_answer = " Sorin"


print("Controlled-memory checkpoint loaded successfully.")
print(f"New fact: {novel_prompt}{novel_answer}")

# -------------------------
# LOSS FUNCTIONS
# -------------------------

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
# BEFORE NEW FACT UPDATE
# -------------------------

model.eval()

protected_before = evaluate_examples(
    model,
    protected_examples
)

novel_loss_before = answer_loss(
    model,
    novel_prompt,
    novel_answer
)

print("\n=== BEFORE NEW FACT UPDATE ===")
print(f"Novel fact loss: {novel_loss_before:.4f}")

print("\nProtected memories:")

for prompt, answer, loss in protected_before:
    print(f"{prompt}{answer}")
    print(f"  Loss: {loss:.4f}")


# -------------------------
# CREATE ADAPTED COPY
# -------------------------

adapted_model = copy.deepcopy(model)
adapted_model.train()


# -------------------------
# LEARN NEW FACT ONLY
# -------------------------

optimizer = torch.optim.AdamW(
    adapted_model.parameters(),
    lr=1e-5,
    weight_decay=0.0
)

full_text = novel_prompt + novel_answer

prompt_ids = tokenizer(
    novel_prompt,
    return_tensors="pt"
)["input_ids"].to(device)

full_ids = tokenizer(
    full_text,
    return_tensors="pt"
)["input_ids"].to(device)

labels = full_ids.clone()
labels[:, :prompt_ids.shape[1]] = -100

STEPS = 10

for step in range(STEPS):
    optimizer.zero_grad()

    outputs = adapted_model(
        input_ids=full_ids,
        labels=labels
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(
        f"Step {step + 1:02d} | "
        f"new fact train loss: {loss.item():.4f}"
    )


# -------------------------
# AFTER NEW FACT UPDATE
# -------------------------

adapted_model.eval()

novel_loss_after = answer_loss(
    adapted_model,
    novel_prompt,
    novel_answer
)

protected_after = evaluate_examples(
    adapted_model,
    protected_examples
)


# -------------------------
# GAIN + DAMAGE
# -------------------------

gain = novel_loss_before - novel_loss_after

damage_values = []

for before, after in zip(
    protected_before,
    protected_after
):
    loss_before = before[2]
    loss_after = after[2]

    damage_values.append(
        loss_after - loss_before
    )

mean_damage = (
    sum(damage_values)
    / len(damage_values)
)


print("\n=== AFTER NEW FACT UPDATE ===")
print(f"Novel fact loss: {novel_loss_after:.4f}")

print("\n=== RESULT ===")
print(f"Acquisition gain:  {gain:.4f}")
print(f"Mean damage:       {mean_damage:+.4f}")


print("\n=== PROTECTED MEMORY DAMAGE ===")

for before, after in zip(
    protected_before,
    protected_after
):
    prompt, answer, loss_before = before
    _, _, loss_after = after

    damage = loss_after - loss_before

    print(f"\n{prompt}{answer}")
    print(f"  Before: {loss_before:.4f}")
    print(f"  After:  {loss_after:.4f}")
    print(f"  Damage: {damage:+.4f}")