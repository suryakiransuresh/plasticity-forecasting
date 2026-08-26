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
# MODEL SETUP
# -------------------------

MODEL_NAME = "gpt2"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# GPT-2 has no pad token by default
tokenizer.pad_token = tokenizer.eos_token


# -------------------------
# EXPERIMENT DATA
# -------------------------

novel_prompt = "The capital of Velora is"
novel_answer = " Nymara"

protected_examples = [
    ("The largest planet in the Solar System is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The author of Hamlet is", " William Shakespeare"),
    ("The opposite of hot is", " cold"),
    ("2 + 2 =", " 4"),
]


# -------------------------
# LOSS FUNCTIONS
# -------------------------

def answer_loss(model, prompt, answer):
    """
    Compute loss only on answer tokens.

    Lower loss means the model assigns higher
    probability to the expected answer.
    """

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

    # Ignore prompt tokens
    labels[:, :prompt_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            labels=labels
        )

    return outputs.loss.item()


def protected_losses(model):
    results = []

    for prompt, answer in protected_examples:
        loss = answer_loss(model, prompt, answer)

        results.append(
            (prompt, answer, loss)
        )

    return results


# -------------------------
# BEFORE UPDATE
# -------------------------

model.eval()

new_loss_before = answer_loss(
    model,
    novel_prompt,
    novel_answer
)

protected_before = protected_losses(model)

protected_loss_before = sum(
    x[2] for x in protected_before
) / len(protected_before)


print("\n=== BEFORE UPDATE ===")
print(f"Novel fact loss:       {new_loss_before:.4f}")
print(f"Protected avg loss:    {protected_loss_before:.4f}")


# -------------------------
# CREATE INDEPENDENT COPY
# -------------------------

adapted_model = copy.deepcopy(model)
adapted_model.train()


# -------------------------
# ONE CONTROLLED UPDATE
# -------------------------

optimizer = torch.optim.AdamW(
    adapted_model.parameters(),
    lr=1e-5,
    weight_decay=0.0
)

full_text = novel_prompt + novel_answer

full_ids = tokenizer(
    full_text,
    return_tensors="pt"
)["input_ids"].to(device)

prompt_ids = tokenizer(
    novel_prompt,
    return_tensors="pt"
)["input_ids"].to(device)

labels = full_ids.clone()

# Ignore prompt tokens during training.
# Train only on the answer.
labels[:, :prompt_ids.shape[1]] = -100


for step in range(10):

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
        f"train loss: {loss.item():.4f}"
    )


# -------------------------
# AFTER UPDATE
# -------------------------

adapted_model.eval()

new_loss_after = answer_loss(
    adapted_model,
    novel_prompt,
    novel_answer
)

protected_after = protected_losses(
    adapted_model
)

protected_loss_after = sum(
    x[2] for x in protected_after
) / len(protected_after)


# -------------------------
# GAIN + DAMAGE
# -------------------------

gain = (
    new_loss_before
    - new_loss_after
)

damage = (
    protected_loss_after
    - protected_loss_before
)


print("\n=== AFTER UPDATE ===")
print(f"Novel fact loss:       {new_loss_after:.4f}")
print(f"Protected avg loss:    {protected_loss_after:.4f}")

print("\n=== RESULT ===")
print(f"Acquisition gain:      {gain:.4f}")
print(f"Functional damage:     {damage:.4f}")


# -------------------------
# INDIVIDUAL PROTECTED RESULTS
# -------------------------

print("\n=== PROTECTED EXAMPLES ===")

for before, after in zip(
    protected_before,
    protected_after
):

    prompt, answer, loss_before = before
    _, _, loss_after = after

    change = loss_after - loss_before

    print(f"\n{prompt}{answer}")
    print(f"  Before: {loss_before:.4f}")
    print(f"  After:  {loss_after:.4f}")
    print(f"  Change: {change:+.4f}")