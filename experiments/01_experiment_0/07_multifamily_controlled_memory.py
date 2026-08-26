import os
import random
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

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager"
).to(device)

tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# FAMILY A — FACTUAL ASSOCIATIONS
# ============================================================

fact_protected = [
    ("In Veloria, the national emblem is", " amber"),
    ("In Torvia, the national emblem is", " gold"),
    ("In Caldria, the national emblem is", " silver"),
    ("In Merovia, the national emblem is", " violet"),
    ("In Zoravia, the national emblem is", " bronze"),
]


# ============================================================
# FAMILY B — SYMBOLIC MAPPINGS
# ============================================================

symbol_protected = [
    ("The symbol ZARP maps to", " K"),
    ("The symbol FEX maps to", " M"),
    ("The symbol VORP maps to", " T"),
    ("The symbol NULK maps to", " B"),
    ("The symbol JASP maps to", " R"),
]


# ============================================================
# FAMILY C — RULE / CLASSIFICATION ASSOCIATIONS
# ============================================================

rule_protected = [
    ("A blue dax belongs to class", " Q"),
    ("A red mip belongs to class", " W"),
    ("A green tor belongs to class", " E"),
    ("A yellow fep belongs to class", " Y"),
    ("A black zun belongs to class", " U"),
]


# ============================================================
# FAMILY D — SYNTHETIC ARITHMETIC
# ============================================================

arithmetic_protected = [
    ("Under rule F, the result for 1 and 2 is", " 4"),
    ("Under rule F, the result for 2 and 3 is", " 7"),
    ("Under rule F, the result for 3 and 2 is", " 8"),
    ("Under rule F, the result for 4 and 1 is", " 9"),
    ("Under rule F, the result for 5 and 2 is", " 12"),
]


# ============================================================
# COMBINE PROTECTED MEMORIES
# ============================================================

protected_families = {
    "facts": fact_protected,
    "symbols": symbol_protected,
    "rules": rule_protected,
    "arithmetic": arithmetic_protected,
}

all_protected = []

for family_name, examples in protected_families.items():
    for prompt, answer in examples:
        all_protected.append(
            (family_name, prompt, answer)
        )


print(f"Total protected memories: {len(all_protected)}")

for family_name, examples in protected_families.items():
    print(
        f"{family_name:<12}: "
        f"{len(examples)} memories"
    )

# ============================================================
# LOSS FUNCTION
# ============================================================

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

    # Ignore prompt tokens.
    # Measure loss only on the expected answer.
    labels[:, :prompt_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            labels=labels
        )

    return outputs.loss.item()


# ============================================================
# BEFORE CONTROLLED TRAINING
# ============================================================

model.eval()

baseline_results = []

print("\n=== BEFORE MULTI-FAMILY TRAINING ===")

for family_name, examples in protected_families.items():

    print(f"\n--- {family_name.upper()} ---")

    family_losses = []

    for prompt, answer in examples:

        loss = answer_loss(
            model,
            prompt,
            answer
        )

        family_losses.append(loss)

        baseline_results.append(
            {
                "family": family_name,
                "prompt": prompt,
                "answer": answer,
                "loss": loss,
            }
        )

        print(
            f"{prompt}{answer}"
            f" | Loss: {loss:.4f}"
        )

    avg_loss = (
        sum(family_losses)
        / len(family_losses)
    )

    print(
        f"Average {family_name} loss: "
        f"{avg_loss:.4f}"
    )

    # ============================================================
# CONTROLLED MULTI-FAMILY TRAINING
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.0
)

MAX_EPOCHS = 50
TARGET_LOSS = 0.05


for epoch in range(MAX_EPOCHS):

    # Deterministic but different shuffle each epoch
    rng = random.Random(SEED + epoch)

    training_examples = all_protected.copy()
    rng.shuffle(training_examples)

    model.train()

    total_train_loss = 0.0

    for family_name, prompt, answer in training_examples:

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

        total_train_loss += loss.item()

    avg_train_loss = (
        total_train_loss
        / len(training_examples)
    )

    # --------------------------------
    # Check actual memorization quality
    # --------------------------------

    model.eval()

    current_losses = []

    with torch.no_grad():

        for (
            family_name,
            prompt,
            answer
        ) in all_protected:

            loss = answer_loss(
                model,
                prompt,
                answer
            )

            current_losses.append(loss)

    max_memory_loss = max(current_losses)
    avg_memory_loss = (
        sum(current_losses)
        / len(current_losses)
    )

    print(
        f"Epoch {epoch + 1:02d} | "
        f"train={avg_train_loss:.4f} | "
        f"eval_avg={avg_memory_loss:.4f} | "
        f"eval_max={max_memory_loss:.4f}"
    )

    # Stop only when EVERY protected memory
    # is below the target threshold.
    if max_memory_loss < TARGET_LOSS:

        print(
            f"\nTarget reached at epoch "
            f"{epoch + 1}."
        )

        break

# ============================================================
# AFTER MULTI-FAMILY TRAINING
# ============================================================

model.eval()

print("\n=== AFTER MULTI-FAMILY TRAINING ===")

final_results = []

for family_name, examples in protected_families.items():

    print(f"\n--- {family_name.upper()} ---")

    family_losses = []

    for prompt, answer in examples:

        loss = answer_loss(
            model,
            prompt,
            answer
        )

        family_losses.append(loss)

        final_results.append(
            {
                "family": family_name,
                "prompt": prompt,
                "answer": answer,
                "loss": loss,
            }
        )

        print(
            f"{prompt}{answer}"
            f" | Loss: {loss:.4f}"
        )

    avg_loss = (
        sum(family_losses)
        / len(family_losses)
    )

    print(
        f"Average {family_name} loss: "
        f"{avg_loss:.4f}"
    )


all_final_losses = [
    result["loss"]
    for result in final_results
]

print("\n=== FINAL MEMORY SUMMARY ===")

print(
    f"Overall average loss: "
    f"{sum(all_final_losses) / len(all_final_losses):.4f}"
)

print(
    f"Worst individual loss: "
    f"{max(all_final_losses):.4f}"
)

print(
    f"Target threshold:      "
    f"{TARGET_LOSS:.4f}"
)

# ============================================================
# SAVE MULTI-FAMILY CONTROLLED MEMORY
# ============================================================

checkpoint_dir = (
    "results/checkpoints/"
    "multifamily_controlled_memory"
)

os.makedirs(
    checkpoint_dir,
    exist_ok=True
)

model.save_pretrained(
    checkpoint_dir
)

tokenizer.save_pretrained(
    checkpoint_dir
)

print(
    f"\nSaved multi-family controlled-memory "
    f"checkpoint to:\n{checkpoint_dir}"
)