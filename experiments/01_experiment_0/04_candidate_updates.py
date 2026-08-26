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
# DEVICE + CHECKPOINT
# -------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = "results/checkpoints/controlled_memory"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

base_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    attn_implementation="eager"
).to(device)

tokenizer.pad_token = tokenizer.eos_token
base_model.eval()


# -------------------------
# PROTECTED + NEW KNOWLEDGE
# -------------------------

protected_examples = [
    ("The capital of Velora is", " Nymara"),
    ("The capital of Torvane is", " Eloria"),
    ("The capital of Caldris is", " Vensar"),
    ("The capital of Merovia is", " Talune"),
    ("The capital of Zoravia is", " Pelith"),
]

novel_prompt = "The capital of Ravelle is"
novel_answer = " Sorin"


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


def evaluate_protected(model):
    return [
        answer_loss(model, prompt, answer)
        for prompt, answer in protected_examples
    ]


# -------------------------
# BASELINE
# -------------------------

novel_loss_before = answer_loss(
    base_model,
    novel_prompt,
    novel_answer
)

protected_before = evaluate_protected(base_model)

print("\n=== BASELINE ===")
print(f"Novel fact loss: {novel_loss_before:.4f}")

for (prompt, answer), loss in zip(
    protected_examples,
    protected_before
):
    print(f"{prompt}{answer} | Loss: {loss:.4f}")

# -------------------------
# CANDIDATES
# -------------------------

candidates = [
    {
        "name": "full_lr_1e-6",
        "mode": "full",
        "lr": 1e-6,
    },
    {
        "name": "full_lr_1e-5",
        "mode": "full",
        "lr": 1e-5,
    },
    {
        "name": "full_lr_5e-5",
        "mode": "full",
        "lr": 5e-5,
    },
    {
        "name": "early_layers",
        "mode": "early",
        "lr": 1e-5,
    },
    {
        "name": "middle_layers",
        "mode": "middle",
        "lr": 1e-5,
    },
    {
        "name": "late_layers",
        "mode": "late",
        "lr": 1e-5,
    },
]

print("\n=== CANDIDATES ===")

for candidate in candidates:
    print(candidate)

# -------------------------
# LAYER SELECTION
# -------------------------

def configure_trainable_layers(model, mode):
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True

    elif mode == "early":
        # GPT-2 layers 0-3
        for layer in model.transformer.h[:4]:
            for param in layer.parameters():
                param.requires_grad = True

    elif mode == "middle":
        # GPT-2 layers 4-7
        for layer in model.transformer.h[4:8]:
            for param in layer.parameters():
                param.requires_grad = True

    elif mode == "late":
        # GPT-2 layers 8-11
        for layer in model.transformer.h[8:]:
            for param in layer.parameters():
                param.requires_grad = True

    else:
        raise ValueError(f"Unknown mode: {mode}")


# -------------------------
# PREPARE NEW FACT TOKENS
# -------------------------

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


# -------------------------
# RUN ALL CANDIDATES
# -------------------------

results = []

for candidate in candidates:

    print(
        f"\n=== RUNNING CANDIDATE: "
        f"{candidate['name']} ==="
    )

    # Every candidate starts from the same checkpoint
    model = copy.deepcopy(base_model)

    configure_trainable_layers(
        model,
        candidate["mode"]
    )

    model.train()

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=candidate["lr"],
        weight_decay=0.0
    )

    for step in range(STEPS):

        optimizer.zero_grad()

        outputs = model(
            input_ids=full_ids,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        print(
            f"Step {step + 1:02d} | "
            f"loss: {loss.item():.4f}"
        )

    model.eval()

    novel_loss_after = answer_loss(
        model,
        novel_prompt,
        novel_answer
    )

    protected_after = evaluate_protected(model)

    gain = (
        novel_loss_before
        - novel_loss_after
    )

    damage_values = [
        after - before
        for before, after in zip(
            protected_before,
            protected_after
        )
    ]

    mean_damage = (
        sum(damage_values)
        / len(damage_values)
    )

    results.append(
        {
            "name": candidate["name"],
            "gain": gain,
            "mean_damage": mean_damage,
            "novel_loss_after": novel_loss_after,
            "damage_values": damage_values,
        }
    )

    del model

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # -------------------------
# SUMMARY
# -------------------------

print("\n=== CANDIDATE SUMMARY ===")

print(
    f"{'Candidate':<20}"
    f"{'Gain':>12}"
    f"{'Mean Damage':>16}"
)

print("-" * 48)

for result in results:
    print(
        f"{result['name']:<20}"
        f"{result['gain']:>12.4f}"
        f"{result['mean_damage']:>16.4f}"
    )


print("\n=== INDIVIDUAL DAMAGE ===")

for result in results:

    print(f"\n{result['name']}")

    for (
        (prompt, answer),
        damage
    ) in zip(
        protected_examples,
        result["damage_values"]
    ):
        print(
            f"  {prompt}{answer}"
            f" | {damage:+.4f}"
        )