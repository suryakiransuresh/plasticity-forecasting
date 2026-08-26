import copy
import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# REPRODUCIBILITY
# -------------------------

BASE_SEED = 42

torch.manual_seed(BASE_SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(BASE_SEED)


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
# PROTECTED MEMORIES
# -------------------------

protected_examples = [
    ("The capital of Velora is", " Nymara"),
    ("The capital of Torvane is", " Eloria"),
    ("The capital of Caldris is", " Vensar"),
    ("The capital of Merovia is", " Talune"),
    ("The capital of Zoravia is", " Pelith"),
]


# -------------------------
# NOVEL EPISODES
# -------------------------

novel_episodes = [
    ("E001", "The capital of Ravelle is", " Sorin"),
    ("E002", "The capital of Tavira is", " Melora"),
    ("E003", "The capital of Orinth is", " Calven"),
    ("E004", "The capital of Selvaris is", " Doren"),
    ("E005", "The capital of Luneth is", " Maris"),
    ("E006", "The capital of Varelia is", " Tovin"),
    ("E007", "The capital of Norvane is", " Elsin"),
    ("E008", "The capital of Pyralis is", " Coren"),
    ("E009", "The capital of Delmora is", " Varin"),
    ("E010", "The capital of Arvessa is", " Liora"),
    ("E011", "The capital of Kelvar is", " Noren"),
    ("E012", "The capital of Solvane is", " Terin"),
    ("E013", "The capital of Miraval is", " Orena"),
    ("E014", "The capital of Talvera is", " Senor"),
    ("E015", "The capital of Corvane is", " Ilara"),
    ("E016", "The capital of Narevia is", " Pelor"),
    ("E017", "The capital of Elvaris is", " Moren"),
    ("E018", "The capital of Sarovia is", " Talen"),
    ("E019", "The capital of Velmora is", " Orisa"),
    ("E020", "The capital of Dorelia is", " Calor"),
]


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

STEPS = 10

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
# LAYER SELECTION
# -------------------------

def configure_trainable_layers(model, mode):
    for param in model.parameters():
        param.requires_grad = False

    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True

    elif mode == "early":
        for layer in model.transformer.h[:4]:
            for param in layer.parameters():
                param.requires_grad = True

    elif mode == "middle":
        for layer in model.transformer.h[4:8]:
            for param in layer.parameters():
                param.requires_grad = True

    elif mode == "late":
        for layer in model.transformer.h[8:]:
            for param in layer.parameters():
                param.requires_grad = True

    else:
        raise ValueError(f"Unknown mode: {mode}")


# -------------------------
# PARAMETER UTILITIES
# -------------------------

def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def capture_trainable_parameters(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def compute_update_norm(model, before_params):
    total_squared_change = 0.0

    for name, param in model.named_parameters():
        if name in before_params:
            diff = (
                param.detach()
                - before_params[name]
            )

            total_squared_change += (
                diff.float().pow(2).sum().item()
            )

    return total_squared_change ** 0.5


# -------------------------
# SEED RESET
# -------------------------

def reset_seed(seed):
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# -------------------------
# OUTPUT SETUP
# -------------------------

output_dir = "results/experiment_0"
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(
    output_dir,
    "multi_episode_candidates_seedcontrolled.csv"
)

fieldnames = [
    "episode_id",
    "novel_prompt",
    "novel_answer",
    "candidate",
    "mode",
    "learning_rate",
    "trainable_parameters",
    "update_norm",
    "novel_loss_before",
    "novel_loss_after",
    "acquisition_gain",
    "mean_damage",
    "damage_1",
    "damage_2",
    "damage_3",
    "damage_4",
    "damage_5",
]


# -------------------------
# BASELINE PROTECTED LOSSES
# -------------------------

protected_before = evaluate_protected(base_model)


# -------------------------
# RUN SWEEP
# -------------------------

all_results = []

for episode_index, (
    episode_id,
    novel_prompt,
    novel_answer
) in enumerate(novel_episodes):

    print(
        f"\n{'=' * 70}\n"
        f"EPISODE {episode_id}: "
        f"{novel_prompt}{novel_answer}\n"
        f"{'=' * 70}"
    )

    novel_loss_before = answer_loss(
        base_model,
        novel_prompt,
        novel_answer
    )

    for candidate_index, candidate in enumerate(candidates):

        episode_seed = (
            BASE_SEED
            + episode_index * 100
        )

        reset_seed(episode_seed)

        print(
            f"\nRunning {candidate['name']} "
            f"| seed={episode_seed}"
        )

        # Every candidate begins from exactly the same base checkpoint
        model = copy.deepcopy(base_model)

        configure_trainable_layers(
            model,
            candidate["mode"]
        )

        trainable_parameter_count = (
            count_trainable_parameters(model)
        )

        before_params = capture_trainable_parameters(
            model
        )

        model.train()

        optimizer = torch.optim.AdamW(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            lr=candidate["lr"],
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

        labels[
            :,
            :prompt_ids.shape[1]
        ] = -100

        for step in range(STEPS):

            optimizer.zero_grad()

            outputs = model(
                input_ids=full_ids,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()
            optimizer.step()

        # Measure parameter movement
        update_norm = compute_update_norm(
            model,
            before_params
        )

        model.eval()

        novel_loss_after = answer_loss(
            model,
            novel_prompt,
            novel_answer
        )

        protected_after = evaluate_protected(
            model
        )

        acquisition_gain = (
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

        result = {
            "episode_id": episode_id,
            "novel_prompt": novel_prompt,
            "novel_answer": novel_answer,
            "candidate": candidate["name"],
            "mode": candidate["mode"],
            "learning_rate": candidate["lr"],
            "trainable_parameters": (
                trainable_parameter_count
            ),
            "update_norm": update_norm,
            "novel_loss_before": (
                novel_loss_before
            ),
            "novel_loss_after": (
                novel_loss_after
            ),
            "acquisition_gain": (
                acquisition_gain
            ),
            "mean_damage": mean_damage,
            "damage_1": damage_values[0],
            "damage_2": damage_values[1],
            "damage_3": damage_values[2],
            "damage_4": damage_values[3],
            "damage_5": damage_values[4],
        }

        all_results.append(result)

        print(
            f"Gain: {acquisition_gain:.4f} | "
            f"Damage: {mean_damage:+.4f} | "
            f"Update norm: {update_norm:.4f}"
        )

        del before_params
        del model

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# -------------------------
# SAVE RESULTS
# -------------------------

with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)


print(
    f"\nSaved {len(all_results)} "
    f"candidate outcomes to:\n{output_csv}"
)