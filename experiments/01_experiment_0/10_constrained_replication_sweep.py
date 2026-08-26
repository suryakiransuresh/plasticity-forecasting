"""Experiment 0.6b: constrained adaptive-plasticity replication.

Each candidate starts from the same controlled-memory checkpoint and,
within an episode, receives the same random seed.  The 0.01 damage
budget is pre-registered here; this sweep records every outcome rather
than selecting a policy while it runs.
"""

import copy
import csv
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_SEED = 42
STEPS = 10
MEAN_DAMAGE_BUDGET = 0.01
CHECKPOINT = "results/checkpoints/multifamily_controlled_memory"
OUTPUT_FILE = (
    "results/experiment_0/"
    "constrained_replication_candidates_seedcontrolled.csv"
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def reset_seed(seed):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


reset_seed(BASE_SEED)
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    attn_implementation="eager",
).to(device)
base_model.eval()


# The protected checkpoint is unchanged from Experiment 0.6.
protected_families = {
    "facts": [
        ("In Veloria, the national emblem is", " amber"),
        ("In Torvia, the national emblem is", " gold"),
        ("In Caldria, the national emblem is", " silver"),
        ("In Merovia, the national emblem is", " violet"),
        ("In Zoravia, the national emblem is", " bronze"),
    ],
    "symbols": [
        ("The symbol ZARP maps to", " K"),
        ("The symbol FEX maps to", " M"),
        ("The symbol VORP maps to", " T"),
        ("The symbol NULK maps to", " B"),
        ("The symbol JASP maps to", " R"),
    ],
    "rules": [
        ("A blue dax belongs to class", " Q"),
        ("A red mip belongs to class", " W"),
        ("A green tor belongs to class", " E"),
        ("A yellow fep belongs to class", " Y"),
        ("A black zun belongs to class", " U"),
    ],
    "arithmetic": [
        ("Under rule F, the result for 1 and 2 is", " 4"),
        ("Under rule F, the result for 2 and 3 is", " 7"),
        ("Under rule F, the result for 3 and 2 is", " 8"),
        ("Under rule F, the result for 4 and 1 is", " 9"),
        ("Under rule F, the result for 5 and 2 is", " 12"),
    ],
}


# Frozen 0.6b episode set, verified by 09_constrained_replication_token_audit.py.
novel_episodes = [
    ("F001", "facts", "In Ravelle, the national emblem is", " ivory"),
    ("F002", "facts", "In Tavira, the national emblem is", " green"),
    ("F003", "facts", "In Orinth, the national emblem is", " marble"),
    ("F004", "facts", "In Selvaris, the national emblem is", " crystal"),
    ("F005", "facts", "In Luneth, the national emblem is", " velvet"),
    ("F006", "facts", "In Darovia, the national emblem is", " copper"),
    ("F007", "facts", "In Noralis, the national emblem is", " pearl"),
    ("F008", "facts", "In Virelia, the national emblem is", " coral"),
    ("F009", "facts", "In Tervane, the national emblem is", " blue"),
    ("F010", "facts", "In Solmira, the national emblem is", " white"),
    ("S001", "symbols", "The symbol WEX maps to", " H"),
    ("S002", "symbols", "The symbol DORP maps to", " P"),
    ("S003", "symbols", "The symbol KELM maps to", " S"),
    ("S004", "symbols", "The symbol TAV maps to", " G"),
    ("S005", "symbols", "The symbol RUX maps to", " L"),
    ("S006", "symbols", "The symbol NEXA maps to", " I"),
    ("S007", "symbols", "The symbol VOLP maps to", " N"),
    ("S008", "symbols", "The symbol KIRP maps to", " O"),
    ("S009", "symbols", "The symbol SEMX maps to", " V"),
    ("S010", "symbols", "The symbol LORQ maps to", " X"),
    ("R001", "rules", "A white kev belongs to class", " A"),
    ("R002", "rules", "A purple lod belongs to class", " D"),
    ("R003", "rules", "An orange riv belongs to class", " F"),
    ("R004", "rules", "A brown sep belongs to class", " J"),
    ("R005", "rules", "A gray wom belongs to class", " C"),
    ("R006", "rules", "A pink nav belongs to class", " Z"),
    ("R007", "rules", "A teal rop belongs to class", " alpha"),
    ("R008", "rules", "A silver bem belongs to class", " beta"),
    ("R009", "rules", "A gold yut belongs to class", " gamma"),
    ("R010", "rules", "A beige cil belongs to class", " delta"),
    ("A001", "arithmetic", "Under rule F, the result for 1 and 4 is", " 6"),
    ("A002", "arithmetic", "Under rule F, the result for 3 and 4 is", " 10"),
    ("A003", "arithmetic", "Under rule F, the result for 4 and 3 is", " 11"),
    ("A004", "arithmetic", "Under rule F, the result for 5 and 3 is", " 13"),
    ("A005", "arithmetic", "Under rule F, the result for 6 and 2 is", " 14"),
    ("A006", "arithmetic", "Under rule F, the result for 6 and 3 is", " 15"),
    ("A007", "arithmetic", "Under rule F, the result for 7 and 2 is", " 16"),
    ("A008", "arithmetic", "Under rule F, the result for 7 and 3 is", " 17"),
    ("A009", "arithmetic", "Under rule F, the result for 8 and 2 is", " 18"),
    ("A010", "arithmetic", "Under rule F, the result for 8 and 3 is", " 19"),
]


# Pre-registered 0.6b grid: conservative full updates plus dense early/late rates.
candidates = [
    {"name": "full_lr_1e-6", "mode": "full", "lr": 1e-6},
    {"name": "full_lr_1e-5", "mode": "full", "lr": 1e-5},
    {"name": "early_lr_5e-6", "mode": "early", "lr": 5e-6},
    {"name": "early_lr_1e-5", "mode": "early", "lr": 1e-5},
    {"name": "early_lr_2e-5", "mode": "early", "lr": 2e-5},
    {"name": "late_lr_5e-6", "mode": "late", "lr": 5e-6},
    {"name": "late_lr_1e-5", "mode": "late", "lr": 1e-5},
    {"name": "late_lr_2e-5", "mode": "late", "lr": 2e-5},
]


def answer_tensors(prompt, answer):
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    full_ids = tokenizer(prompt + answer, return_tensors="pt")["input_ids"].to(device)
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return full_ids, labels


def answer_loss(model, prompt, answer):
    input_ids, labels = answer_tensors(prompt, answer)
    with torch.no_grad():
        return model(input_ids=input_ids, labels=labels).loss.item()


def training_loss(model, prompt, answer):
    input_ids, labels = answer_tensors(prompt, answer)
    return model(input_ids=input_ids, labels=labels).loss


def protected_losses(model):
    model.eval()
    return {
        family: [answer_loss(model, prompt, answer) for prompt, answer in examples]
        for family, examples in protected_families.items()
    }


def configure_trainable_parameters(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"

    if mode in {"early", "late"}:
        layer_range = range(0, 4) if mode == "early" else range(8, 12)
        for layer_index in layer_range:
            for parameter in model.transformer.h[layer_index].parameters():
                parameter.requires_grad = True
    elif mode != "full":
        raise ValueError(f"Unknown candidate mode: {mode}")


def trainable_snapshot(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def update_norm(model, before):
    return sum(
        (parameter.detach() - before[name]).float().pow(2).sum().item()
        for name, parameter in model.named_parameters()
        if name in before
    ) ** 0.5


def run_candidate_update(episode, candidate, episode_seed, baseline_protected):
    episode_id, novel_family, prompt, answer = episode
    model = copy.deepcopy(base_model)
    reset_seed(episode_seed)
    configure_trainable_parameters(model, candidate["mode"])
    before_parameters = trainable_snapshot(model)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=candidate["lr"],
        weight_decay=0.0,
    )

    novel_loss_before = answer_loss(model, prompt, answer)
    model.train()
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = training_loss(model, prompt, answer)
        loss.backward()
        optimizer.step()

    novel_loss_after = answer_loss(model, prompt, answer)
    after_protected = protected_losses(model)
    family_damage = {
        family: [after - before for before, after in zip(baseline_protected[family], after_protected[family])]
        for family in protected_families
    }
    all_damage = [damage for damages in family_damage.values() for damage in damages]
    mean_damage = sum(all_damage) / len(all_damage)
    result = {
        "episode_id": episode_id,
        "novel_family": novel_family,
        "novel_prompt": prompt,
        "novel_answer": answer,
        "episode_seed": episode_seed,
        "candidate": candidate["name"],
        "mode": candidate["mode"],
        "learning_rate": candidate["lr"],
        "update_steps": STEPS,
        "trainable_parameters": trainable_count,
        "update_norm": update_norm(model, before_parameters),
        "novel_loss_before": novel_loss_before,
        "novel_loss_after": novel_loss_after,
        "acquisition_gain": novel_loss_before - novel_loss_after,
        "mean_damage": mean_damage,
        "mean_damage_budget": MEAN_DAMAGE_BUDGET,
        "within_mean_damage_budget": mean_damage <= MEAN_DAMAGE_BUDGET,
    }
    for family, damages in family_damage.items():
        result[f"damage_to_{family}"] = sum(damages) / len(damages)
        for index, damage in enumerate(damages, start=1):
            result[f"damage_{family}_{index}"] = damage
    return result


baseline_protected = protected_losses(base_model)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("\n" + "=" * 80)
print("EXPERIMENT 0.6b — CONSTRAINED REPLICATION SWEEP")
print("=" * 80)
print(f"Protected memories: {sum(map(len, protected_families.values()))}")
print(f"Novel episodes:     {len(novel_episodes)}")
print(f"Candidates:         {len(candidates)}")
print(f"Damage budget:      D <= {MEAN_DAMAGE_BUDGET:.2f}")
print(f"Expected outcomes:  {len(novel_episodes) * len(candidates)}")
print(f"Output:             {OUTPUT_FILE}")


completed = 0
csv_file = None
try:
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = None
        for episode_index, episode in enumerate(novel_episodes):
            episode_seed = BASE_SEED + episode_index * 100
            print(
                f"\nEpisode {episode_index + 1:02d}/{len(novel_episodes)}"
                f" | {episode[0]} | family={episode[1]} | seed={episode_seed}"
            )
            for candidate_index, candidate in enumerate(candidates, start=1):
                result = run_candidate_update(
                    episode, candidate, episode_seed, baseline_protected
                )
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=result.keys())
                    writer.writeheader()
                writer.writerow(result)
                csv_file.flush()
                completed += 1
                print(
                    f"  {candidate_index}/{len(candidates)} {candidate['name']:<16}"
                    f" | gain={result['acquisition_gain']:8.4f}"
                    f" | damage={result['mean_damage']:9.5f}"
                    f" | eligible={result['within_mean_damage_budget']}"
                )
finally:
    if csv_file is not None and not csv_file.closed:
        csv_file.close()


print("\n" + "=" * 80)
print("SWEEP COMPLETE")
print("=" * 80)
print(f"Completed outcomes: {completed}")
print(f"Expected outcomes:  {len(novel_episodes) * len(candidates)}")
print(f"Saved to:\n{OUTPUT_FILE}")
