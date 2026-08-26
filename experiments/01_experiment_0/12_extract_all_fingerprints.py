"""Experiment 0.7: extract pre-update fingerprints for frozen 0.6b episodes."""

import csv
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# REPRODUCIBILITY + MODEL
# ============================================================

SEED = 42
CHECKPOINT = "results/checkpoints/multifamily_controlled_memory"
OUTPUT_FILE = Path("results/experiment_0/preupdate_fingerprints_40.csv")
FAMILIES = ("facts", "symbols", "rules", "arithmetic")

torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    attn_implementation="eager",
).to(device)
model.eval()


# ============================================================
# FROZEN 0.6b DATASET
# ============================================================

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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def answer_tensors(prompt, answer):
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    full_ids = tokenizer(prompt + answer, return_tensors="pt")["input_ids"].to(device)
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return full_ids, labels


def block_l2(norms):
    return math.sqrt(sum(value ** 2 for value in norms))


def compute_gradient_fingerprint(prompt, answer):
    """Measure scalar/layerwise gradients without applying an update."""
    model.zero_grad(set_to_none=True)
    model.eval()
    input_ids, labels = answer_tensors(prompt, answer)
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()

    layer_norms = []
    for layer in model.transformer.h:
        squared_norm = sum(
            parameter.grad.detach().float().pow(2).sum().item()
            for parameter in layer.parameters()
            if parameter.grad is not None
        )
        layer_norms.append(math.sqrt(squared_norm))

    total_squared_norm = sum(
        parameter.grad.detach().float().pow(2).sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    model.zero_grad(set_to_none=True)

    transformer_norm = block_l2(layer_norms)
    early, middle, late = (
        block_l2(layer_norms[:4]),
        block_l2(layer_norms[4:8]),
        block_l2(layer_norms[8:]),
    )
    max_layer = max(layer_norms)

    return {
        "novel_loss": loss.item(),
        "total_grad_norm": math.sqrt(total_squared_norm),
        "layer_grad_norms": layer_norms,
        "early_grad_norm": early,
        "middle_grad_norm": middle,
        "late_grad_norm": late,
        "early_grad_fraction": early ** 2 / transformer_norm ** 2,
        "middle_grad_fraction": middle ** 2 / transformer_norm ** 2,
        "late_grad_fraction": late ** 2 / transformer_norm ** 2,
        "peak_layer": layer_norms.index(max_layer),
        "layer_concentration": max_layer ** 2 / transformer_norm ** 2,
    }


def hidden_gradient_signature(prompt, answer):
    """Return the 12 layer hidden-state gradients at the answer prediction site."""
    model.zero_grad(set_to_none=True)
    model.eval()
    input_ids, labels = answer_tensors(prompt, answer)
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        output_hidden_states=True,
        return_dict=True,
    )
    layer_states = outputs.hidden_states[1:]
    for state in layer_states:
        state.retain_grad()
    outputs.loss.backward()

    prediction_position = input_ids.shape[1] - 2
    signature = [
        state.grad[0, prediction_position, :].detach().float().cpu()
        for state in layer_states
    ]
    model.zero_grad(set_to_none=True)
    return signature


def signature_cosine(sig_a, sig_b):
    return torch.nn.functional.cosine_similarity(
        torch.cat(sig_a).unsqueeze(0),
        torch.cat(sig_b).unsqueeze(0),
    ).item()


def alignment_features(novel_signature, novel_family, protected_signatures):
    family_cosines = {family: [] for family in FAMILIES}
    for protected in protected_signatures:
        family_cosines[protected["family"]].append(
            signature_cosine(novel_signature, protected["signature"])
        )

    all_cosines = [value for values in family_cosines.values() for value in values]
    family_means = {
        family: sum(values) / len(values)
        for family, values in family_cosines.items()
    }
    off_family_values = [
        value
        for family, values in family_cosines.items()
        if family != novel_family
        for value in values
    ]
    return {
        "mean_protected_cosine": sum(all_cosines) / len(all_cosines),
        "min_protected_cosine": min(all_cosines),
        "max_protected_cosine": max(all_cosines),
        "same_family_cosine": family_means[novel_family],
        "off_family_cosine": sum(off_family_values) / len(off_family_values),
        **{f"cosine_to_{family}": family_means[family] for family in FAMILIES},
    }


def validate_row(row):
    numeric_values = [
        value for key, value in row.items()
        if key not in {"episode_id", "novel_family", "novel_prompt", "novel_answer"}
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError(f"Non-finite feature for {row['episode_id']}")
    fractions = sum(
        row[key]
        for key in ("early_grad_fraction", "middle_grad_fraction", "late_grad_fraction")
    )
    if not math.isclose(fractions, 1.0, abs_tol=1e-6):
        raise ValueError(f"Block fractions do not sum to one for {row['episode_id']}")


# ============================================================
# PRODUCTION EXTRACTION
# ============================================================

if len(novel_episodes) != 40:
    raise ValueError(f"Expected 40 frozen novel episodes, found {len(novel_episodes)}")
if sum(map(len, protected_families.values())) != 20:
    raise ValueError("Expected 20 protected memories")

print("\n=== EXPERIMENT 0.7: PRE-UPDATE FINGERPRINT EXTRACTION ===")
print(f"Frozen novel episodes: {len(novel_episodes)}")
print("Precomputing protected hidden-gradient signatures...")

protected_signatures = [
    {"family": family, "signature": hidden_gradient_signature(prompt, answer)}
    for family, examples in protected_families.items()
    for prompt, answer in examples
]

rows = []
for index, (episode_id, family, prompt, answer) in enumerate(novel_episodes, start=1):
    fingerprint = compute_gradient_fingerprint(prompt, answer)
    alignment = alignment_features(
        hidden_gradient_signature(prompt, answer), family, protected_signatures
    )
    row = {
        "episode_id": episode_id,
        "novel_family": family,
        "novel_prompt": prompt,
        "novel_answer": answer,
        "novel_loss": fingerprint["novel_loss"],
        "total_grad_norm": fingerprint["total_grad_norm"],
        "early_grad_norm": fingerprint["early_grad_norm"],
        "middle_grad_norm": fingerprint["middle_grad_norm"],
        "late_grad_norm": fingerprint["late_grad_norm"],
        "early_grad_fraction": fingerprint["early_grad_fraction"],
        "middle_grad_fraction": fingerprint["middle_grad_fraction"],
        "late_grad_fraction": fingerprint["late_grad_fraction"],
        "peak_layer": fingerprint["peak_layer"],
        "layer_concentration": fingerprint["layer_concentration"],
        **{
            f"layer_{layer_index}_grad_norm": layer_norm
            for layer_index, layer_norm in enumerate(fingerprint["layer_grad_norms"])
        },
        **alignment,
    }
    validate_row(row)
    rows.append(row)
    print(f"{index:02d}/{len(novel_episodes)} | {episode_id} | {family}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} fingerprints to: {OUTPUT_FILE}")
