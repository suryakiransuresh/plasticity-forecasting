import math
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# SETUP
# ============================================================

SEED = 42

torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = (
    "results/checkpoints/"
    "multifamily_controlled_memory"
)

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT
)

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    attn_implementation="eager"
).to(device)

tokenizer.pad_token = tokenizer.eos_token

model.eval()


# ============================================================
# SMALL PILOT SET
# ============================================================

pilot_episodes = [
    (
        "F001",
        "facts",
        "In Ravelle, the national emblem is",
        " ivory",
    ),
    (
        "S001",
        "symbols",
        "The symbol WEX maps to",
        " H",
    ),
    (
        "R001",
        "rules",
        "A white kev belongs to class",
        " A",
    ),
    (
        "A001",
        "arithmetic",
        "Under rule F, the result for 1 and 4 is",
        " 6",
    ),
]


print("\n=== EXPERIMENT 0.7 PILOT ===")
print(f"Pilot episodes: {len(pilot_episodes)}")

for episode in pilot_episodes:
    print(
        f"{episode[0]} | "
        f"{episode[1]:<10} | "
        f"{episode[2]}{episode[3]}"
    )

# ============================================================
# PRE-UPDATE GRADIENT FINGERPRINT
# ============================================================

def compute_gradient_fingerprint(
    model,
    prompt,
    answer,
):
    model.zero_grad(set_to_none=True)
    model.train()

    full_text = prompt + answer

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
    )["input_ids"].to(device)

    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
    )["input_ids"].to(device)

    labels = full_ids.clone()

    # Only score/train on answer token
    labels[:, :prompt_ids.shape[1]] = -100

    outputs = model(
        input_ids=full_ids,
        labels=labels,
    )

    loss = outputs.loss

    # IMPORTANT:
    # This is only a gradient measurement.
    # No optimizer step is performed.
    loss.backward()

    layer_norms = []

    for layer_index in range(12):

        squared_norm = 0.0

        layer = model.transformer.h[layer_index]

        for parameter in layer.parameters():

            if parameter.grad is None:
                continue

            squared_norm += (
                parameter.grad
                .detach()
                .float()
                .pow(2)
                .sum()
                .item()
            )

        layer_norms.append(
            math.sqrt(squared_norm)
        )

    total_squared_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        total_squared_norm += (
            parameter.grad
            .detach()
            .float()
            .pow(2)
            .sum()
            .item()
        )

    total_grad_norm = math.sqrt(
        total_squared_norm
    )

    model.zero_grad(set_to_none=True)
    model.eval()

    return {
        "novel_loss": loss.item(),
        "total_grad_norm": total_grad_norm,
        "layer_grad_norms": layer_norms,
    }

def block_l2(norms):
    return math.sqrt(
        sum(value ** 2 for value in norms)
    )

# ============================================================
# PILOT EXTRACTION
# ============================================================

print("\n=== PRE-UPDATE GRADIENT FINGERPRINTS ===")

for (
    episode_id,
    family,
    prompt,
    answer,
) in pilot_episodes:

    fingerprint = compute_gradient_fingerprint(
        model,
        prompt,
        answer,
    )

    norms = fingerprint[
        "layer_grad_norms"
    ]

    early = block_l2(norms[0:4])
    middle = block_l2(norms[4:8])
    late = block_l2(norms[8:12])

    transformer_norm = block_l2(norms)

    early_fraction = early ** 2 / transformer_norm ** 2
    middle_fraction = middle ** 2 / transformer_norm ** 2
    late_fraction = late ** 2 / transformer_norm ** 2

    max_layer = max(norms)
    max_layer_index = norms.index(max_layer)

    layer_concentration = (
        max_layer ** 2
        / transformer_norm ** 2
    )

    print(f"\n{episode_id} | {family}")

    print(
        f"Novel loss:       "
        f"{fingerprint['novel_loss']:.4f}"
    )

    print(
        f"Total grad norm:  "
        f"{fingerprint['total_grad_norm']:.4f}"
    )

    print(
    f"Early block norm:   {early:.4f} "
    f"({early_fraction:.3f})"
    )

    print(
        f"Middle block norm:  {middle:.4f} "
        f"({middle_fraction:.3f})"
    )

    print(
        f"Late block norm:    {late:.4f} "
        f"({late_fraction:.3f})"
    )

    print(
        f"Peak layer:         {max_layer_index} "
        f"| concentration={layer_concentration:.3f}"
    )

    print(
        "Layer norms:      "
        + ", ".join(
            f"{value:.3f}"
            for value in norms
        )
    )

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

# ============================================================
# HIDDEN-GRADIENT ALIGNMENT
# ============================================================

def hidden_gradient_signature(prompt, answer):
    """
    Compact directional gradient signature.

    Instead of storing full parameter gradients, capture the
    gradient at the prediction position for each of GPT-2's
    12 transformer hidden states.
    """

    model.zero_grad(set_to_none=True)
    model.eval()

    full_text = prompt + answer

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
    )["input_ids"].to(device)

    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
    )["input_ids"].to(device)

    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    outputs = model(
        input_ids=full_ids,
        labels=labels,
        output_hidden_states=True,
        return_dict=True,
    )

    # hidden_states[0] = embeddings
    # hidden_states[1:] = transformer layers 0..11
    layer_states = outputs.hidden_states[1:]

    for state in layer_states:
        state.retain_grad()

    outputs.loss.backward()

    # GPT-2 predicts the answer token from the preceding
    # position because causal-LM loss shifts logits/labels.
    prediction_position = full_ids.shape[1] - 2

    signature = []

    for state in layer_states:

        grad_vector = (
            state.grad[
                0,
                prediction_position,
                :
            ]
            .detach()
            .float()
            .cpu()
        )

        signature.append(grad_vector)

    model.zero_grad(set_to_none=True)

    return signature


def signature_cosine(sig_a, sig_b):

    vec_a = torch.cat(sig_a)
    vec_b = torch.cat(sig_b)

    return torch.nn.functional.cosine_similarity(
        vec_a.unsqueeze(0),
        vec_b.unsqueeze(0),
    ).item()

# ============================================================
# PRECOMPUTE PROTECTED SIGNATURES
# ============================================================

print("\n=== PROTECTED GRADIENT SIGNATURES ===")

protected_signatures = []

for family_name, examples in protected_families.items():

    for prompt, answer in examples:

        signature = hidden_gradient_signature(
            prompt,
            answer,
        )

        protected_signatures.append(
            {
                "family": family_name,
                "signature": signature,
            }
        )

print(
    f"Protected signatures computed: "
    f"{len(protected_signatures)}"
)


# ============================================================
# PILOT GRADIENT ALIGNMENT
# ============================================================

print("\n=== PRE-UPDATE GRADIENT ALIGNMENT ===")

for (
    episode_id,
    novel_family,
    prompt,
    answer,
) in pilot_episodes:

    novel_signature = (
        hidden_gradient_signature(
            prompt,
            answer,
        )
    )

    all_cosines = []

    family_cosines = {
        "facts": [],
        "symbols": [],
        "rules": [],
        "arithmetic": [],
    }

    for protected in protected_signatures:

        cosine = signature_cosine(
            novel_signature,
            protected["signature"],
        )

        all_cosines.append(cosine)

        family_cosines[
            protected["family"]
        ].append(cosine)

    family_means = {
        family: sum(values) / len(values)
        for family, values
        in family_cosines.items()
    }

    same_family_mean = (
        family_means[novel_family]
    )

    off_family_values = []

    for family, values in family_cosines.items():

        if family != novel_family:
            off_family_values.extend(values)

    off_family_mean = (
        sum(off_family_values)
        / len(off_family_values)
    )

    print(f"\n{episode_id} | {novel_family}")

    print(
        f"Mean cosine:       "
        f"{sum(all_cosines) / len(all_cosines):.4f}"
    )

    print(
        f"Minimum cosine:    "
        f"{min(all_cosines):.4f}"
    )

    print(
        f"Maximum cosine:    "
        f"{max(all_cosines):.4f}"
    )

    print(
        f"Same-family mean:  "
        f"{same_family_mean:.4f}"
    )

    print(
        f"Off-family mean:   "
        f"{off_family_mean:.4f}"
    )

    for family in [
        "facts",
        "symbols",
        "rules",
        "arithmetic",
    ]:

        print(
            f"Cosine -> {family:<10}"
            f"{family_means[family]:.4f}"
        )