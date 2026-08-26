import copy
import csv
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# REPRODUCIBILITY
# ============================================================

BASE_SEED = 42

torch.manual_seed(BASE_SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(BASE_SEED)


# ============================================================
# DEVICE + CHECKPOINT
# ============================================================

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

base_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    attn_implementation="eager"
).to(device)

tokenizer.pad_token = tokenizer.eos_token
base_model.eval()


# ============================================================
# PROTECTED MEMORIES
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


# Flatten protected memories while preserving family labels

all_protected = []

for family_name, examples in protected_families.items():

    for prompt, answer in examples:

        all_protected.append(
            (family_name, prompt, answer)
        )


# ============================================================
# NOVEL EPISODES
# ============================================================

novel_episodes = [

    # ------------------------
    # FACTUAL ASSOCIATIONS
    # ------------------------

    (
        "F001",
        "facts",
        "In Ravelle, the national emblem is",
        " ivory",
    ),
    (
        "F002",
        "facts",
        "In Tavira, the national emblem is",
        " green",
    ),
    (
        "F003",
        "facts",
        "In Orinth, the national emblem is",
        " marble",
    ),
    (
        "F004",
        "facts",
        "In Selvaris, the national emblem is",
        " crystal",
    ),
    (
        "F005",
        "facts",
        "In Luneth, the national emblem is",
        " velvet",
    ),

    # ------------------------
    # SYMBOLIC MAPPINGS
    # ------------------------

    (
        "S001",
        "symbols",
        "The symbol WEX maps to",
        " H",
    ),
    (
        "S002",
        "symbols",
        "The symbol DORP maps to",
        " P",
    ),
    (
        "S003",
        "symbols",
        "The symbol KELM maps to",
        " S",
    ),
    (
        "S004",
        "symbols",
        "The symbol TAV maps to",
        " G",
    ),
    (
        "S005",
        "symbols",
        "The symbol RUX maps to",
        " L",
    ),

    # ------------------------
    # RULE / CLASSIFICATION
    # ------------------------

    (
        "R001",
        "rules",
        "A white kev belongs to class",
        " A",
    ),
    (
        "R002",
        "rules",
        "A purple lod belongs to class",
        " D",
    ),
    (
        "R003",
        "rules",
        "An orange riv belongs to class",
        " F",
    ),
    (
        "R004",
        "rules",
        "A brown sep belongs to class",
        " J",
    ),
    (
        "R005",
        "rules",
        "A gray wom belongs to class",
        " C",
    ),

    # ------------------------
    # SYNTHETIC ARITHMETIC
    # ------------------------

    (
        "A001",
        "arithmetic",
        "Under rule F, the result for 1 and 4 is",
        " 6",
    ),
    (
        "A002",
        "arithmetic",
        "Under rule F, the result for 3 and 4 is",
        " 10",
    ),
    (
        "A003",
        "arithmetic",
        "Under rule F, the result for 4 and 3 is",
        " 11",
    ),
    (
        "A004",
        "arithmetic",
        "Under rule F, the result for 5 and 3 is",
        " 13",
    ),
    (
        "A005",
        "arithmetic",
        "Under rule F, the result for 6 and 2 is",
        " 14",
    ),
]


# ============================================================
# CANDIDATE UPDATE STRATEGIES
# ============================================================

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


# ============================================================
# SANITY CHECK
# ============================================================

print("\n=== EXPERIMENT 0.6 SETUP ===")

print(
    f"Protected memories: {len(all_protected)}"
)

print(
    f"Novel episodes:     {len(novel_episodes)}"
)

print(
    f"Candidates:         {len(candidates)}"
)

print(
    f"Expected outcomes:  "
    f"{len(novel_episodes) * len(candidates)}"
)

for family_name in protected_families:

    protected_count = len(
        protected_families[family_name]
    )

    novel_count = sum(
        1
        for episode in novel_episodes
        if episode[1] == family_name
    )

    print(
        f"{family_name:<12}"
        f" protected={protected_count}"
        f" novel={novel_count}"
    )

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_seed(seed):
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


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
    # Loss is measured only on the expected answer.
    labels[:, :prompt_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            labels=labels
        )

    return outputs.loss.item()


def training_loss(model, prompt, answer):
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

    outputs = model(
        input_ids=full_ids,
        labels=labels
    )

    return outputs.loss


def measure_protected_losses(model):
    """
    Returns:
        {
            "facts": [...],
            "symbols": [...],
            "rules": [...],
            "arithmetic": [...]
        }
    """

    model.eval()

    results = {}

    for family_name, examples in protected_families.items():

        losses = []

        for prompt, answer in examples:

            loss = answer_loss(
                model,
                prompt,
                answer
            )

            losses.append(loss)

        results[family_name] = losses

    return results


def flatten_family_losses(loss_dict):
    values = []

    for family_name in protected_families:
        values.extend(
            loss_dict[family_name]
        )

    return values

# ============================================================
# PARAMETER SELECTION
# ============================================================

def configure_trainable_parameters(model, mode):

    # First freeze everything.
    for parameter in model.parameters():
        parameter.requires_grad = False

    if mode == "full":

        for parameter in model.parameters():
            parameter.requires_grad = True

    elif mode == "early":

        for layer_index in range(0, 4):

            for parameter in (
                model.transformer.h[
                    layer_index
                ].parameters()
            ):
                parameter.requires_grad = True

    elif mode == "middle":

        for layer_index in range(4, 8):

            for parameter in (
                model.transformer.h[
                    layer_index
                ].parameters()
            ):
                parameter.requires_grad = True

    elif mode == "late":

        for layer_index in range(8, 12):

            for parameter in (
                model.transformer.h[
                    layer_index
                ].parameters()
            ):
                parameter.requires_grad = True

    else:
        raise ValueError(
            f"Unknown candidate mode: {mode}"
        )


def count_trainable_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def capture_trainable_parameters(model):

    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def compute_update_norm(
    model,
    parameter_before
):

    squared_norm = 0.0

    for name, parameter in model.named_parameters():

        if name not in parameter_before:
            continue

        difference = (
            parameter.detach()
            - parameter_before[name]
        )

        squared_norm += (
            difference.float()
            .pow(2)
            .sum()
            .item()
        )

    return squared_norm ** 0.5

# ============================================================
# HELPER SANITY TEST
# ============================================================

print("\n=== HELPER FUNCTION CHECK ===")

baseline_protected = measure_protected_losses(
    base_model
)

for family_name, losses in baseline_protected.items():

    print(
        f"{family_name:<12}"
        f" avg_loss="
        f"{sum(losses) / len(losses):.4f}"
        f" max_loss={max(losses):.4f}"
    )

print(
    "Overall protected average:",
    f"{sum(flatten_family_losses(baseline_protected)) / 20:.4f}"
)

# ============================================================
# SINGLE CANDIDATE UPDATE
# ============================================================

def run_candidate_update(
    episode,
    candidate,
    episode_seed,
    baseline_protected,
):

    (
        episode_id,
        novel_family,
        novel_prompt,
        novel_answer,
    ) = episode

    # --------------------------------------------------------
    # Start every candidate from exactly the same checkpoint
    # --------------------------------------------------------

    model = copy.deepcopy(base_model)

    reset_seed(episode_seed)

    configure_trainable_parameters(
        model,
        candidate["mode"]
    )

    trainable_parameters = (
        count_trainable_parameters(model)
    )

    parameter_before = (
        capture_trainable_parameters(model)
    )

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=candidate["lr"],
        weight_decay=0.0,
    )

    # --------------------------------------------------------
    # BEFORE UPDATE
    # --------------------------------------------------------

    model.eval()

    novel_loss_before = answer_loss(
        model,
        novel_prompt,
        novel_answer,
    )

    # --------------------------------------------------------
    # PERFORM CANDIDATE UPDATE
    # --------------------------------------------------------

    model.train()

    for step in range(STEPS):

        optimizer.zero_grad()

        loss = training_loss(
            model,
            novel_prompt,
            novel_answer,
        )

        loss.backward()
        optimizer.step()

    # --------------------------------------------------------
    # AFTER UPDATE
    # --------------------------------------------------------

    model.eval()

    novel_loss_after = answer_loss(
        model,
        novel_prompt,
        novel_answer,
    )

    protected_after = (
        measure_protected_losses(model)
    )

    acquisition_gain = (
        novel_loss_before
        - novel_loss_after
    )

    # --------------------------------------------------------
    # DAMAGE BY PROTECTED FAMILY
    # --------------------------------------------------------

    family_damage = {}

    individual_damage = {}

    for protected_family in protected_families:

        before_losses = (
            baseline_protected[
                protected_family
            ]
        )

        after_losses = (
            protected_after[
                protected_family
            ]
        )

        damages = [
            after - before
            for before, after
            in zip(
                before_losses,
                after_losses,
            )
        ]

        family_damage[
            protected_family
        ] = (
            sum(damages)
            / len(damages)
        )

        individual_damage[
            protected_family
        ] = damages

    all_damage_values = []

    for protected_family in protected_families:
        all_damage_values.extend(
            individual_damage[
                protected_family
            ]
        )

    mean_damage = (
        sum(all_damage_values)
        / len(all_damage_values)
    )

    update_norm = compute_update_norm(
        model,
        parameter_before,
    )

    result = {
        "episode_id": episode_id,
        "novel_family": novel_family,
        "novel_prompt": novel_prompt,
        "novel_answer": novel_answer,

        "candidate": candidate["name"],
        "mode": candidate["mode"],
        "learning_rate": candidate["lr"],

        "trainable_parameters":
            trainable_parameters,

        "update_norm":
            update_norm,

        "novel_loss_before":
            novel_loss_before,

        "novel_loss_after":
            novel_loss_after,

        "acquisition_gain":
            acquisition_gain,

        "mean_damage":
            mean_damage,

        "damage_to_facts":
            family_damage["facts"],

        "damage_to_symbols":
            family_damage["symbols"],

        "damage_to_rules":
            family_damage["rules"],

        "damage_to_arithmetic":
            family_damage["arithmetic"],
    }

    # Keep all 20 individual damage values as well.
    for protected_family in protected_families:

        for index, damage in enumerate(
            individual_damage[
                protected_family
            ],
            start=1,
        ):

            key = (
                f"damage_"
                f"{protected_family}_"
                f"{index}"
            )

            result[key] = damage

    return result

# ============================================================
# FULL MULTI-FAMILY CANDIDATE SWEEP
# ============================================================

OUTPUT_DIR = "results/experiment_0"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "multifamily_candidates_seedcontrolled.csv",
)


print("\n" + "=" * 80)
print("EXPERIMENT 0.6 — MULTI-FAMILY CANDIDATE SWEEP")
print("=" * 80)

print(f"Episodes:   {len(novel_episodes)}")
print(f"Candidates: {len(candidates)}")
print(
    f"Total runs: "
    f"{len(novel_episodes) * len(candidates)}"
)

print(f"Output:     {OUTPUT_FILE}")


# ------------------------------------------------------------
# Run all episodes
# ------------------------------------------------------------

all_results = []

csv_file = None
writer = None


try:

    for episode_index, episode in enumerate(
        novel_episodes
    ):

        (
            episode_id,
            novel_family,
            novel_prompt,
            novel_answer,
        ) = episode

        # IMPORTANT:
        # Every candidate for this episode receives
        # exactly the same random seed.
        episode_seed = (
            BASE_SEED
            + episode_index * 100
        )

        print("\n" + "-" * 80)

        print(
            f"Episode "
            f"{episode_index + 1:02d}/"
            f"{len(novel_episodes)}"
            f" | {episode_id}"
            f" | family={novel_family}"
        )

        print(
            f"Novel item: "
            f"{novel_prompt}{novel_answer}"
        )

        print(
            f"Episode seed: "
            f"{episode_seed}"
        )

        # ----------------------------------------------------
        # Run all six candidate updates
        # ----------------------------------------------------

        for candidate_index, candidate in enumerate(
            candidates
        ):

            result = run_candidate_update(
                episode=episode,
                candidate=candidate,
                episode_seed=episode_seed,
                baseline_protected=baseline_protected,
            )

            all_results.append(result)

            # --------------------------------------------
            # Initialize CSV after first result
            # --------------------------------------------

            if writer is None:

                csv_file = open(
                    OUTPUT_FILE,
                    "w",
                    newline="",
                    encoding="utf-8",
                )

                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=list(
                        result.keys()
                    ),
                )

                writer.writeheader()

            # --------------------------------------------
            # Save every result immediately
            # --------------------------------------------

            writer.writerow(result)

            csv_file.flush()

            print(
                f"  "
                f"{candidate_index + 1}/"
                f"{len(candidates)} "
                f"{candidate['name']:<16}"
                f" | gain="
                f"{result['acquisition_gain']:>8.4f}"
                f" | damage="
                f"{result['mean_damage']:>9.5f}"
                f" | norm="
                f"{result['update_norm']:>7.4f}"
            )


finally:

    if csv_file is not None:
        csv_file.close()


# ============================================================
# FINAL SWEEP SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SWEEP COMPLETE")
print("=" * 80)

print(
    f"Completed outcomes: "
    f"{len(all_results)}"
)

print(
    f"Expected outcomes:  "
    f"{len(novel_episodes) * len(candidates)}"
)

print(
    f"Saved to:\n"
    f"{OUTPUT_FILE}"
)