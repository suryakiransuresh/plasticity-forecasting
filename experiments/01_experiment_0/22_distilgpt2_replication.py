"""Experiment 0.12: compact DistilGPT-2 replication and sanity gate.

This is the pre-sweep setup for a second causal-LM replication.  It preserves
the original controlled-memory protocol with 24 novel episodes (six per
family) and six candidate updates.  The controlled-memory checkpoint is
reused only after the expanded episode set passes ``--stage audit``.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "distilgpt2"
SEED = 42
RESULTS_DIR = Path("results/experiment_0")
AUDIT_FILE = RESULTS_DIR / "distilgpt2_24_sanity_token_audit.json"
CHECKPOINT_DIR = Path("results/checkpoints/distilgpt2_controlled_memory")
CONTROLLED_MEMORY_SUMMARY_FILE = CHECKPOINT_DIR / "controlled_memory_summary.json"
MODEL_CACHE_DIR = Path("results/cache/huggingface")

# DistilGPT-2 has six transformer blocks, so this is the faithful compact
# analogue of GPT-2's early (0--3) and late (8--11) update strategies.
N_TRANSFORMER_BLOCKS = 6
EARLY_BLOCKS = range(0, 2)
LATE_BLOCKS = range(4, 6)
CONTROLLED_MEMORY_EPOCHS = 200
CONTROLLED_MEMORY_LR = 1e-5
CONTROLLED_MEMORY_TARGET_LOSS = 0.05
UPDATE_STEPS = 10
MEAN_DAMAGE_BUDGET = 0.01

# Three protected associations per family retain all four task families while
# keeping the second-model replication tractable on the M2.
PROTECTED_FAMILIES = {
    "facts": [
        ("In Veloria, the national emblem is", " amber"),
        ("In Torvia, the national emblem is", " gold"),
        ("In Caldria, the national emblem is", " silver"),
    ],
    "symbols": [
        ("The symbol ZARP maps to", " K"),
        ("The symbol FEX maps to", " M"),
        ("The symbol VORP maps to", " T"),
    ],
    "rules": [
        ("A blue dax belongs to class", " Q"),
        ("A red mip belongs to class", " W"),
        ("A green tor belongs to class", " E"),
    ],
    "arithmetic": [
        ("Under rule F, the result for 1 and 2 is", " 4"),
        ("Under rule F, the result for 2 and 3 is", " 7"),
        ("Under rule F, the result for 3 and 2 is", " 8"),
    ],
}

# Fixed before looking at any expanded-run outcomes: 24 episodes x 6 candidates
# yields 144 updates. Novel target tokens are audited as unique and disjoint
# from protected targets.
NOVEL_EPISODES = [
    ("D2_F001", "facts", "In Ravelle, the national emblem is", " ivory"),
    ("D2_F002", "facts", "In Tavira, the national emblem is", " green"),
    ("D2_F003", "facts", "In Orinth, the national emblem is", " marble"),
    ("D2_S001", "symbols", "The symbol WEX maps to", " H"),
    ("D2_S002", "symbols", "The symbol DORP maps to", " P"),
    ("D2_S003", "symbols", "The symbol KELM maps to", " S"),
    ("D2_R001", "rules", "A white kev belongs to class", " A"),
    ("D2_R002", "rules", "A purple lod belongs to class", " D"),
    ("D2_R003", "rules", "An orange riv belongs to class", " F"),
    ("D2_A001", "arithmetic", "Under rule F, the result for 1 and 4 is", " 6"),
    ("D2_A002", "arithmetic", "Under rule F, the result for 3 and 4 is", " 10"),
    ("D2_A003", "arithmetic", "Under rule F, the result for 4 and 3 is", " 11"),
    ("D2_F004", "facts", "In Belvara, the national emblem is", " black"),
    ("D2_F005", "facts", "In Cormia, the national emblem is", " blue"),
    ("D2_F006", "facts", "In Darsen, the national emblem is", " orange"),
    ("D2_S004", "symbols", "The symbol NELP maps to", " G"),
    ("D2_S005", "symbols", "The symbol RAVK maps to", " J"),
    ("D2_S006", "symbols", "The symbol TULP maps to", " L"),
    ("D2_R004", "rules", "A black vup belongs to class", " B"),
    ("D2_R005", "rules", "A yellow zed belongs to class", " C"),
    ("D2_R006", "rules", "A brown fim belongs to class", " R"),
    ("D2_A004", "arithmetic", "Under rule F, the result for 4 and 4 is", " 12"),
    ("D2_A005", "arithmetic", "Under rule F, the result for 5 and 3 is", " 13"),
    ("D2_A006", "arithmetic", "Under rule F, the result for 3 and 5 is", " 14"),
]

CANDIDATES = [
    {"name": "full_lr_1e-6", "mode": "full", "lr": 1e-6},
    {"name": "full_lr_1e-5", "mode": "full", "lr": 1e-5},
    {"name": "early_lr_5e-6", "mode": "early", "lr": 5e-6},
    {"name": "early_lr_1e-5", "mode": "early", "lr": 1e-5},
    {"name": "late_lr_5e-6", "mode": "late", "lr": 5e-6},
    {"name": "late_lr_1e-5", "mode": "late", "lr": 1e-5},
]

PLANNED_STAGES = (
    "1. Reuse the frozen controlled-memory checkpoint on the 12 protected items.",
    "2. Sweep all 144 episode/candidate updates from that same checkpoint.",
    "3. Extract pre-update magnitude/profile and alignment fingerprints.",
    "4. Evaluate candidate-only versus fingerprint forecasting with grouped CV.",
)


def select_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def answer_token_audit(tokenizer) -> list[dict]:
    """Check every protected and novel target before a loss/sweep is defined."""
    rows = []
    for split, examples in (
        ("protected", [(family, *item) for family, items in PROTECTED_FAMILIES.items() for item in items]),
        ("novel", [(family, prompt, answer) for _, family, prompt, answer in NOVEL_EPISODES]),
    ):
        for family, prompt, answer in examples:
            token_ids = tokenizer.encode(answer, add_special_tokens=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
            rows.append({
                "split": split,
                "family": family,
                "answer": answer,
                "answer_token_ids": token_ids,
                "answer_token_count": len(token_ids),
                "prompt_token_count": len(prompt_ids),
                "full_token_count": len(full_ids),
                "boundary_matches": full_ids[:len(prompt_ids)] == prompt_ids,
            })
    return rows


def target_collision_audit(tokenizer) -> dict:
    """Prove all novel answer tokens are unique and protected-disjoint."""
    protected = {tokenizer.encode(answer, add_special_tokens=False)[0]
                 for examples in PROTECTED_FAMILIES.values() for _, answer in examples}
    novel = [tokenizer.encode(answer, add_special_tokens=False)[0]
             for _, _, _, answer in NOVEL_EPISODES]
    return {
        "novel_targets_unique": len(novel) == len(set(novel)),
        "protected_novel_target_overlap": sorted(protected.intersection(novel)),
        "episode_ids_unique": len(NOVEL_EPISODES) == len({row[0] for row in NOVEL_EPISODES}),
    }


def run_sanity_audit() -> None:
    """Load DistilGPT-2 and prove the compact protocol is token/loss compatible."""
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)

    device = select_device()
    # Keep a reproducible model cache inside the repository rather than relying
    # on a user-home cache that may be unavailable on another machine.
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    token_rows = answer_token_audit(tokenizer)
    invalid_tokens = [row for row in token_rows if row["answer_token_count"] != 1]
    invalid_boundaries = [row for row in token_rows if not row["boundary_matches"]]
    collisions = target_collision_audit(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager", cache_dir=MODEL_CACHE_DIR
    ).to(device)
    model.eval()
    blocks = model.transformer.h
    if len(blocks) != N_TRANSFORMER_BLOCKS:
        raise RuntimeError(
            f"Expected {N_TRANSFORMER_BLOCKS} DistilGPT-2 blocks, found {len(blocks)}."
        )

    # A single answer-only forward/backward pass validates masking and that
    # all layers are reachable for later profile/alignment extraction.
    _, _, prompt, answer = NOVEL_EPISODES[0]
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    full_ids = tokenizer(prompt + answer, return_tensors="pt")["input_ids"].to(device)
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=full_ids, labels=labels).loss
    loss.backward()
    layers_with_gradient = [
        index
        for index, layer in enumerate(blocks)
        if any(parameter.grad is not None for parameter in layer.parameters())
    ]
    model.zero_grad(set_to_none=True)

    report = {
        "model": MODEL_NAME,
        "device": str(device),
        "transformer_blocks": len(blocks),
        "protected_items": sum(map(len, PROTECTED_FAMILIES.values())),
        "novel_episodes": len(NOVEL_EPISODES),
        "candidates": len(CANDIDATES),
        "planned_updates": len(NOVEL_EPISODES) * len(CANDIDATES),
        "single_token_answers": len(token_rows) - len(invalid_tokens),
        "total_answers": len(token_rows),
        "invalid_token_answers": invalid_tokens,
        "invalid_prompt_boundaries": invalid_boundaries,
        "target_collision_audit": collisions,
        "smoke_loss": float(loss.item()),
        "finite_smoke_loss": math.isfinite(loss.item()),
        "layers_with_gradient": layers_with_gradient,
        "audit_passed": (
            not invalid_tokens
            and not invalid_boundaries
            and collisions["novel_targets_unique"]
            and not collisions["protected_novel_target_overlap"]
            and collisions["episode_ids_unique"]
            and math.isfinite(loss.item())
            and layers_with_gradient == list(range(N_TRANSFORMER_BLOCKS))
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("=== DISTILGPT-2 SANITY + TOKEN AUDIT ===")
    print(f"Device: {device}; transformer blocks: {len(blocks)}")
    print(
        f"Single-token answers: {report['single_token_answers']}/{report['total_answers']} "
        f"| prompt boundaries valid: {not invalid_boundaries}"
    )
    print(
        f"Answer-only smoke loss: {report['smoke_loss']:.4f} "
        f"| layers with gradients: {layers_with_gradient}"
    )
    print(f"Planned 24-episode sweep: {report['planned_updates']} updates")
    print(f"Audit passed: {report['audit_passed']}")
    print(f"Saved audit: {AUDIT_FILE}")
    if not report["audit_passed"]:
        raise RuntimeError("DistilGPT-2 audit failed; do not start the replication sweep.")


def make_answer_only_batch(tokenizer, prompt: str, answer: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one association and mask every token except its answer."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(prompt + answer, return_tensors="pt")["input_ids"]
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    return full_ids.to(device), labels.to(device)


def answer_loss(model, tokenizer, prompt: str, answer: str, device: torch.device) -> float:
    """Return the answer-token cross-entropy for a single association."""
    input_ids, labels = make_answer_only_batch(tokenizer, prompt, answer, device)
    with torch.no_grad():
        return float(model(input_ids=input_ids, labels=labels).loss.item())


def protected_examples() -> list[tuple[str, str, str]]:
    return [
        (family, prompt, answer)
        for family, items in PROTECTED_FAMILIES.items()
        for prompt, answer in items
    ]


def evaluate_protected_losses(model, tokenizer, device: torch.device) -> tuple[list[dict], dict[str, float]]:
    """Evaluate every protected item and produce per-family mean losses."""
    model.eval()
    item_losses = []
    for family, prompt, answer in protected_examples():
        item_losses.append({
            "family": family,
            "prompt": prompt,
            "answer": answer,
            "loss": answer_loss(model, tokenizer, prompt, answer, device),
        })
    family_losses = {
        family: sum(row["loss"] for row in item_losses if row["family"] == family)
        / len(PROTECTED_FAMILIES[family])
        for family in PROTECTED_FAMILIES
    }
    return item_losses, family_losses


def run_controlled_memory(target_loss: float, max_epochs: int) -> None:
    """Memorize all protected associations with answer-only loss."""
    if not AUDIT_FILE.exists():
        raise RuntimeError(f"Missing {AUDIT_FILE}; run --stage audit first.")
    audit = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    if not audit.get("audit_passed"):
        raise RuntimeError("DistilGPT-2 audit did not pass; refusing to train.")

    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager", cache_dir=MODEL_CACHE_DIR
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONTROLLED_MEMORY_LR, weight_decay=0.0)

    final_items = []
    final_families = {}
    epochs_completed = 0
    for epoch in range(max_epochs):
        examples = protected_examples()
        ordering = torch.randperm(len(examples), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
        model.train()
        train_losses = []
        for index in ordering:
            _, prompt, answer = examples[index]
            input_ids, labels = make_answer_only_batch(tokenizer, prompt, answer, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        final_items, final_families = evaluate_protected_losses(model, tokenizer, device)
        worst_loss = max(row["loss"] for row in final_items)
        epochs_completed = epoch + 1
        print(
            f"Epoch {epochs_completed:03d} | train={sum(train_losses) / len(train_losses):.4f} "
            f"| eval_avg={sum(row['loss'] for row in final_items) / len(final_items):.4f} "
            f"| eval_max={worst_loss:.4f}"
        )
        if worst_loss < target_loss:
            break

    worst_item = max(final_items, key=lambda row: row["loss"])
    target_met = worst_item["loss"] < target_loss
    summary = {
        "model": MODEL_NAME,
        "seed": SEED,
        "device": str(device),
        "loss_type": "answer_only_cross_entropy",
        "learning_rate": CONTROLLED_MEMORY_LR,
        "target_loss": target_loss,
        "max_epochs": max_epochs,
        "epochs_completed": epochs_completed,
        "target_met": target_met,
        "final_item_losses": final_items,
        "final_per_family_mean_losses": final_families,
        "final_mean_loss": sum(row["loss"] for row in final_items) / len(final_items),
        "worst_item": worst_item,
        "worst_item_loss": worst_item["loss"],
    }
    if not target_met:
        raise RuntimeError(
            f"Target loss {target_loss:.4f} was not reached after {max_epochs} epochs; "
            f"worst item is {worst_item['loss']:.4f}."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    CONTROLLED_MEMORY_SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("=== DISTILGPT-2 CONTROLLED MEMORY COMPLETE ===")
    print(f"Epochs: {epochs_completed}; target: {target_loss:.4f}; worst loss: {worst_item['loss']:.4f}")
    print(f"Per-family mean losses: {json.dumps(final_families, sort_keys=True)}")
    print(f"Saved checkpoint and summary: {CHECKPOINT_DIR}")


def print_plan() -> None:
    print("=== DISTILGPT-2 COMPACT REPLICATION PLAN ===")
    print(f"Model: {MODEL_NAME}; protected items: 12; novel episodes: {len(NOVEL_EPISODES)}")
    print(f"Candidates: {', '.join(candidate['name'] for candidate in CANDIDATES)}")
    for stage in PLANNED_STAGES:
        print(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "controlled-memory", "plan"), default="audit")
    parser.add_argument("--target-loss", type=float, default=CONTROLLED_MEMORY_TARGET_LOSS)
    parser.add_argument("--max-epochs", type=int, default=CONTROLLED_MEMORY_EPOCHS)
    args = parser.parse_args()
    if args.stage == "audit":
        run_sanity_audit()
    elif args.stage == "controlled-memory":
        run_controlled_memory(args.target_loss, args.max_epochs)
    else:
        print_plan()
