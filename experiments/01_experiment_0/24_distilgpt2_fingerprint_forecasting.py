"""Compact DistilGPT-2 fingerprint and forecasting replication.

Extracts pre-update fingerprints from the controlled-memory checkpoint, joins
them to the fixed 144-row candidate sweep, and evaluates strictly held-out
episodes with grouped cross-validation.  This is a qualitative second-model
replication; it deliberately uses the same compact protocol as scripts 22/23.
"""

import csv
import json
import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import torch
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer


PLAN_PATH = Path(__file__).with_name("22_distilgpt2_replication.py")
plan_spec = spec_from_file_location("distilgpt2_plan", PLAN_PATH)
plan = module_from_spec(plan_spec)
assert plan_spec.loader is not None
plan_spec.loader.exec_module(plan)

FINGERPRINT_FILE = plan.RESULTS_DIR / "distilgpt2_24_preupdate_fingerprints.csv"
DATASET_FILE = plan.RESULTS_DIR / "distilgpt2_24_forecasting_dataset.csv"
SUMMARY_CSV = plan.RESULTS_DIR / "distilgpt2_24_fingerprint_forecasting_summary.csv"
SUMMARY_JSON = plan.RESULTS_DIR / "distilgpt2_24_fingerprint_forecasting_summary.json"
OUTCOME_FILE = plan.RESULTS_DIR / "distilgpt2_24_candidate_sweep.csv"
N_SPLITS = 20
TEST_FRACTION = 0.25
RANDOM_STATE = 42
N_ESTIMATORS = 300
FAMILIES = tuple(plan.PROTECTED_FAMILIES)

CANDIDATE_FEATURES = ["candidate", "mode", "learning_rate"]
MAGNITUDE_PROFILE_FEATURES = [
    "novel_loss", "total_grad_norm",
    *[f"layer_{layer}_grad_norm" for layer in range(plan.N_TRANSFORMER_BLOCKS)],
    "early_grad_norm", "middle_grad_norm", "late_grad_norm",
    "early_grad_fraction", "middle_grad_fraction", "late_grad_fraction",
    "peak_layer", "layer_concentration",
]
ALIGNMENT_FEATURES = [
    "mean_protected_cosine", "min_protected_cosine", "max_protected_cosine",
    "same_family_cosine", "off_family_cosine",
    *[f"cosine_to_{family}" for family in FAMILIES],
]
FEATURE_GROUPS = {
    "candidate_only": CANDIDATE_FEATURES,
    "magnitude_profile_plus_candidate": MAGNITUDE_PROFILE_FEATURES + CANDIDATE_FEATURES,
    "alignment_plus_candidate": ALIGNMENT_FEATURES + CANDIDATE_FEATURES,
    "full_fingerprint_plus_candidate": (
        MAGNITUDE_PROFILE_FEATURES + ALIGNMENT_FEATURES + CANDIDATE_FEATURES
    ),
}
TARGETS = ("acquisition_gain", "mean_protected_damage")


def set_seed() -> None:
    torch.manual_seed(plan.SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(plan.SEED)


def block_l2(norms: list[float]) -> float:
    return math.sqrt(sum(value ** 2 for value in norms))


def gradient_fingerprint(model, tokenizer, device, prompt: str, answer: str) -> dict:
    """Measure answer-loss gradients before any candidate update is applied."""
    model.zero_grad(set_to_none=True)
    input_ids, labels = plan.make_answer_only_batch(tokenizer, prompt, answer, device)
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    layer_norms = []
    for block in model.transformer.h:
        squared = sum(
            parameter.grad.detach().float().pow(2).sum().item()
            for parameter in block.parameters() if parameter.grad is not None
        )
        layer_norms.append(math.sqrt(squared))
    total_squared = sum(
        parameter.grad.detach().float().pow(2).sum().item()
        for parameter in model.parameters() if parameter.grad is not None
    )
    model.zero_grad(set_to_none=True)

    transformer_norm = block_l2(layer_norms)
    if transformer_norm == 0:
        raise ValueError("Transformer gradient norm is zero.")
    groups = (layer_norms[0:2], layer_norms[2:4], layer_norms[4:6])
    group_norms = [block_l2(group) for group in groups]
    peak_norm = max(layer_norms)
    return {
        "novel_loss": float(loss.item()),
        "total_grad_norm": math.sqrt(total_squared),
        "layer_grad_norms": layer_norms,
        "early_grad_norm": group_norms[0], "middle_grad_norm": group_norms[1],
        "late_grad_norm": group_norms[2],
        "early_grad_fraction": group_norms[0] ** 2 / transformer_norm ** 2,
        "middle_grad_fraction": group_norms[1] ** 2 / transformer_norm ** 2,
        "late_grad_fraction": group_norms[2] ** 2 / transformer_norm ** 2,
        "peak_layer": layer_norms.index(peak_norm),
        "layer_concentration": peak_norm ** 2 / transformer_norm ** 2,
    }


def hidden_gradient_signature(model, tokenizer, device, prompt: str, answer: str) -> list[torch.Tensor]:
    """Gradient at the answer-prediction hidden state for every block."""
    model.zero_grad(set_to_none=True)
    input_ids, labels = plan.make_answer_only_batch(tokenizer, prompt, answer, device)
    outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True, return_dict=True)
    states = outputs.hidden_states[1:]
    if len(states) != plan.N_TRANSFORMER_BLOCKS:
        raise RuntimeError(f"Expected {plan.N_TRANSFORMER_BLOCKS} block states, got {len(states)}.")
    for state in states:
        state.retain_grad()
    outputs.loss.backward()
    position = input_ids.shape[1] - 2
    signature = [state.grad[0, position, :].detach().float().cpu() for state in states]
    model.zero_grad(set_to_none=True)
    return signature


def cosine(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return float(torch.nn.functional.cosine_similarity(
        torch.cat(a).unsqueeze(0), torch.cat(b).unsqueeze(0)
    ).item())


def alignment_features(novel_signature, novel_family: str, protected_signatures: list[dict]) -> dict:
    by_family = {family: [] for family in FAMILIES}
    for protected in protected_signatures:
        by_family[protected["family"]].append(cosine(novel_signature, protected["signature"]))
    means = {family: sum(values) / len(values) for family, values in by_family.items()}
    all_values = [value for values in by_family.values() for value in values]
    off_family = [value for family, values in by_family.items() if family != novel_family for value in values]
    return {
        "mean_protected_cosine": sum(all_values) / len(all_values),
        "min_protected_cosine": min(all_values), "max_protected_cosine": max(all_values),
        "same_family_cosine": means[novel_family],
        "off_family_cosine": sum(off_family) / len(off_family),
        **{f"cosine_to_{family}": means[family] for family in FAMILIES},
    }


def extract_fingerprints() -> pd.DataFrame:
    if not plan.CHECKPOINT_DIR.exists():
        raise RuntimeError(f"Missing controlled-memory checkpoint: {plan.CHECKPOINT_DIR}")
    set_seed()
    device = plan.select_device()
    tokenizer = AutoTokenizer.from_pretrained(plan.CHECKPOINT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        plan.CHECKPOINT_DIR, attn_implementation="eager"
    ).to(device)
    model.eval()
    if len(model.transformer.h) != plan.N_TRANSFORMER_BLOCKS:
        raise RuntimeError("Checkpoint architecture does not match DistilGPT-2's six blocks.")
    print(f"Extracting {len(plan.NOVEL_EPISODES)} fingerprints on {device}; protected signatures first.")
    protected_signatures = [
        {"family": family, "signature": hidden_gradient_signature(model, tokenizer, device, prompt, answer)}
        for family, examples in plan.PROTECTED_FAMILIES.items()
        for prompt, answer in examples
    ]
    expected_ids = {episode[0] for episode in plan.NOVEL_EPISODES}
    if FINGERPRINT_FILE.exists():
        existing = pd.read_csv(FINGERPRINT_FILE)
        if not set(existing["episode_id"]).issubset(expected_ids) or existing["episode_id"].duplicated().any():
            raise ValueError(f"Existing {FINGERPRINT_FILE} is not a valid resumable fingerprint file.")
        rows = existing.to_dict(orient="records")
    else:
        rows = []
    completed = {row["episode_id"] for row in rows}
    for index, (episode_id, family, prompt, answer) in enumerate(plan.NOVEL_EPISODES, start=1):
        if episode_id in completed:
            print(f"  {index:02d}/{len(plan.NOVEL_EPISODES)} {episode_id} (resumed)")
            continue
        fingerprint = gradient_fingerprint(model, tokenizer, device, prompt, answer)
        row = {
            "episode_id": episode_id, "novel_family": family,
            "novel_prompt": prompt, "novel_answer": answer,
            **{f"layer_{i}_grad_norm": value for i, value in enumerate(fingerprint.pop("layer_grad_norms"))},
            **fingerprint,
            **alignment_features(hidden_gradient_signature(model, tokenizer, device, prompt, answer), family, protected_signatures),
        }
        fractions = sum(row[key] for key in ("early_grad_fraction", "middle_grad_fraction", "late_grad_fraction"))
        if not math.isclose(fractions, 1.0, abs_tol=1e-6) or not all(
            math.isfinite(value) for key, value in row.items()
            if key not in {"episode_id", "novel_family", "novel_prompt", "novel_answer"}
        ):
            raise ValueError(f"Invalid fingerprint for {episode_id}.")
        rows.append(row)
        # CPU-only environments may time out mid-extraction; every completed
        # episode is therefore durable and the next invocation resumes it.
        pd.DataFrame(rows).to_csv(FINGERPRINT_FILE, index=False)
        print(f"  {index:02d}/{len(plan.NOVEL_EPISODES)} {episode_id}")
    frame = pd.DataFrame(rows)
    frame.to_csv(FINGERPRINT_FILE, index=False)
    return frame


def make_model(features: list[str], X: pd.DataFrame) -> Pipeline:
    categorical = [column for column in features if is_string_dtype(X[column])]
    numeric = [column for column in features if column not in categorical]
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ])),
        ("model", RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def evaluate(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_FRACTION, random_state=RANDOM_STATE)
    for group_name, features in FEATURE_GROUPS.items():
        for target in TARGETS:
            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(dataset[features], dataset[target], groups=dataset["episode_id"]), start=1
            ):
                train, test = dataset.iloc[train_idx], dataset.iloc[test_idx]
                model = make_model(features, train[features])
                model.fit(train[features], train[target])
                prediction = model.predict(test[features])
                rows.append({
                    "feature_group": group_name, "target": target, "fold": fold,
                    "test_episodes": test["episode_id"].nunique(), "test_rows": len(test),
                    "mae": mean_absolute_error(test[target], prediction),
                    "r2": r2_score(test[target], prediction),
                })
    scores = pd.DataFrame(rows)
    summary = scores.groupby(["feature_group", "target"], sort=False).agg(
        folds=("fold", "count"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
    ).reset_index()
    summary.insert(2, "feature_count", summary["feature_group"].map(lambda name: len(FEATURE_GROUPS[name])))
    return summary


def main() -> None:
    if not OUTCOME_FILE.exists():
        raise RuntimeError(f"Missing 144-row candidate sweep: {OUTCOME_FILE}")
    fingerprints = extract_fingerprints()
    outcomes = pd.read_csv(OUTCOME_FILE)
    expected = len(plan.NOVEL_EPISODES) * len(plan.CANDIDATES)
    if len(outcomes) != expected or outcomes.duplicated(["episode_id", "candidate"]).any():
        raise ValueError(f"Expected {expected} unique candidate outcomes, found {len(outcomes)}.")
    dataset = outcomes.merge(fingerprints, on="episode_id", how="left", validate="many_to_one", suffixes=("", "_fp"))
    if len(dataset) != expected or dataset["novel_loss"].isna().any():
        raise ValueError("Fingerprint/outcome merge is incomplete.")
    dataset.to_csv(DATASET_FILE, index=False)
    summary = evaluate(dataset)
    summary.to_csv(SUMMARY_CSV, index=False)
    payload = {
        "model": plan.MODEL_NAME, "checkpoint": str(plan.CHECKPOINT_DIR),
        "fingerprint_rows": len(fingerprints), "outcome_rows": len(outcomes), "merged_rows": len(dataset),
        "cv": {"scheme": "Repeated GroupShuffleSplit by episode", "splits": N_SPLITS,
               "test_fraction": TEST_FRACTION, "episodes": dataset["episode_id"].nunique()},
        "targets": list(TARGETS), "feature_groups": {name: features for name, features in FEATURE_GROUPS.items()},
        "results": summary.to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n=== DISTILGPT-2 FINGERPRINT FORECASTING ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved fingerprints: {FINGERPRINT_FILE}")
    print(f"Saved merged dataset: {DATASET_FILE}")
    print(f"Saved summaries: {SUMMARY_CSV}, {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
