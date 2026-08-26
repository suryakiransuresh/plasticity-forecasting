"""144-update DistilGPT-2 candidate replication sweep; resumable and incremental."""

import argparse
import copy
import csv
import gc
import json
import math
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PLAN_PATH = Path(__file__).with_name("22_distilgpt2_replication.py")
plan_spec = spec_from_file_location("distilgpt2_plan", PLAN_PATH)
plan = module_from_spec(plan_spec)
assert plan_spec.loader is not None
plan_spec.loader.exec_module(plan)

BASE_SEED = plan.SEED
CHECKPOINT_DIR = plan.CHECKPOINT_DIR
RESULTS_DIR = plan.RESULTS_DIR
OUTPUT_FILE = RESULTS_DIR / "distilgpt2_24_candidate_sweep.csv"
SUMMARY_FILE = RESULTS_DIR / "distilgpt2_24_candidate_sweep_summary.json"
AGGREGATE_FILE = RESULTS_DIR / "distilgpt2_24_candidate_sweep_aggregate.csv"
FRONTIER_FILE = RESULTS_DIR / "distilgpt2_24_candidate_sweep_frontier.csv"


def reset_seed(seed):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def configure_trainable_parameters(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    if mode == "full":
        return
    if mode == "early":
        blocks = plan.EARLY_BLOCKS
    elif mode == "late":
        blocks = plan.LATE_BLOCKS
    else:
        raise ValueError(f"Unknown candidate mode: {mode}")
    for index in blocks:
        for parameter in model.transformer.h[index].parameters():
            parameter.requires_grad = True


def trainable_snapshot(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def update_norm(model, before):
    return math.sqrt(sum((parameter.detach() - before[name]).float().pow(2).sum().item()
                         for name, parameter in model.named_parameters() if name in before))


def protected_loss_map(model, tokenizer, device):
    model.eval()
    return {family: [plan.answer_loss(model, tokenizer, prompt, answer, device)
                     for prompt, answer in examples]
            for family, examples in plan.PROTECTED_FAMILIES.items()}


def run_candidate_update(base_model, tokenizer, device, episode, candidate, episode_seed, baseline):
    episode_id, novel_family, prompt, answer = episode
    # base_model is loaded directly from CHECKPOINT_DIR; each deepcopy has its exact state.
    model = copy.deepcopy(base_model)
    reset_seed(episode_seed)
    configure_trainable_parameters(model, candidate["mode"])
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    before_parameters = trainable_snapshot(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=candidate["lr"], weight_decay=0.0)
    novel_loss_before = plan.answer_loss(model, tokenizer, prompt, answer, device)
    model.train()
    for _ in range(plan.UPDATE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        input_ids, labels = plan.make_answer_only_batch(tokenizer, prompt, answer, device)
        model(input_ids=input_ids, labels=labels).loss.backward()
        optimizer.step()
    novel_loss_after = plan.answer_loss(model, tokenizer, prompt, answer, device)
    protected_after = protected_loss_map(model, tokenizer, device)
    family_damage = {family: [after - before for before, after in zip(baseline[family], protected_after[family])]
                     for family in plan.PROTECTED_FAMILIES}
    all_damage = [damage for values in family_damage.values() for damage in values]
    result = {
        "episode_id": episode_id, "novel_family": novel_family, "novel_prompt": prompt,
        "novel_answer": answer, "episode_seed": episode_seed, "checkpoint": str(CHECKPOINT_DIR),
        "candidate": candidate["name"], "mode": candidate["mode"], "learning_rate": candidate["lr"],
        "update_steps": plan.UPDATE_STEPS, "trainable_parameters": trainable_count,
        "update_norm": update_norm(model, before_parameters), "novel_loss_before": novel_loss_before,
        "novel_loss_after": novel_loss_after, "acquisition_gain": novel_loss_before - novel_loss_after,
        "mean_protected_damage": sum(all_damage) / len(all_damage),
    }
    for family, damages in family_damage.items():
        result[f"damage_to_{family}"] = sum(damages) / len(damages)
        for index, damage in enumerate(damages, start=1):
            result[f"damage_{family}_{index}"] = damage
    del optimizer, before_parameters, model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def read_completed():
    if not OUTPUT_FILE.exists():
        return [], set()
    with OUTPUT_FILE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row["episode_id"], row["candidate"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Existing {OUTPUT_FILE} contains duplicate episode/candidate rows.")
    return rows, set(keys)


def value(row, key):
    return float(row[key])


def pareto_frontier(rows):
    """Nondominated rows: maximize gain while minimizing protected damage."""
    frontier = []
    for row in rows:
        gain, damage = value(row, "acquisition_gain"), value(row, "mean_protected_damage")
        dominated = any(value(other, "acquisition_gain") >= gain and
                        value(other, "mean_protected_damage") <= damage and
                        (value(other, "acquisition_gain") > gain or
                         value(other, "mean_protected_damage") < damage)
                        for other in rows)
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (-value(row, "acquisition_gain"), value(row, "mean_protected_damage")))


def write_summaries(rows, expected):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["candidate"]].append(row)
    metrics = ["acquisition_gain", "mean_protected_damage", "update_norm"] + \
              [f"damage_to_{family}" for family in plan.PROTECTED_FAMILIES]
    aggregate = []
    for candidate in plan.CANDIDATES:
        values = grouped[candidate["name"]]
        if values:
            item = {"candidate": candidate["name"], "mode": candidate["mode"],
                    "learning_rate": candidate["lr"], "outcomes": len(values)}
            for metric in metrics:
                item[f"mean_{metric}"] = sum(value(row, metric) for row in values) / len(values)
            aggregate.append(item)
    aggregate_frontier = pareto_frontier([
        {"candidate": item["candidate"], "acquisition_gain": item["mean_acquisition_gain"],
         "mean_protected_damage": item["mean_mean_protected_damage"]} for item in aggregate])
    frontier_names = {item["candidate"] for item in aggregate_frontier}
    for item in aggregate:
        item["frontier_eligible"] = item["candidate"] in frontier_names
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if aggregate:
        with AGGREGATE_FILE.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
            writer.writeheader(); writer.writerows(aggregate)
    row_frontier = pareto_frontier(rows)
    if row_frontier:
        with FRONTIER_FILE.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row_frontier[0]))
            writer.writeheader(); writer.writerows(row_frontier)
    summary = {
        "model": plan.MODEL_NAME, "checkpoint": str(CHECKPOINT_DIR),
        "seed_scheme": "episode_seed = 42 + episode_index * 100; reset before each candidate",
        "episodes": len(plan.NOVEL_EPISODES), "candidates": len(plan.CANDIDATES),
        "expected_outcomes": expected, "completed_outcomes": len(rows), "complete": len(rows) == expected,
        "aggregate_by_candidate": aggregate, "aggregate_pareto_frontier": sorted(frontier_names),
        "outcome_pareto_frontier_count": len(row_frontier),
        "outcome_pareto_frontier": [{key: row[key] for key in
            ("episode_id", "novel_family", "candidate", "acquisition_gain", "mean_protected_damage")}
            for row in row_frontier],
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(resume):
    if not CHECKPOINT_DIR.exists():
        raise RuntimeError(f"Missing controlled-memory checkpoint: {CHECKPOINT_DIR}")
    controlled = json.loads(plan.CONTROLLED_MEMORY_SUMMARY_FILE.read_text(encoding="utf-8"))
    if not controlled.get("target_met"):
        raise RuntimeError("Controlled-memory target was not met; refusing to sweep.")
    expected = len(plan.NOVEL_EPISODES) * len(plan.CANDIDATES)
    existing_rows, completed = read_completed() if resume else ([], set())
    if not resume and OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    device = plan.select_device()
    reset_seed(BASE_SEED)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, attn_implementation="eager").to(device)
    base_model.eval()
    baseline = protected_loss_map(base_model, tokenizer, device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; completed: {len(existing_rows)}/{expected}; output: {OUTPUT_FILE}")
    with OUTPUT_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = None
        for episode_index, episode in enumerate(plan.NOVEL_EPISODES):
            episode_seed = BASE_SEED + episode_index * 100
            for candidate in plan.CANDIDATES:
                key = (episode[0], candidate["name"])
                if key in completed:
                    continue
                row = run_candidate_update(base_model, tokenizer, device, episode, candidate, episode_seed, baseline)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if handle.tell() == 0:
                        writer.writeheader()
                writer.writerow(row); handle.flush()
                existing_rows.append({key: str(item) for key, item in row.items()})
                completed.add(key)
                write_summaries(existing_rows, expected)
                print(f"{len(existing_rows):02d}/{expected} {episode[0]} {candidate['name']} "
                      f"gain={row['acquisition_gain']:.4f} damage={row['mean_protected_damage']:.5f}", flush=True)
    write_summaries(existing_rows, expected)
    print(f"Complete: {len(existing_rows)}/{expected}; summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    main(resume=not args.no_resume)
