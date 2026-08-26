#!/usr/bin/env python3
"""Non-editing preflight for Qwen2.5-0.5B on five RippleEdits facts.

Run from the repository root:
  python experiments/02_external_validation/00_rippleedits_qwen_sanity.py --download

The optional download clones the official benchmark only.  This script never
optimizes, edits, or saves model weights.
"""

import argparse
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OFFICIAL_REPO = "https://github.com/edenbiran/RippleEdits.git"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "external_validation" / "RippleEdits"
DEFAULT_OUTPUT = ROOT / "results" / "external_validation" / "rippleedits_qwen_sanity.json"
DEFAULT_CACHE_DIR = ROOT / "results" / "cache" / "huggingface"


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def find_benchmark(data_dir, subset):
    """Find a case-insensitive official subset JSON under a cloned checkout."""
    direct = [data_dir / "data" / "benchmark", data_dir / "benchmark", data_dir]
    names = {f"{subset.lower()}.json", f"rippleedits_{subset.lower()}.json"}
    for base in direct:
        if base.is_dir():
            for path in base.glob("*.json"):
                if path.name.lower() in names or path.stem.lower() == subset.lower():
                    return path
    raise FileNotFoundError(
        f"Could not find {subset}.json below {data_dir}. Expected the official "
        "repository layout data/benchmark/{RECENT,RANDOM,POPULAR}.json."
    )


def download_if_requested(data_dir, should_download):
    if data_dir.exists() or not should_download:
        return
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official RippleEdits benchmark to {data_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", OFFICIAL_REPO, str(data_dir)], check=True
    )


def edit_pair(record):
    """Convert the benchmark's declarative edit to a prompt and target suffix.

    Most official prompts follow "... is TARGET."; a minority use relation
    wording such as "... follows TARGET.". The separator space is deliberately
    assigned to the answer string: Qwen can merge a trailing prompt-space with
    the first answer token, which would make answer-only labels ambiguous.
    """
    text = record["edit"]["prompt"].strip()
    relation = next((token for token in (" is ", " follows ") if token in text), None)
    if relation is None:
        raise ValueError(f"Cannot split edit prompt into prompt/answer: {text!r}")
    prefix, answer = text.rsplit(relation, 1)
    answer = answer.rstrip(".?!").strip()
    if not answer:
        raise ValueError(f"Empty answer after splitting edit prompt: {text!r}")
    return prefix + relation.rstrip(), " " + answer


def answer_only_loss(model, tokenizer, prompt, answer, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + answer, add_special_tokens=False, return_tensors="pt").input_ids
    prefix_ok = torch.equal(full_ids[:, : prompt_ids.shape[1]], prompt_ids)
    if not prefix_ok:
        raise ValueError("Tokenizer changed prompt tokens at the answer boundary.")
    input_ids = full_ids.to(device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=labels).loss
    return float(loss.cpu()), int(prompt_ids.shape[1]), int(input_ids.shape[1]), prefix_ok


def gradient_check(model, tokenizer, prompt, answer, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = tokenizer(prompt + answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    norms = [p.grad.detach().float().norm().item() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach().cpu()),
        "finite": bool(torch.isfinite(loss).item() and all(math.isfinite(x) for x in norms)),
        "nonzero_gradient_tensors": sum(x > 0.0 for x in norms),
        "gradient_l2": float(math.sqrt(sum(x * x for x in norms))),
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }


def memory_stats(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    stats = {"process_peak_rss_mib": round(rss_bytes / 2**20, 1)}
    if device.type == "mps":
        stats.update({
            "mps_current_allocated_mib": round(torch.mps.current_allocated_memory() / 2**20, 1),
            "mps_driver_allocated_mib": round(torch.mps.driver_allocated_memory() / 2**20, 1),
        })
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--subset", choices=("RECENT", "RANDOM", "POPULAR"), default="RANDOM")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--download", action="store_true", help="Clone the official public benchmark if absent.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")
    device = choose_device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")
    download_if_requested(args.data_dir, args.download)
    benchmark_path = find_benchmark(args.data_dir, args.subset)
    records = json.loads(benchmark_path.read_text())[: args.limit]
    if not records:
        raise RuntimeError(f"No records in {benchmark_path}")

    print(f"RippleEdits: {benchmark_path} ({len(records)} records; top-level keys: {list(records[0])})")
    print("First edit keys:", list(records[0].get("edit", {})))
    print(f"Loading {args.model} on {device} ...")
    load_start = time.perf_counter()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(args.cache_dir))
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=str(args.cache_dir)).to(device).eval()
    model_load_seconds = round(time.perf_counter() - load_start, 3)
    layers = getattr(model.config, "num_hidden_layers", None)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.config._name_or_path}; layers: {layers}; parameters: {parameters:,}; device: {device}")

    examples = []
    for index, record in enumerate(records):
        prompt, answer = edit_pair(record)
        loss, prompt_tokens, total_tokens, prefix_ok = answer_only_loss(model, tokenizer, prompt, answer, device)
        examples.append({
            "index": index,
            "edit_prompt": record["edit"]["prompt"],
            "prompt": prompt,
            "expected_answer": answer,
            "prompt_tokens": prompt_tokens,
            "answer_tokens": total_tokens - prompt_tokens,
            "prompt_is_token_prefix": prefix_ok,
            "baseline_answer_only_loss": loss,
        })
        print(f"{index}: answer_tokens={total_tokens - prompt_tokens}, loss={loss:.4f}, boundary_ok={prefix_ok}")

    gradient = gradient_check(model, tokenizer, examples[0]["prompt"], examples[0]["expected_answer"], device)
    print("Gradient check:", gradient)
    summary = {
        "purpose": "sanity-only; no optimizer step or model edit was performed",
        "benchmark": {"path": str(benchmark_path), "subset": args.subset, "records": len(records),
                      "record_keys": list(records[0]), "edit_keys": list(records[0].get("edit", {}))},
        "model": {"name": args.model, "layers": layers, "parameters": parameters,
                  "dtype": str(next(model.parameters()).dtype), "device": str(device),
                  "load_seconds": model_load_seconds},
        "examples": examples,
        "gradient_check": gradient,
        "runtime_memory": memory_stats(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SANITY FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
