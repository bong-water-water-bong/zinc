#!/usr/bin/env python3
"""Compare prompt encoding on the designated model host; no GPU initialization."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids(binary, model, corpus, reference=False):
    command = [str(binary), str(model), str(corpus)]
    if reference:
        command = [
            str(binary),
            "-m",
            str(model),
            "-f",
            str(corpus),
            "--ids",
            "--no-escape",
            "--log-disable",
        ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    # zinc-tokenize prints its JSON array on stderr, after optional log lines.
    output = result.stdout if reference else result.stderr
    ids = json.loads(output.strip().splitlines()[-1])
    if not isinstance(ids, list) or any(type(token) is not int for token in ids):
        raise ValueError("invalid tokenizer output")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--models", required=True, help="Comma-separated GGUF paths")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--reference-tokenizer",
        type=Path,
        help=(
            "llama-tokenize executable; require candidate reference parity "
            "and record baseline differences"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100_000 or args.rounds < 1:
        parser.error("runs must be between 1 and 100000; rounds must be positive")
    corpora = sorted(args.corpus_root.resolve().glob("*.txt"))
    if not corpora:
        parser.error("corpus-root must contain .txt files")
    models = [args.model_root.resolve() / name for name in args.models.split(",")]
    builds = {
        "baseline": args.baseline.resolve(),
        "candidate": args.candidate.resolve(),
    }
    variants = {label: root / "zig-out/bin" for label, root in builds.items()}
    record = {
        "complete": False,
        "methodology": {
            "rounds": args.rounds,
            "timed_encodings_per_run": args.runs,
            "warmup_encodings": 3,
            "allocator": "Each benchmark uses its production allocator",
            "cross_build_parity": "Full encodePrompt token arrays before timing",
            "reference": "Candidate must match llama.cpp on exact file bytes"
            if args.reference_tokenizer
            else None,
            "timed_parity": "Every timed encoding compared with its full reference",
        },
        "runner_sha256": sha(Path(__file__)),
        "sources_sha256": {
            label: {
                name: sha(root / name)
                for name in ("src/model/tokenizer.zig", "src/bench_tokenize.zig")
                if (root / name).is_file()
            }
            for label, root in builds.items()
        },
        "binaries": {
            label: {
                name: sha(root / name)
                for name in ("zinc-tokenize", "zinc-tokenize-bench")
            }
            for label, root in variants.items()
        },
        "models_sha256": {model.name: sha(model) for model in models},
        "corpora_sha256": {corpus.name: sha(corpus) for corpus in corpora},
        "references": [],
        "rows": [],
    }
    if args.reference_tokenizer:
        args.reference_tokenizer = args.reference_tokenizer.resolve()
        record["reference_binary_sha256"] = sha(args.reference_tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    expected = {}
    try:
        for model in models:
            for corpus in corpora:
                baseline = token_ids(
                    variants["baseline"] / "zinc-tokenize", model, corpus
                )
                candidate = token_ids(
                    variants["candidate"] / "zinc-tokenize", model, corpus
                )
                equal = baseline == candidate
                if not equal and not args.reference_tokenizer:
                    raise RuntimeError(f"Token mismatch: {model.name}, {corpus.name}")
                ids = {"baseline": baseline, "candidate": candidate}
                if args.reference_tokenizer:
                    reference = token_ids(
                        args.reference_tokenizer, model, corpus, reference=True
                    )
                    if candidate != reference:
                        raise RuntimeError(
                            f"Reference mismatch: {model.name}, {corpus.name}"
                        )
                    ids["reference"] = reference
                entry = {
                    "model": model.name,
                    "corpus": corpus.name,
                    "cross_build_match": equal,
                    "tokens": {label: len(tokens) for label, tokens in ids.items()},
                    "ids_sha256": {
                        label: hashlib.sha256(
                            json.dumps(tokens, separators=(",", ":")).encode()
                        ).hexdigest()
                        for label, tokens in ids.items()
                    },
                }
                if args.reference_tokenizer:
                    entry.update(
                        baseline_matches_reference=baseline == reference,
                        candidate_matches_reference=True,
                    )
                record["references"].append(entry)
                expected[model.name, corpus.name] = entry
                save()
            for trial in range(args.rounds):
                order = list(variants) if trial % 2 == 0 else list(reversed(variants))
                for label in order:
                    for corpus in corpora:
                        result = subprocess.run(
                            [
                                str(variants[label] / "zinc-tokenize-bench"),
                                str(model),
                                str(corpus),
                                str(args.runs),
                            ],
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=180,
                        )
                        row = json.loads(result.stdout)
                        counts = expected[model.name, corpus.name]["tokens"]
                        if row["tokens"] != counts[label]:
                            raise RuntimeError(
                                "Timed encoding has an unexpected token count"
                            )
                        row.update(
                            variant=label,
                            model=model.name,
                            corpus=corpus.name,
                            round=trial,
                        )
                        record["rows"].append(row)
                        save()
                        print(json.dumps(row), flush=True)
        for model in models:
            for corpus in corpora:
                rows = [
                    row
                    for row in record["rows"]
                    if row["model"] == model.name and row["corpus"] == corpus.name
                ]
                groups = [rows]
                if not expected[model.name, corpus.name]["cross_build_match"]:
                    groups = [
                        [row for row in rows if row["variant"] == label]
                        for label in variants
                    ]
                for group in groups:
                    if len({(row["token_hash"], row["tokens"]) for row in group}) != 1:
                        raise RuntimeError(
                            f"Timed token mismatch: {model.name}, {corpus.name}"
                        )
        record.update(
            complete=True,
            output_parity=all(
                entry["cross_build_match"] for entry in record["references"]
            ),
            candidate_reference_parity=True if args.reference_tokenizer else None,
        )
    finally:
        save()


if __name__ == "__main__":
    main()
