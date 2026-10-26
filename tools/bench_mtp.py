#!/usr/bin/env python3
"""Run only on the designated GPU node; every server is scoped to one context."""

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import compare_builds as b

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--variants", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--prompts", type=Path, required=True)
p.add_argument("--device", type=int, required=True)
p.add_argument("--runs", type=int, default=2)
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--rounds", type=int, default=1)
p.add_argument(
    "--scenarios", default="core,context-medium,context-long,decode-extended"
)
a = p.parse_args()
if min(a.runs, a.warmup, a.rounds) < 1:
    p.error("runs, warmup and rounds must be positive")
if a.output.exists():
    p.error("output already exists; choose a new path to preserve earlier measurements")
root = a.root.resolve()
variants = json.loads(a.variants.read_text())
scenarios = [
    s for s in json.loads(a.prompts.read_text()) if s["id"] in a.scenarios.split(",")
]
if len(scenarios) != len(set(a.scenarios.split(","))):
    p.error("missing or duplicate scenario IDs in prompts")
if not variants or len({v["id"] for v in variants}) != len(variants):
    p.error("variants must have distinct IDs")
for variant in variants:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", variant["id"]):
        p.error(
            "variant IDs may contain only letters, numbers, dots, dashes and underscores"
        )
    for key in variant.get("env", {}):
        if not key.startswith(
            ("ZINC_MTP", "ZINC_SPEC_", "ZINC_Q4K_BTOK_", "ZINC_Q5K_BTOK_")
        ):
            p.error("only MTP performance switches may be set by a variant")
a.output.parent.mkdir(parents=True, exist_ok=True)
b.a = SimpleNamespace(
    output=a.output,
    device=a.device,
    stream=True,
    lock_clocks=False,
    candidate=None,
    candidate_env=[],
)
model = a.model.resolve()
record = {
    "model": model.name,
    "model_size": model.stat().st_size,
    "runs": a.runs,
    "warmup": a.warmup,
    "rounds": a.rounds,
    "context": 4096,
    "backend": "rocm",
    "sampling": "greedy",
    "dpm": "unchanged",
    "rows": [],
    "variants": variants,
    "complete": False,
}
reference = {}
fingerprints = {}
for variant in variants:
    directory = root / variant["directory"]
    fingerprints[variant["id"]] = {
        "binary_sha256": b.sha(directory / "zig-out/bin/zinc"),
        "kernels_sha256": b.sha(directory / "src/shaders/cuda/kernels.cu"),
        "forward_sha256": b.sha(directory / "src/compute/forward_cuda.zig"),
    }
    revision_file = directory / "SOURCE_REVISION"
    if revision_file.is_file():
        revision = revision_file.read_text().strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision):
            fingerprints[variant["id"]]["source_revision"] = revision
record["fingerprints"] = fingerprints


def save():
    temporary = a.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(a.output)


try:
    for trial in range(a.rounds):
        order = variants if trial % 2 == 0 else list(reversed(variants))
        for variant in order:
            directory = root / variant["directory"]
            b.a.candidate = directory
            b.a.candidate_env = [
                f"{key}={value}" for key, value in variant.get("env", {}).items()
            ]
            with b.server(directory, model, f"{variant['id']}-r{trial}") as (
                url,
                log,
                load_ms,
                proc,
            ):
                for scenario in scenarios:
                    measured = []
                    for run in range(-a.warmup, a.runs):
                        offset = log.stat().st_size
                        row = b.request(
                            url, log, scenario["prompt"], scenario["max_tokens"]
                        )
                        requires_mtp = str(
                            variant.get("env", {}).get("ZINC_MTP", "1")
                        ).lower() not in {"0", "off", "false", "no"}
                        deadline = time.monotonic() + 5
                        while True:
                            with log.open(errors="replace") as stream:
                                stream.seek(offset)
                                text = stream.read()
                            acceptance = re.findall(
                                r"NextN/MTP: request accepted (\d+)/(\d+) draft tokens \(([\d.]+)%\) over (\d+) cycles",
                                text,
                            )
                            if (
                                acceptance
                                or not requires_mtp
                                or time.monotonic() > deadline
                            ):
                                break
                            time.sleep(0.001)
                        if acceptance:
                            accepted, drafted, percent, cycles = acceptance[-1]
                            row.update(
                                accepted=int(accepted),
                                drafted=int(drafted),
                                acceptance_pct=float(percent),
                                cycles=int(cycles),
                            )
                        if requires_mtp and not acceptance:
                            raise RuntimeError(
                                f"Missing MTP statistics for {variant['id']}/{scenario['id']}"
                            )
                        key = scenario["id"]
                        reference.setdefault(key, row["output_sha256"])
                        row["matches_reference"] = (
                            row["output_sha256"] == reference[key]
                        )
                        if not row["matches_reference"]:
                            raise RuntimeError(
                                f"Output mismatch for {variant['id']}/{key}: {row}"
                            )
                        if run >= 0:
                            row.update(
                                variant=variant["id"],
                                scenario=key,
                                round=trial,
                                run=run,
                                load_ms=load_ms,
                            )
                            record["rows"].append(row)
                            measured.append(row)
                            save()
                    print(
                        json.dumps(
                            {
                                "variant": variant["id"],
                                "scenario": scenario["id"],
                                "round": trial,
                                "decode_tps": statistics.median(
                                    r["decode_tps"] for r in measured
                                ),
                                "acceptance_pct": statistics.median(
                                    r.get("acceptance_pct", 0) for r in measured
                                ),
                                "latency_ms": statistics.median(
                                    r["latency_ms"] for r in measured
                                ),
                            }
                        ),
                        flush=True,
                    )
    b.idle()
    record["complete"] = True
finally:
    save()
