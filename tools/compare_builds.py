#!/usr/bin/env python3
"""Matched reusable-server benchmarks. Run on the designated GPU node only."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--candidate", type=Path)
p.add_argument("--candidate-env", action="append", default=[])
p.add_argument("--model-root", type=Path, required=True)
p.add_argument("--models", default="Qwen3.5-9B-Q4_K_M.gguf")
p.add_argument("--runs", type=int, default=3)
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--rounds", type=int, default=1)
p.add_argument("--device", type=int, required=True)
p.add_argument("--scenarios", default="short,prefill,context")
p.add_argument("--stream", action="store_true")
p.add_argument(
    "--max-decode-regression-pct",
    type=float,
    help="Fail if any server round's median decode throughput regresses by more than this percent",
)
p.add_argument("--clock-mode", choices=("high", "profile_peak"), default="high")
p.add_argument(
    "--lock-clocks",
    action="store_true",
    help="Temporarily lock discrete AMD GPU clocks high; requires sysfs write access",
)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args() if __name__ == "__main__" else None
if a:
    if a.runs < 1 or a.rounds < 1 or a.warmup < 1:
        p.error("runs, rounds and warmup must be positive")
    if a.max_decode_regression_pct is not None and (
        not a.candidate or not 0 <= a.max_decode_regression_pct <= 100
    ):
        p.error(
            "decode regression limit requires a candidate and a value from 0 to 100"
        )
    a.model_root = a.model_root.resolve()
    a.output = a.output.resolve()
    a.output.parent.mkdir(parents=True, exist_ok=True)


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
    signal.signal(sig, interrupted)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def idle():
    for name in ("zinc", "llama-server", "llama-cli", "llama-bench"):
        rc = subprocess.run(["pgrep", "-x", name], stdout=subprocess.PIPE, check=False)
        if rc.returncode == 0:
            raise RuntimeError("GPU process already active: " + name)


@contextlib.contextmanager
def clocks():
    if not a.lock_clocks:
        yield
        return
    saved = []
    try:
        for path in Path("/sys/class/drm").glob("card*/device/pp_dpm_mclk"):
            if not re.fullmatch(r"card[0-9]+", path.parent.parent.name):
                continue
            # Ignore integrated graphics; its clock interface can report EBUSY.
            if int(path.with_name("mem_info_vram_total").read_text()) < 8 * 1024**3:
                continue
            control = path.with_name("power_dpm_force_performance_level")
            saved.append((control, control.read_text().strip()))
            control.write_text(a.clock_mode)
        yield
    finally:
        for control, value in reversed(saved):
            control.write_text(value)


@contextlib.contextmanager
def server(workdir, model, label):
    idle()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    logfile = a.output.parent / (a.output.stem + "-" + label + ".log")
    env = os.environ.copy()
    env["RADV_PERFTEST"] = "coop_matrix"
    env["ZINC_SHADER_DIR"] = str(workdir / "zig-out/share/zinc/shaders")
    if a.candidate and workdir == a.candidate.resolve():
        for item in a.candidate_env:
            k, v = item.split("=", 1)
            env[k] = v
    cmd = [
        str(workdir / "zig-out/bin/zinc"),
        "-m",
        str(model),
        "-d",
        str(a.device),
        "--context",
        "4096",
        "--parallel",
        "1",
        "--port",
        str(port),
    ]
    started = time.perf_counter()
    with logfile.open("wb", buffering=0) as log:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            url = "http://127.0.0.1:" + str(port)
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        "server exited: " + logfile.read_text(errors="replace")[-4000:]
                    )
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("server startup timeout")
            with clocks():
                yield url, logfile, (time.perf_counter() - started) * 1000, proc
            if proc.poll() is not None:
                raise RuntimeError("server exited unexpectedly: " + str(logfile))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=15)


def request(url, logfile, prompt, tokens):
    offset = logfile.stat().st_size
    payload = {"max_tokens": tokens, "stream": a.stream, "temperature": 0, "top_p": 1}
    if a.stream:
        payload.update(
            messages=[{"role": "user", "content": prompt}], enable_thinking=False
        )
    else:
        payload["prompt"] = prompt
    body = json.dumps(payload).encode()
    endpoint = "/v1/chat/completions" if a.stream else "/v1/completions"
    req = urllib.request.Request(
        url + endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    arrivals = []
    with urllib.request.urlopen(req, timeout=300) as response:
        if not a.stream:
            result = json.load(response)
        else:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise RuntimeError("server did not return SSE")
            parts = []
            finish = None
            done = False
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                for choice in event.get("choices", []):
                    token = choice.get(
                        "text", choice.get("delta", {}).get("content", "")
                    )
                    if token:
                        parts.append(token)
                        arrivals.append((time.perf_counter() - started) * 1000)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
            if not done:
                raise RuntimeError("SSE response ended without [DONE]")
            result = {"choices": [{"text": "".join(parts), "finish_reason": finish}]}
    latency = (time.perf_counter() - started) * 1000
    if result.get("error"):
        raise RuntimeError(result["error"])
    deadline = time.monotonic() + 10
    while True:
        with logfile.open("r", errors="replace") as f:
            f.seek(offset)
            text = f.read()
        gen = re.findall(
            r"Generated\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+ms\s+[—-]\s+([\d.]+)\s+tok/s",
            text,
        )
        pf = re.findall(
            r"Prefill(?:\s+complete)?\s*:\s*(\d+)\s+tokens\s+in\s+([\d.]+)\s*ms\s*\(([\d.]+)\s+tok/s\)",
            text,
        )
        if gen and pf:
            break
        if time.monotonic() > deadline:
            raise RuntimeError("missing server timings: " + text[-4000:])
        time.sleep(0.01)
    usage = result.get(
        "usage",
        {
            "prompt_tokens": int(pf[-1][0]),
            "completion_tokens": int(gen[-1][0]),
            "total_tokens": int(pf[-1][0]) + int(gen[-1][0]),
        },
    )
    choice = result["choices"][0]
    digest = hashlib.sha256(
        json.dumps(
            {
                "usage": usage,
                "text": choice.get("text"),
                "finish_reason": choice.get("finish_reason"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "generated_tokens": usage["completion_tokens"],
        "decode_tps": float(gen[-1][2]),
        "prefill_tps": float(pf[-1][2]),
        "decode_ms": float(gen[-1][1]),
        "prefill_ms": float(pf[-1][1]),
        "latency_ms": latency,
        "output_sha256": digest,
        "ttft_ms": arrivals[0] if arrivals else None,
        "inter_chunk_p95_ms": sorted(y - x for x, y in zip(arrivals, arrivals[1:]))[  # noqa: RUF007 (Python 3.9)
            int((len(arrivals) - 2) * 0.95)
        ]
        if len(arrivals) > 1
        else None,
        "chunks": len(arrivals),
        "loadavg": list(os.getloadavg()),
    }


base = "List consecutive positive integers separated by spaces, starting at 1, and continue for as long as possible. 1 2 3 4 5"
context = "The copper robot catalogues quiet stars. Each archive entry contains a name, an orbital period, and a short description. "
scenarios = {
    "short": (base, 256),
    "prefill": (context * 32 + "\n" + base, 128),
    "context": (context * 96 + "\n" + base, 128),
}


def shader_digest(workdir):
    digest = hashlib.sha256()
    for file in sorted((workdir / "zig-out/share/zinc/shaders").glob("*.spv")):
        digest.update(file.name.encode())
        digest.update(bytes.fromhex(sha(file)))
    return digest.hexdigest()


def compare_decode(rows, max_regression_pct):
    """Compare each independent server round so a fast restart cannot hide a slow one."""
    groups = {}
    for row in rows:
        key = (row["model"], row["scenario"], row["round"])
        groups.setdefault(key, {}).setdefault(row["variant"], []).append(
            row["decode_tps"]
        )
    comparisons = []
    for (model, scenario, trial), variants in sorted(groups.items()):
        if "candidate" not in variants:
            continue
        baseline = statistics.median(variants["baseline"])
        candidate = statistics.median(variants["candidate"])
        change = 100 * (candidate / baseline - 1) if baseline > 0 else None
        comparisons.append(
            {
                "model": model,
                "scenario": scenario,
                "round": trial,
                "baseline_tps": baseline,
                "candidate_tps": candidate,
                "change_pct": change,
                "passes": max_regression_pct is None
                or (change is not None and change >= -max_regression_pct),
            }
        )
    return comparisons


if __name__ == "__main__":
    for scenario in a.scenarios.split(","):
        if scenario not in scenarios:
            p.error("unknown scenario: " + scenario)
    variants = {"baseline": a.baseline.resolve()}
    if a.candidate:
        variants["candidate"] = a.candidate.resolve()
    record = {
        "methodology": {
            "runs": a.runs,
            "warmup": a.warmup,
            "rounds": a.rounds,
            "server_reused": True,
            "alternating_order": True,
            "context": 4096,
            "kv_quant": 0,
            "sampling": "greedy",
            "dpm": a.clock_mode + ", restored after run"
            if a.lock_clocks
            else "unchanged",
            "io": "loopback chat completions"
            if a.stream
            else "loopback raw completions",
            "stream": a.stream,
            "max_decode_regression_pct": a.max_decode_regression_pct,
        },
        "candidate_environment": dict(item.split("=", 1) for item in a.candidate_env),
        "binaries": {k: sha(v / "zig-out/bin/zinc") for k, v in variants.items()},
        "shader_tree_sha256": {k: shader_digest(v) for k, v in variants.items()},
        "rows": [],
    }

    def save():
        a.output.write_text(json.dumps(record, indent=2) + "\n")

    try:
        idle()
        with contextlib.nullcontext():
            for model_name in a.models.split(","):
                model = a.model_root / model_name
                model_sha = sha(model)
                for trial in range(a.rounds):
                    order = list(variants)
                    if trial % 2:
                        order.reverse()
                    for variant in order:
                        label = model.stem + "-" + variant + "-" + str(trial)
                        with server(variants[variant], model, label) as (
                            url,
                            logfile,
                            startup,
                            proc,
                        ):
                            for scenario in a.scenarios.split(","):
                                prompt, tokens = scenarios[scenario]
                                for warm in range(a.warmup):
                                    request(url, logfile, prompt, tokens)
                                for run in range(a.runs):
                                    row = request(url, logfile, prompt, tokens)
                                    row.update(
                                        model=model_name,
                                        model_sha256=model_sha,
                                        variant=variant,
                                        round=trial,
                                        run=run,
                                        scenario=scenario,
                                        startup_ms=startup,
                                    )
                                    record["rows"].append(row)
                                    save()
                                    print(
                                        json.dumps(
                                            {
                                                k: row[k]
                                                for k in (
                                                    "model",
                                                    "variant",
                                                    "round",
                                                    "run",
                                                    "scenario",
                                                    "prompt_tokens",
                                                    "generated_tokens",
                                                    "prefill_tps",
                                                    "decode_tps",
                                                    "latency_ms",
                                                )
                                            }
                                        ),
                                        flush=True,
                                    )
        outputs = {}
        for row in record["rows"]:
            key = (row["model"], row["scenario"])
            outputs.setdefault(key, set()).add(row["output_sha256"])
        record["output_parity"] = all(len(hashes) == 1 for hashes in outputs.values())
        if not record["output_parity"]:
            raise RuntimeError("output parity failed; inspect the recorded hashes")
        record["decode_comparison"] = compare_decode(
            record["rows"], a.max_decode_regression_pct
        )
        if any(not row["passes"] for row in record["decode_comparison"]):
            raise RuntimeError("decode throughput regressed; inspect decode_comparison")
        record["complete"] = True
    finally:
        save()
