#!/usr/bin/env python3
"""Run only on the GPU node; validate three serial, scoped server variants."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import compare_builds as b

p = argparse.ArgumentParser(
    description="GPU-node-only MTP output, budget, reset and streaming validation"
)
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--candidate", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--cases", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", type=int, required=True)
p.add_argument("--sdk", action="store_true")
a = p.parse_args()
output = a.output.resolve()
if output.exists():
    p.error("output already exists; choose a new path")
output.parent.mkdir(parents=True, exist_ok=True)
cases = json.loads(a.cases.read_text())
variants = [
    ("baseline", a.baseline.resolve(), {"ZINC_MTP": "1"}),
    ("candidate", a.candidate.resolve(), {"ZINC_MTP": "1"}),
    ("ordinary", a.baseline.resolve(), {"ZINC_MTP": "0"}),
]
record = {"complete": False, "cases": cases, "rows": [], "checks": {}}
reference = {}
b.a = SimpleNamespace(
    output=output,
    device=a.device,
    stream=False,
    lock_clocks=False,
    candidate=None,
    candidate_env=[],
)


def save():
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)


def request(url, case, stream):
    payload = {
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": case["max_tokens"],
        "stream": stream,
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": False,
    }
    if "stop" in case:
        payload["stop"] = case["stop"]
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    parts = []
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=300) as response:
        if not stream:
            body = json.load(response)
            assert not body.get("error"), body
            choice = body["choices"][0]
            parts = [choice["message"]["content"]]
            finish = choice["finish_reason"]
            usage = body.get("usage")
        else:
            done = False
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                event = line[6:].strip()
                if event == b"[DONE]":
                    done = True
                    break
                body = json.loads(event)
                assert not body.get("error"), body
                if body.get("usage"):
                    usage = body["usage"]
                for choice in body.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        parts.append(content)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
            assert done, "stream ended before [DONE]"
    text = "".join(parts)
    assert finish in ("stop", "length") or (stream and finish is None), finish
    if not stream:
        assert usage is not None
        assert 0 <= usage["completion_tokens"] <= case["max_tokens"], usage
        assert usage["prompt_tokens"] + usage["completion_tokens"] <= 4096, usage
    canonical = {"text": text, "finish_reason": finish}
    return {
        "text": text,
        "finish_reason": finish,
        "usage": usage,
        "output_sha256": hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }


try:
    for label, directory, env in variants:
        workdir = directory
        b.a.candidate = workdir
        b.a.candidate_env = [f"{k}={v}" for k, v in env.items()]
        with b.server(workdir, a.model.resolve(), label) as (url, log, load_ms, proc):
            for case in cases:
                pair = []
                for stream in (False, True):
                    row = request(url, case, stream)
                    key = (case["id"], stream)
                    reference.setdefault(key, row["output_sha256"])
                    row.update(
                        variant=label,
                        case=case["id"],
                        stream=stream,
                        matches_baseline=row["output_sha256"] == reference[key],
                    )
                    record["rows"].append(row)
                    pair.append(row)
                    save()
                    if not row["matches_baseline"]:
                        raise RuntimeError(f"{label} changed {key}: {row}")
                if pair[0]["text"] != pair[1]["text"]:
                    raise RuntimeError(f"Streaming differs for {label}/{case['id']}")
            if label == "candidate" and a.sdk:
                sdk_log = output.with_suffix(".sdk.log")
                with sdk_log.open("w") as out:
                    sdk = subprocess.run(
                        ["bun", "tests/test_openai_sdk.ts", "--base-url", url + "/v1"],
                        cwd=workdir,
                        check=False,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                    )
                record["checks"]["sdk_exit_code"] = sdk.returncode
                record["checks"]["sdk_pass_lines"] = sum(
                    "PASS:" in line for line in sdk_log.read_text().splitlines()
                )
                record["checks"]["sdk_skip_lines"] = sum(
                    "SKIP" in line for line in sdk_log.read_text().splitlines()
                )
                assert sdk.returncode == 0, sdk_log.read_text()[-3000:]
            if label != "ordinary" and any(case["max_tokens"] >= 32 for case in cases):
                assert "NextN/MTP: request accepted" in log.read_text(), (
                    "MTP was not exercised"
                )
            print(
                json.dumps(
                    {
                        "variant": label,
                        "cases": len(cases),
                        "requests": 2 * len(cases),
                        "matched": sum(
                            r["matches_baseline"]
                            for r in record["rows"]
                            if r["variant"] == label
                        ),
                    }
                ),
                flush=True,
            )
    b.idle()
    record["checks"]["candidate_matches_baseline"] = all(
        r["matches_baseline"] for r in record["rows"] if r["variant"] == "candidate"
    )
    record["checks"]["ordinary_matches_baseline"] = all(
        r["matches_baseline"] for r in record["rows"] if r["variant"] == "ordinary"
    )
    record["complete"] = True
finally:
    save()
