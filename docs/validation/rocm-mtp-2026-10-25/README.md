# ROCm MTP follow-up evidence

The [report](../../ROCM_MTP_0_17.md) compares published baseline
`b81f9946f021d62481e41384e731f236ff2e407a` with runtime candidate
`cad5327110a8895781bf0318e98d441548747596`. Both use Zig 0.17.0, ROCm
ReleaseFast, the same Qwen 3.8 27B Q4_K_M GGUF and the existing automatic GPU
power policy. All GPU workloads ran serially on the designated remote GPU host.

`archives.json` records the byte length and SHA256 of each uncompressed artifact
and deterministic gzip archive. Decompress with `gzip -dc` or Python's
`gzip.open`. No model weights or executable binaries are included.

| Artifact | Coverage |
| --- | --- |
| `final.json.gz` | Primary result: 72 measured streaming requests, 48 warmups, three alternating server rounds; source, binary, kernel and forward-code fingerprints |
| `summary.json.gz` | Derived medians, per-round medians and improvement percentages; input SHA256 |
| `suite-prompts.json.gz` | Four exact public performance-suite prompts and output limits |
| `final-variants.json.gz` | Relative build-directory configuration for the primary matrix |
| `screen.json.gz` | Eight settings, four workloads, 64 measured requests; baseline binary with vocabulary/depth/adaptive overrides |
| `fusion-screen.json.gz` | Four variants, four workloads, 32 measured requests; fusion prototype defaults to 65,536 draft rows |
| `four-screen.json.gz` | Four variants, four workloads, 32 measured requests; rejected four-draft prototype |
| `rejected-four-draft.patch.gz` | Rejected prototype relative to `cad53271`; apply only in a separate experimental checkout |
| `api-cases.json.gz` | Seventeen exact API case inputs |
| `api-validation.json.gz` | 102 JSON/SSE responses from baseline MTP, candidate MTP and ordinary decode; all parity checks pass; existing generic SDK health-metadata failure explicitly retained |
| `api-public-smoke.json.gz` | Published helper smoke: four boundary/reset cases, three variants, two response modes, 24 matching responses |
| `metadata.json.gz` | Model checksum, hardware/software versions, compiler/archive hashes, global compiler update and GPU validator outcome |
| `final-node-state.json.gz` | Final system/wrapper Zig 0.17.0 checks, preserved previous compiler, unchanged runtime fingerprints, automatic power policy and zero remaining inference processes |
| `verification.json.gz` | Local and remote test coverage, 60-case GPU/CPU validator and negative control, API coverage and limitations |
| `kernel-profile.json.gz` | Fifty aggregated kernel rows, 43,206 dispatches from a diagnostic baseline 96-token CLI run including initialization and prefill |

The exploratory screens use two measured requests after one warmup per cell.
They selected the final settings and are not pooled with the primary
three-round measurements. `candidate` in the fusion screen is the earlier
fusion-only prototype; `candidate-four` is the rejected five-row verifier.
Their fingerprints identify these intermediate binaries. The final matrix
uses the frozen `cad53271` runtime with three drafts and 98,304 draft rows.

The four-draft patch also restores the prototype's 65,536-row default, adds
five-row projection kernels and a fourth recurrent-state history slot, and
allows four drafts through the server. Its CLI draft limit remains three;
the experiment exercised the server. The patch passes `git apply --check`
against `cad53271`. It is evidence, not an enabled feature.

The full API record combines 68 baseline/candidate responses with a later
34-response ordinary-decode pass after the optional generic SDK suite stopped
on missing health metadata. The focused cases all completed and matched. The
published helper was then run separately for the 24-response smoke. Existing
ROCm omissions of SSE finish metadata and stop-parameter handling are recorded
in the report; the stop case checks behavioral parity and subsequent reset.

The profile excludes agent, queue and process trace identifiers. Its aggregated
durations include model setup and prefill, so they must not be interpreted as
the primary server's decode-only distribution.
