#!/usr/bin/env python3
"""Render the blog's Gemma TTFT chart from the archived RDNA measurements.

Requires matplotlib. Run from the repository root; optional --preview writes
a PNG for visual review alongside the standalone SVG used by the site.
"""

import argparse
import gzip
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/validation/zig-0.17-bpe/vulkan-final.json.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/public/blog/2026-10-24-zig-017-bpe-gemma-ttft.svg"),
    )
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.input.suffix == ".gz" else open
    with opener(args.input, "rt") as source:
        record = json.load(source)
    if not record.get("complete") or not record.get("output_parity"):
        raise ValueError("chart requires a completed comparison with output parity")
    model = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    rows = [row for row in record["rows"] if row["model"] == model]
    scenarios = ("short", "prefill", "context")
    variants = ("baseline", "candidate")
    rounds = sorted({row["round"] for row in rows})
    if len(rounds) != 3 or any(
        sum(row["scenario"] == case and row["variant"] == variant for row in rows) != 9
        for case in scenarios
        for variant in variants
    ):
        raise ValueError("expected three rounds and nine requests per case and build")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Liberation Sans",
                "sans-serif",
            ],
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "zinc-zig17-bpe",
            "axes.edgecolor": "#cbd5df",
            "text.color": "#163c41",
            "axes.labelcolor": "#36545c",
            "xtick.color": "#526972",
            "ytick.color": "#163c41",
        }
    )
    background = "#f6f8fa"
    colors = {"baseline": "#90a3b4", "candidate": "#087f78"}
    offsets = {"baseline": -0.16, "candidate": 0.16}
    fig, ax = plt.subplots(figsize=(12, 6.6), facecolor=background)
    ax.set_facecolor(background)
    medians = {}
    labels = []
    for index, case in enumerate(scenarios):
        case_rows = [row for row in rows if row["scenario"] == case]
        counts = {row["prompt_tokens"] for row in case_rows}
        if len(counts) != 1:
            raise ValueError("prompt token counts differ")
        name = {
            "short": "Short prompt",
            "prefill": "Medium prompt",
            "context": "Long prompt",
        }[case]
        labels.append(f"{name}\n{counts.pop():,} prompt tokens")
        for variant in variants:
            selected = [row for row in case_rows if row["variant"] == variant]
            median = statistics.median(row["ttft_ms"] for row in selected)
            medians[case, variant] = median
            y = index + offsets[variant]
            ax.barh(y, median, height=0.25, color=colors[variant], zorder=2)
            ax.text(median + 24, y, f"{median:,.1f} ms", va="center", fontsize=11)
            for trial, jitter in zip(rounds, (-0.045, 0, 0.045)):
                value = statistics.median(
                    row["ttft_ms"] for row in selected if row["round"] == trial
                )
                ax.scatter(
                    value,
                    y + jitter,
                    s=23,
                    facecolor="white",
                    edgecolor="#163c41",
                    linewidth=0.8,
                    zorder=3,
                )
        reduction = 100 * (1 - medians[case, "candidate"] / medians[case, "baseline"])
        change = (
            "Essentially level"
            if abs(reduction) < 0.5
            else f"{reduction:.1f}% less time"
        )
        ax.text(
            2240,
            index + 0.16,
            change,
            ha="right",
            va="center",
            color="#087f78",
            fontsize=11,
            weight="bold",
        )

    ax.set_yticks(range(len(scenarios)), labels)
    ax.tick_params(axis="y", length=0, pad=15)
    ax.set_ylim(2.55, -0.55)
    ax.set_xlim(0, 2300)
    ax.set_xticks((0, 500, 1000, 1500, 2000))
    ax.set_xlabel(
        "Time to first visible output (milliseconds) · lower is better", labelpad=12
    )
    ax.grid(axis="x", color="#dce4e9", linewidth=0.8, zorder=0)
    for edge in ("top", "right", "left"):
        ax.spines[edge].set_visible(False)
    ax.legend(
        handles=[
            Patch(color=colors["baseline"], label="Zig 0.17 baseline"),
            Patch(color=colors["candidate"], label="Zig 0.17 + BPE changes"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="#163c41",
                label="Server-round median",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.02),
        frameon=False,
        ncol=3,
        fontsize=10,
        handlelength=1.6,
        columnspacing=1.5,
    )
    fig.text(
        0.035,
        0.935,
        "Faster prompt encoding cuts Gemma’s first wait",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.035,
        0.885,
        "Gemma 4 26B-A4B · Q4_K_M · Radeon AI PRO R9700 · Vulkan",
        fontsize=12,
    )
    fig.text(
        0.035,
        0.055,
        "Bars: median of 9 measured requests per build and prompt. Dots: medians of 3 independent server rounds.\n"
        "2 warmups per case and server · greedy sampling · 4,096-token context · automatic GPU power management.\n"
        "Same models, prompts and shaders. Baseline: the already optimized Zig 0.17 migration (9bb61b12).",
        fontsize=9,
        color="#526972",
        linespacing=1.7,
    )
    fig.subplots_adjust(left=0.185, right=0.975, top=0.76, bottom=0.24)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        metadata={
            "Date": None,
            "Creator": "ZINC benchmark chart",
            "Title": "Gemma first-output latency after BPE optimization",
        },
    )
    if args.output.suffix.lower() == ".svg":
        # Matplotlib pads path-data lines with spaces. Normalize serialization
        # so regeneration also passes the repository's whitespace check.
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines())
            + "\n"
        )
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
