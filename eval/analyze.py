"""
eval/analyze.py
Reads experiments/*_scored.jsonl and produces:
  1. Quality comparison table (mean ± std per condition)
  2. Latency vs. quality tradeoff scatter plot
  3. Per-stage latency breakdown bar chart
  4. Token cost comparison

Usage:
    python3 eval/analyze.py                    # all scored files
    python3 eval/analyze.py --output eval/results/
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
DEFAULT_OUTPUT = ROOT / "eval" / "results"

ARCH_ORDER = ["single_agent", "multi_agent", "multi_agent_overseer"]
ARCH_COLORS = {
    "single_agent":         "#4C72B0",
    "multi_agent":          "#55A868",
    "multi_agent_overseer": "#C44E52",
}
ARCH_LABELS = {
    "single_agent":         "Single Agent (A)",
    "multi_agent":          "Multi-Agent (B)",
    "multi_agent_overseer": "Multi-Agent + Overseer (C)",
}


def load_scored_data(experiments_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(experiments_dir.glob("*_scored.jsonl")):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                scores = row.get("judge_scores", {})
                stages = row.get("stages", [])
                stage_latency = {s["stage"]: s.get("elapsed", 0) for s in stages}
                total_tokens = sum(
                    s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)
                    for s in stages
                )
                rows.append({
                    "architecture":         row.get("architecture", "?"),
                    "note_path":            row.get("note_path", "?"),
                    "label":                row.get("label", ""),
                    "tag_avg":              scores.get("tag_avg"),
                    "summary_faithfulness": scores.get("summary_faithfulness"),
                    "link_avg":             scores.get("link_avg"),
                    "latency_s":            row.get("elapsed_seconds"),
                    "inference_calls":      row.get("inference_calls"),
                    "total_tokens":         total_tokens,
                    "peak_rss_delta_mb":    row.get("peak_rss_delta_mb"),
                    "latency_tagger":       stage_latency.get("tagger"),
                    "latency_summarizer":   stage_latency.get("summarizer"),
                    "latency_linker":       stage_latency.get("linker"),
                    "latency_single":       stage_latency.get("single_agent"),
                    "latency_overseer":     stage_latency.get("overseer"),
                })
    return pd.DataFrame(rows)


def print_quality_table(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("QUALITY COMPARISON TABLE (mean ± std)")
    print("=" * 70)
    metrics = ["tag_avg", "summary_faithfulness", "link_avg", "latency_s", "total_tokens"]
    avail = [c for c in metrics if c in df.columns]

    table = (
        df.groupby("architecture")[avail]
        .agg(["mean", "std"])
        .round(3)
    )
    present_archs = [a for a in ARCH_ORDER if a in table.index]
    print(table.loc[present_archs].to_string())
    print()


def plot_quality_table(df: pd.DataFrame, output_dir: Path):
    metrics = {
        "tag_avg":              ("Tag Quality (0–2)", 0, 2),
        "summary_faithfulness": ("Summary Faithfulness (1–3)", 1, 3),
        "link_avg":             ("Link Quality (0–2)", 0, 2),
    }
    avail = {k: v for k, v in metrics.items() if k in df.columns}
    if not avail:
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 4), sharey=False)
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, (title, ymin, ymax)) in zip(axes, avail.items()):
        present = [a for a in ARCH_ORDER if a in df["architecture"].unique()]
        data = [df[df["architecture"] == a][col].dropna().values for a in present]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, arch in zip(bp["boxes"], present):
            patch.set_facecolor(ARCH_COLORS.get(arch, "#888888"))
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels([ARCH_LABELS.get(a, a) for a in present],
                            rotation=15, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(ymin - 0.1, ymax + 0.1)

    fig.suptitle("Quality Metrics by Architecture", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "quality_boxplots.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_tradeoff(df: pd.DataFrame, output_dir: Path):
    """Latency (x) vs quality (y) scatter — the centerpiece slide."""
    quality_col = "tag_avg" if "tag_avg" in df.columns else None
    if quality_col is None or "latency_s" not in df.columns:
        print("Skipping tradeoff plot (missing columns).")
        return

    df_clean = df.dropna(subset=[quality_col, "latency_s"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in ARCH_ORDER:
        sub = df_clean[df_clean["architecture"] == arch]
        if sub.empty:
            continue
        ax.scatter(
            sub["latency_s"], sub[quality_col],
            color=ARCH_COLORS[arch], alpha=0.35, s=40,
            label=f"{ARCH_LABELS[arch]} (n={len(sub)})"
        )
        # Per-architecture mean marker
        mx, my = sub["latency_s"].mean(), sub[quality_col].mean()
        ax.scatter(mx, my, color=ARCH_COLORS[arch], s=180, marker="*",
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(f"  {ARCH_LABELS[arch]}\n  ({mx:.1f}s, {my:.2f})",
                    (mx, my), fontsize=7.5, color=ARCH_COLORS[arch])

    ax.set_xlabel("End-to-end latency (seconds)", fontsize=10)
    ax.set_ylabel("Tag Quality — avg score (0–2)", fontsize=10)
    ax.set_title("Latency vs. Quality Tradeoff by Architecture", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = output_dir / "latency_vs_quality.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_stage_latency(df: pd.DataFrame, output_dir: Path):
    """Stacked bar of per-stage latency per architecture."""
    stage_cols = {
        "single_agent":         ["latency_single"],
        "multi_agent":          ["latency_tagger", "latency_summarizer", "latency_linker"],
        "multi_agent_overseer": ["latency_tagger", "latency_summarizer",
                                  "latency_linker", "latency_overseer"],
    }
    stage_colors = {
        "latency_single":       "#4C72B0",
        "latency_tagger":       "#55A868",
        "latency_summarizer":   "#C44E52",
        "latency_linker":       "#8172B2",
        "latency_overseer":     "#CCB974",
    }
    stage_names = {
        "latency_single":       "Single Agent",
        "latency_tagger":       "Tagger",
        "latency_summarizer":   "Summarizer",
        "latency_linker":       "Linker",
        "latency_overseer":     "Overseer",
    }

    present = [a for a in ARCH_ORDER if a in df["architecture"].unique()]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(present))
    bottoms = [0.0] * len(present)
    handles = []

    all_stages = ["latency_tagger", "latency_summarizer", "latency_linker",
                  "latency_overseer", "latency_single"]
    for stage_col in all_stages:
        if stage_col not in df.columns:
            continue
        heights = []
        for arch in present:
            val = df[df["architecture"] == arch][stage_col].mean()
            heights.append(val if pd.notna(val) else 0)
        bars = ax.bar(x, heights, bottom=bottoms, color=stage_colors[stage_col],
                      label=stage_names[stage_col], alpha=0.85)
        handles.append(bars[0])
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(list(x))
    ax.set_xticklabels([ARCH_LABELS.get(a, a) for a in present], fontsize=8)
    ax.set_ylabel("Latency (seconds)", fontsize=10)
    ax.set_title("Per-Stage Latency Breakdown", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "stage_latency_breakdown.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def save_summary_csv(df: pd.DataFrame, output_dir: Path):
    metrics = ["tag_avg", "summary_faithfulness", "link_avg",
               "latency_s", "inference_calls", "total_tokens"]
    avail = [c for c in metrics if c in df.columns]
    summary = (
        df.groupby("architecture")[avail]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    out = output_dir / "summary_table.csv"
    summary.to_csv(out)
    print(f"Saved: {out}")

    raw_out = output_dir / "all_results.csv"
    df.to_csv(raw_out, index=False)
    print(f"Saved: {raw_out}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Directory to save plots and CSVs.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_scored_data(EXPERIMENTS_DIR)
    if df.empty:
        print("No scored results found. Run eval/run_evaluation.py then eval/judge.py first.")
        return

    print(f"Loaded {len(df)} rows across architectures: {df['architecture'].unique().tolist()}")

    print_quality_table(df)
    plot_quality_table(df, output_dir)
    plot_tradeoff(df, output_dir)
    plot_stage_latency(df, output_dir)
    save_summary_csv(df, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
