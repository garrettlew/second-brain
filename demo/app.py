"""
demo/app.py — Second Brain: Single Agent vs Multi-Agent Demo

Run with:
    source .venv/bin/activate
    streamlit run demo/app.py
"""
import json
import sys
import time
from pathlib import Path

import streamlit as st

# Allow imports from the repo root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from main import (
    Agent, load_prompt, get_vault_candidates,
    run_single_agent, run_multi_agent, run_multi_agent_overseer,
    peak_rss_mb,
)
import chromadb
import ollama

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VAULT_PATH = Path("/Users/michellesheu/Documents/Obsidian Vault/notion/UCSC")
AGENT_MODEL = "qwen3.5:9b"
EMBED_MODEL = "mxbai-embed-large"
TEMPERATURE = 0.3

ARCH_LABELS = {
    "single_agent":          "🔵 Single Agent (Condition A)",
    "multi_agent":           "🟢 Multi-Agent Pipeline (Condition B)",
    "multi_agent_overseer":  "🟠 Multi-Agent + Overseer (Condition C)",
}

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Ollama and ChromaDB...")
def get_resources():
    model_client = ollama.Client(host="http://localhost:11434")
    prompts = {
        "tagger":        load_prompt("tagger", "v1"),
        "summarizer":    load_prompt("summarizer", "v1"),
        "linker":        load_prompt("linker", "v1"),
        "single_agent":  load_prompt("single_agent", "v1"),
        "overseer":      load_prompt("overseer", "v1"),
    }
    agent = Agent(model_client, prompts, model_type=AGENT_MODEL, temperature=TEMPERATURE)
    collection = chromadb.PersistentClient().get_or_create_collection("second-brain")

    def embed(text: str) -> list[float]:
        return model_client.embeddings(prompt=text, model=EMBED_MODEL)["embedding"]

    return agent, collection, embed, model_client


def get_note_files() -> list[str]:
    return sorted(
        str(p.relative_to(VAULT_PATH))
        for p in VAULT_PATH.rglob("*.md")
    )


def run_architecture(arch: str, note_text: str, note_id: str,
                     agent, collection, embed_fn) -> dict:
    if arch == "single_agent":
        return run_single_agent(agent, note_text, collection, embed_fn, note_id)
    elif arch == "multi_agent":
        return run_multi_agent(agent, note_text, collection, embed_fn, note_id)
    elif arch == "multi_agent_overseer":
        return run_multi_agent_overseer(agent, note_text, collection, embed_fn, note_id)
    raise ValueError(arch)


def render_result(result: dict, arch: str):
    tags = result.get("tags", [])
    summary = result.get("summary", "")
    links = result.get("links", [])
    elapsed = result.get("elapsed_seconds", 0)
    calls = result.get("inference_calls", 0)
    stages = result.get("stages", [])
    total_tokens = sum(
        s.get("prompt_tokens", 0) + s.get("completion_tokens", 0) for s in stages
    )

    # System metrics row
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("⏱ Latency", f"{elapsed}s")
    col_b.metric("📞 LLM Calls", calls)
    col_c.metric("🔤 Total Tokens", total_tokens)

    # Tags
    st.markdown("**Tags**")
    st.write(" · ".join(f"`{t}`" for t in tags) if tags else "_none_")

    # Summary
    st.markdown("**Summary**")
    st.info(summary or "_none_")

    # Links
    st.markdown("**Suggested Links**")
    if links:
        for lnk in links:
            st.write(f"📎 **{lnk.get('note_title', '?')}** — {lnk.get('justification', '')}")
    else:
        st.write("_no links (vault may need indexing first)_")

    # Overseer details (Condition C only)
    if arch == "multi_agent_overseer" and result.get("overseer_review"):
        rv = result["overseer_review"]
        with st.expander("🔍 Overseer Review"):
            approved = rv.get("approved", False)
            st.write(f"**Approved:** {'✅ Yes' if approved else '❌ No'}")
            issues = rv.get("issues_found", [])
            if issues:
                st.write("**Issues found:**")
                for issue in issues:
                    st.write(f"  - {issue}")
            if rv.get("retry_recommended"):
                st.warning("Overseer recommended a retry (capped at 1).")

    # Stage-level breakdown
    if stages:
        with st.expander("📊 Per-stage breakdown"):
            for s in stages:
                p, c = s.get("prompt_tokens", 0), s.get("completion_tokens", 0)
                st.write(
                    f"**{s['stage']}** — {s['elapsed']}s | "
                    f"prompt: {p} tok | completion: {c} tok"
                )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Second Brain — Agent Architecture Demo",
    page_icon="🧠",
    layout="wide",
)
st.title("🧠 Second Brain — Agent Architecture Comparison")
st.caption("CSE 232B · Single Agent (A) vs. Multi-Agent Pipeline (B) vs. + Overseer (C)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    note_files = get_note_files()
    selected_note = st.selectbox("Select a note", note_files,
                                  index=note_files.index("CSE30/10-01.md")
                                  if "CSE30/10-01.md" in note_files else 0)
    architectures_to_run = st.multiselect(
        "Architectures to compare",
        options=list(ARCH_LABELS.keys()),
        default=["single_agent", "multi_agent"],
        format_func=lambda k: ARCH_LABELS[k],
    )
    run_btn = st.button("▶ Run Enrichment", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Vault path**")
    st.code(str(VAULT_PATH), language=None)
    st.markdown("**Models**")
    st.write(f"Agent: `{AGENT_MODEL}`")
    st.write(f"Embed: `{EMBED_MODEL}`")
    st.write(f"Temp: `{TEMPERATURE}`")

# Load note
note_path = VAULT_PATH / selected_note
note_text = note_path.read_text() if note_path.exists() else ""

# Original note preview
with st.expander("📄 Original Note", expanded=False):
    st.markdown(note_text or "_note not found_")

# Run
if run_btn and architectures_to_run:
    agent, collection, embed_fn, _ = get_resources()

    cols = st.columns(len(architectures_to_run))
    for col, arch in zip(cols, architectures_to_run):
        with col:
            st.subheader(ARCH_LABELS[arch])
            with st.spinner(f"Running {arch}..."):
                rss_before = peak_rss_mb()
                result = run_architecture(arch, note_text, selected_note,
                                          agent, collection, embed_fn)
                rss_delta = round(peak_rss_mb() - rss_before, 2)
            result["peak_rss_delta_mb"] = rss_delta
            render_result(result, arch)

# Results comparison table (from JSONL logs)
st.divider()
st.subheader("📈 Evaluation Results (from logged experiments)")

import pandas as pd

scored_files = list((ROOT / "experiments").glob("*_scored.jsonl"))
if not scored_files:
    st.info("No scored results yet. Run `eval/run_evaluation.py` then `eval/judge.py` to populate this table.")
else:
    rows = []
    for f in scored_files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    scores = row.get("judge_scores", {})
                    rows.append({
                        "Architecture": row.get("architecture", "?"),
                        "Note": row.get("note_path", "?"),
                        "Tag Avg": scores.get("tag_avg"),
                        "Summary Faithfulness": scores.get("summary_faithfulness"),
                        "Link Avg": scores.get("link_avg"),
                        "Latency (s)": row.get("elapsed_seconds"),
                        "LLM Calls": row.get("inference_calls"),
                    })

    if rows:
        df = pd.DataFrame(rows)
        summary = (
            df.groupby("Architecture")
            .agg(
                Tag_Avg_mean=("Tag Avg", "mean"),
                Tag_Avg_std=("Tag Avg", "std"),
                Summary_Faith_mean=("Summary Faithfulness", "mean"),
                Summary_Faith_std=("Summary Faithfulness", "std"),
                Link_Avg_mean=("Link Avg", "mean"),
                Latency_mean=("Latency (s)", "mean"),
                Latency_std=("Latency (s)", "std"),
                N=("Note", "count"),
            )
            .round(3)
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)

        # Tradeoff plot
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            df_clean = df.dropna(subset=["Tag Avg", "Latency (s)"])
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.scatterplot(
                data=df_clean, x="Latency (s)", y="Tag Avg",
                hue="Architecture", alpha=0.5, ax=ax,
            )
            # Per-architecture means
            means = df_clean.groupby("Architecture")[["Latency (s)", "Tag Avg"]].mean()
            for arch_name, row in means.iterrows():
                ax.scatter(row["Latency (s)"], row["Tag Avg"], s=200,
                           marker="*", zorder=5, label=f"{arch_name} mean")
            ax.set_title("Latency vs. Tag Quality — Architecture Tradeoff")
            ax.set_xlabel("End-to-end latency (seconds)")
            ax.set_ylabel("Tag Avg (0–2)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render tradeoff plot: {e}")

        with st.expander("Raw results table"):
            st.dataframe(df, use_container_width=True)
