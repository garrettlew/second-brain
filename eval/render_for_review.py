"""
eval/render_for_review.py
Render a results CSV into a human-readable review document for manual scoring.

Usage:
    # Default input is single_agent_results_scored.csv in the repo root, output is review.md:
    python3 eval/render_for_review.py

    # Does same thing as above command; You can specify a different input CSV or output file:
    python3 eval/render_for_review.py --input single_agent_results_scored.csv --out review.md

    # Wrap output in minimal HTML for easier reading in a browser:
    python3 eval/render_for_review.py --input single_agent_results_scored.csv --out review.html --html

    # Also load the source vault so you can judge summary faithfulness against the original note
    python3 eval/render_for_review.py --vaultpath "/path/to/vault" --out review.md

    # Example vault
    python3 eval/render_for_review.py --vaultpath "/Users/michellesheu/Documents/Obsidian Vault/notion/UCSC" --out review.md

The --vaultpath flag loads each note's full source text from disk so summary
faithfulness can be judged against the original. Without it, the grader works
from the candidate previews shown to the model.
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent.parent

RUBRIC = """\
## Scoring Rubric

**Tag Relevance** (`tag_relevance_score`): 0–2
  - 0 = irrelevant, hallucinated, or completely off-topic
  - 1 = partially relevant or too broad/generic
  - 2 = highly relevant and specific to this note

**Summary Faithfulness** (`summary_faithfulness_score`): 0–3
  - 0 = completely wrong, incoherent, or entirely hallucinated
  - 1 = misses the key point, hallucinates facts, or is largely incoherent
  - 2 = adequate — captures basic premise but misses nuance or has minor inaccuracies
  - 3 = highly faithful — captures core concepts accurately without introducing external facts

**Link Quality** (`link_quality_score`): 0–2 average across links; null if no links proposed
  - 0 = irrelevant or hallucinated connection
  - 1 = plausible but superficial or overly obvious connection
  - 2 = highly useful — reveals a meaningful, specific thematic link

Fill scores directly into your copy of the CSV. This document is read-only.
"""


def load_source_note(vault_path: Path, note_id: str) -> str:
    note_file = vault_path / note_id
    if note_file.exists():
        return note_file.read_text(errors="ignore").strip()
    return ""


def parse_json_cell(value: str, fallback):
    if not value or value.strip() in ("", "[]", "{}"):
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


def first_present(row: dict, *keys, default=""):
    """Return the first column value that is present and non-empty."""
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return default


def render_row_md(row: dict, vault_path: Path | None, idx: int) -> str:
    note_id = row.get("note_id", "unknown")
    condition = row.get("condition", "")
    tags = parse_json_cell(first_present(row, "generated_tags", "tags"), [])
    summary = first_present(row, "generated_summary", "summary").strip()
    links = parse_json_cell(first_present(row, "generated_links", "links"), [])
    candidates = parse_json_cell(row.get("candidate_notes", ""), [])
    latency = first_present(row, "total_latency_seconds", "agent_latency_seconds", "latency_seconds")
    error = row.get("error", "").strip()

    existing_tag_score = row.get("tag_relevance_score", "").strip()
    existing_sum_score = row.get("summary_faithfulness_score", "").strip()
    existing_link_score = row.get("link_quality_score", "").strip()
    existing_comments = row.get("comments", "").strip()

    lines = []
    lines.append(f"---\n")
    lines.append(f"## Note {idx}: `{note_id}`")
    if condition:
        lines.append(f"*Condition: {condition}*")
    lines.append("")

    if error:
        lines.append(f"> **ERROR:** {error}")
        lines.append("")

    if vault_path:
        source_text = load_source_note(vault_path, note_id)
        if source_text:
            lines.append("### Source note")
            lines.append("```")
            lines.append(source_text)
            lines.append("```")
            lines.append("")

    lines.append("### Generated tags")
    if tags:
        lines.append(", ".join(f"`{t}`" for t in tags))
    else:
        lines.append("*(none — see error above)*" if error else "*(none)*")
    lines.append("")

    lines.append("### Summary")
    if summary:
        lines.append(summary)
    else:
        lines.append("*(none)*")
    lines.append("")

    lines.append("### Proposed links")
    if links:
        for link in links:
            title = link.get("note_title") or link.get("id", "?")
            justification = link.get("justification") or link.get("reason", "")
            lines.append(f"- **{title}** — {justification}")
    else:
        lines.append("*(none proposed)*")
    lines.append("")

    lines.append("### Candidate notes shown to the model")
    if candidates:
        for c in candidates:
            title = c.get("note_title", "?")
            preview = (c.get("content_preview") or c.get("summary") or "").replace("\\n", "\n").strip()
            distance = c.get("distance")
            dist_str = f" (distance {distance:.3f})" if isinstance(distance, float) else ""
            lines.append(f"**{title}**{dist_str}")
            if preview:
                lines.append(textwrap.indent(preview, "  > "))
            lines.append("")
    else:
        lines.append("*(none)*")
        lines.append("")

    lines.append(f"*Latency: {latency}s*")
    lines.append("")

    lines.append("### Scores (fill in)")
    lines.append(f"- `tag_relevance_score` (0–2): {existing_tag_score or '___'}")
    lines.append(f"- `summary_faithfulness_score` (0–3): {existing_sum_score or '___'}")
    lines.append(f"- `link_quality_score` (0–2): {existing_link_score or '___'}")
    lines.append(f"- `comments`: {existing_comments or '___'}")
    lines.append("")

    return "\n".join(lines)


def render_markdown(rows: list[dict], vault_path: Path | None) -> str:
    parts = []
    parts.append("# Manual Evaluation Review\n")
    parts.append(RUBRIC)
    parts.append("")
    for i, row in enumerate(rows, 1):
        parts.append(render_row_md(row, vault_path, i))
    return "\n".join(parts)


def wrap_html(md_content: str) -> str:
    escaped = md_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Manual Evaluation Review</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 2em auto; line-height: 1.5; }}
  pre {{ background: #f4f4f4; padding: 1em; overflow-x: auto; white-space: pre-wrap; }}
  code {{ background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }}
  hr {{ margin: 2em 0; }}
  blockquote {{ border-left: 3px solid #ccc; margin: 0; padding-left: 1em; color: #555; }}
</style>
</head>
<body>
<pre>{escaped}</pre>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Render evaluation CSV as a human-readable review doc.")
    parser.add_argument("--input", default=str(ROOT / "single_agent_results_scored.csv"),
                        help="Path to the results CSV (default: single_agent_results_scored.csv)")
    parser.add_argument("--out", default="review.md", help="Output file path (default: review.md)")
    parser.add_argument("--html", action="store_true", help="Wrap output in minimal HTML instead of raw Markdown")
    parser.add_argument("--vaultpath", default=None,
                        help="Path to the Obsidian vault to load source note text for faithfulness grading")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    vault_path = Path(args.vaultpath) if args.vaultpath else None

    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No rows found in CSV.")
        return

    md = render_markdown(rows, vault_path)
    output = wrap_html(md) if args.html else md

    out_path = Path(args.out)
    out_path.write_text(output)
    print(f"Wrote {len(rows)} note(s) to: {out_path}")


if __name__ == "__main__":
    main()
