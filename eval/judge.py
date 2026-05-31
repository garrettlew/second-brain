"""
eval/judge.py
LLM-as-judge: reads experiments/<arch>.jsonl rows and appends judge_scores to each.
Writes scored results to experiments/<arch>_scored.jsonl.

Usage:
    python3 eval/judge.py                              # score all architecture files
    python3 eval/judge.py --architecture single_agent
    python3 eval/judge.py --judge-model qwen3.5:9b    # default
"""
import argparse
import json
import ollama
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator for a note enrichment system. Given the original note text and the system's output (tags, summary, links), score the quality of each output component.

Scoring rubric:

Tag Relevance (score each tag 0-2, then report average):
  0 = irrelevant, hallucinated, or completely off-topic
  1 = partially relevant or too broad/generic
  2 = highly relevant and specific to this note

Summary Faithfulness (1-3):
  1 = misses the key point, hallucinates facts, or is incoherent
  2 = adequate — captures the basic premise but misses nuance or has minor inaccuracies
  3 = highly faithful — captures the core concepts accurately without introducing external facts

Link Quality (score each link 0-2 if present, report average; if no links, score null):
  0 = irrelevant or hallucinated connection
  1 = plausible but superficial or overly obvious connection
  2 = highly useful — reveals a meaningful, specific thematic link

Return ONLY a JSON object. No markdown fencing. No preamble.

Output format:
{
  "tag_scores": [score_per_tag],
  "tag_avg": float,
  "summary_faithfulness": integer 1-3,
  "link_scores": [score_per_link] or null,
  "link_avg": float or null,
  "reasoning": "brief explanation of scores (1-2 sentences)"
}"""


def score_row(row: dict, judge_client: ollama.Client, judge_model: str) -> dict:
    """Score a single experiment row and return judge_scores dict."""
    note_path_str = row.get("note_path", "unknown")
    tags = row.get("tags", [])
    summary = row.get("summary", "")
    links = row.get("links", [])

    # Try to read the original note text for better judging
    corpus_file = ROOT / "eval" / "corpus.json"
    note_text = ""
    try:
        corpus = json.loads(corpus_file.read_text())
        vault_path = corpus["vault_path"]
        note_file = Path(vault_path) / note_path_str
        if note_file.exists():
            note_text = note_file.read_text()[:2000]  # cap to avoid huge prompts
    except Exception:
        pass

    user_message = (
        f"Original note (truncated to 2000 chars):\n{note_text or '[not available]'}\n\n"
        f"System output:\n"
        f"Tags: {json.dumps(tags)}\n"
        f"Summary: {summary}\n"
        f"Links: {json.dumps(links)}"
    )

    try:
        response = judge_client.chat(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            format="json",
            stream=False,
            think=False,
            options={"temperature": 0.0},  # deterministic judge
        )
        return json.loads(response["message"]["content"])
    except Exception as e:
        return {"error": str(e)}


def score_file(jsonl_path: Path, judge_client: ollama.Client, judge_model: str,
               overwrite: bool = False):
    out_path = jsonl_path.parent / jsonl_path.name.replace(".jsonl", "_scored.jsonl")

    if out_path.exists() and not overwrite:
        # Find already-scored note_paths to skip them
        scored_keys = set()
        with out_path.open() as f:
            for line in f:
                row = json.loads(line)
                scored_keys.add((row.get("note_path"), row.get("architecture"),
                                  row.get("label"), row.get("timestamp")))
    else:
        scored_keys = set()

    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"  Scoring {len(rows)} rows from {jsonl_path.name} → {out_path.name}")
    with out_path.open("a") as out_f:
        for i, row in enumerate(rows):
            key = (row.get("note_path"), row.get("architecture"),
                   row.get("label"), row.get("timestamp"))
            if key in scored_keys:
                print(f"    [{i+1}/{len(rows)}] SKIP (already scored): {row.get('note_path')}")
                continue
            print(f"    [{i+1}/{len(rows)}] Scoring: {row.get('note_path')} "
                  f"({row.get('architecture')}) ...", end=" ", flush=True)
            scores = score_row(row, judge_client, judge_model)
            row["judge_scores"] = scores
            row["judge_model"] = judge_model
            out_f.write(json.dumps(row) + "\n")
            tag_avg = scores.get("tag_avg", "?")
            sf = scores.get("summary_faithfulness", "?")
            print(f"tag_avg={tag_avg} sf={sf}")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge for experiment results.")
    parser.add_argument("--architecture", default=None,
                        choices=["single_agent", "multi_agent", "multi_agent_overseer"],
                        help="Score only this architecture (default: all found in experiments/).")
    parser.add_argument("--judge-model", default="qwen3.5:9b",
                        help="Ollama model to use as judge (default: qwen3.5:9b).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-score already-scored rows.")
    args = parser.parse_args()

    judge_client = ollama.Client(host="http://localhost:11434")

    if args.architecture:
        files = [EXPERIMENTS_DIR / f"{args.architecture}.jsonl"]
    else:
        files = list(EXPERIMENTS_DIR.glob("*.jsonl"))
        files = [f for f in files if "_scored" not in f.name]

    if not files:
        print(f"No .jsonl files found in {EXPERIMENTS_DIR}. Run eval/run_evaluation.py first.")
        return

    for jsonl_path in sorted(files):
        if not jsonl_path.exists():
            print(f"  Skipping (not found): {jsonl_path}")
            continue
        score_file(jsonl_path, judge_client, args.judge_model, args.overwrite)

    print("\nDone. Scored files written to experiments/*_scored.jsonl")
    print("Tip: run eval/analyze.py to generate the comparison table and plots.")


if __name__ == "__main__":
    main()
