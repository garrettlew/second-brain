"""
eval/run_evaluation.py
Sweeps all (architecture × note × run_idx) cells and logs results to
experiments/<architecture>.jsonl via main.py.

Usage:
    python3 eval/run_evaluation.py                          # all architectures, all notes, 3 runs
    python3 eval/run_evaluation.py --architectures single_agent multi_agent
    python3 eval/run_evaluation.py --notes "CSE30/10-01.md" "Graphs.md" --runs 1
    python3 eval/run_evaluation.py --dry-run               # print commands without running
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORPUS_FILE = Path(__file__).parent / "corpus.json"

ALL_ARCHITECTURES = ["single_agent", "multi_agent", "multi_agent_overseer"]


def load_corpus() -> dict:
    with CORPUS_FILE.open() as f:
        return json.load(f)


def build_command(vault_path: str, note_id: str, architecture: str,
                  run_idx: int, agent_model: str, embed_model: str,
                  temperature: float) -> list[str]:
    return [
        sys.executable, str(ROOT / "main.py"),
        "--vaultpath",    vault_path,
        "--inputfile",    note_id,
        "--architecture", architecture,
        "--experiment",   architecture,
        "--agent-model",  agent_model,
        "--embed-model",  embed_model,
        "--temperature",  str(temperature),
        "--label",        f"eval_run_{run_idx}",
    ]


def run_cell(cmd: list[str], note_id: str, architecture: str, run_idx: int) -> bool:
    print(f"  [{architecture}] {note_id} (run {run_idx + 1}) ... ", end="", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.perf_counter() - t0, 1)
    if result.returncode == 0:
        print(f"✓ {elapsed}s")
        return True
    else:
        print(f"✗ {elapsed}s — ERROR")
        print(f"    stderr: {result.stderr[-300:]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation sweep.")
    parser.add_argument("--architectures", nargs="+", default=ALL_ARCHITECTURES,
                        choices=ALL_ARCHITECTURES)
    parser.add_argument("--notes", nargs="+", default=None,
                        help="Note IDs to evaluate (default: all in corpus.json).")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per cell (default: 3).")
    parser.add_argument("--agent-model", default="qwen3.5:9b")
    parser.add_argument("--embed-model", default="mxbai-embed-large")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    args = parser.parse_args()

    corpus = load_corpus()
    vault_path = corpus["vault_path"]
    all_notes = [n["id"] for n in corpus["notes"]]
    notes = args.notes if args.notes else all_notes

    total_cells = len(args.architectures) * len(notes) * args.runs
    print(f"Evaluation sweep: {len(args.architectures)} architectures × "
          f"{len(notes)} notes × {args.runs} runs = {total_cells} cells\n")

    success = 0
    failure = 0

    for architecture in args.architectures:
        print(f"\n── Architecture: {architecture} ──")
        for note_id in notes:
            for run_idx in range(args.runs):
                cmd = build_command(vault_path, note_id, architecture, run_idx,
                                    args.agent_model, args.embed_model, args.temperature)
                if args.dry_run:
                    print("  DRY RUN:", " ".join(cmd))
                    continue
                ok = run_cell(cmd, note_id, architecture, run_idx)
                if ok:
                    success += 1
                else:
                    failure += 1

    if not args.dry_run:
        print(f"\n{'='*50}")
        print(f"Done: {success} succeeded, {failure} failed out of {total_cells} cells.")
        print(f"Results in: experiments/{{single_agent,multi_agent,multi_agent_overseer}}.jsonl")


if __name__ == "__main__":
    main()
