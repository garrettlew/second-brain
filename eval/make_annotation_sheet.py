"""
eval/make_annotation_sheet.py
Generate a per-annotator scoring sheet for manual evaluation.

Usage:
    # Blank sheet for a new annotator
    python3 eval/make_annotation_sheet.py --annotator michelle

    # Seed Sona's sheet from the existing scored CSV
    python3 eval/make_annotation_sheet.py --annotator sona --seed-from-existing

    # Point at a different results file
    python3 eval/make_annotation_sheet.py --annotator michelle --input path/to/results.csv

    # Overwrite an existing sheet (e.g. to regenerate after new notes are added)
    python3 eval/make_annotation_sheet.py --annotator michelle --force

Output: eval/annotations/<annotator>.csv with columns:
    note_id, tag_relevance_score, summary_faithfulness_score, link_quality_score, comments

Scoring scales (same as eval/judge.py rubric):
    tag_relevance_score:          0–2
    summary_faithfulness_score:   1–3
    link_quality_score:           0–2  (leave blank if no links were proposed)
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
SCORE_COLS = ["tag_relevance_score", "summary_faithfulness_score", "link_quality_score", "comments"]


def main():
    parser = argparse.ArgumentParser(description="Generate a per-annotator scoring sheet.")
    parser.add_argument("--annotator", required=True,
                        help="Annotator name (used as filename: eval/annotations/<name>.csv)")
    parser.add_argument("--input", default=str(ROOT / "single_agent_results_scored.csv"),
                        help="Source results CSV to pull note_ids from (default: single_agent_results_scored.csv)")
    parser.add_argument("--seed-from-existing", action="store_true",
                        help="Copy existing scores from the results CSV instead of leaving blanks")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the sheet if it already exists")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    ANNOTATIONS_DIR.mkdir(exist_ok=True)
    out_path = ANNOTATIONS_DIR / f"{args.annotator}.csv"

    if out_path.exists() and not args.force:
        print(f"Sheet already exists: {out_path}")
        print("Use --force to overwrite (warning: this will erase in-progress scores).")
        return

    with open(input_path, newline="") as f:
        source_rows = list(csv.DictReader(f))

    if not source_rows:
        print("No rows found in source CSV.")
        return

    rows = []
    for r in source_rows:
        row = {"note_id": r["note_id"]}
        if args.seed_from_existing:
            for col in SCORE_COLS:
                row[col] = r.get(col, "")
        else:
            for col in SCORE_COLS:
                row[col] = ""
        rows.append(row)

    fieldnames = ["note_id"] + SCORE_COLS
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    action = "Seeded" if args.seed_from_existing else "Created blank"
    print(f"{action} sheet for '{args.annotator}' ({len(rows)} notes): {out_path}")
    if not args.seed_from_existing:
        print("Fill in the score columns using review.md as reference, then run:")
        print("    python3 eval/compare_annotations.py")


if __name__ == "__main__":
    main()
