# Manual Annotation Instructions

## Overview

Each annotator scores the pipeline's outputs (tags, summary, links) for 10 notes using a shared rubric. Scores are stored in `eval/annotations/<your-name>.csv`. After all annotators fill in their sheets, run the comparison script to see agreement statistics.

## Scoring Rubric

**Tag Relevance** (`tag_relevance_score`): 0–2
- 0 = irrelevant, hallucinated, or completely off-topic
- 1 = partially relevant or too broad/generic
- 2 = highly relevant and specific to this note

**Summary Faithfulness** (`summary_faithfulness_score`): 1–3
- 1 = misses the key point, hallucinates facts, or is incoherent
- 2 = adequate — captures basic premise but misses nuance or has minor inaccuracies
- 3 = highly faithful — captures core concepts accurately without introducing external facts

**Link Quality** (`link_quality_score`): 0–2
- 0 = irrelevant or hallucinated connection
- 1 = plausible but superficial or overly obvious connection
- 2 = highly useful — reveals a meaningful, specific thematic link
- Leave blank if no links were proposed for that note

## Workflow

### 1. Generate your annotation sheet
```bash
python3 eval/make_annotation_sheet.py --annotator <your-name>
```
This creates `eval/annotations/<your-name>.csv` with one row per note and blank score columns.

### 2. Generate the readable reference doc
```bash
python3 eval/render_for_review.py --out review.md
```
Open `review.md` — it shows each note's generated tags, summary, proposed links with justifications, and the candidate notes the model was shown. Use this as your reference while scoring.

To include the full source note text (recommended for judging summary faithfulness):
```bash
python3 eval/render_for_review.py --vaultpath "/Users/michellesheu/Documents/Obsidian Vault/notion/UCSC" --out review.md
```

### 3. Fill in your scores
Open `eval/annotations/<your-name>.csv` in a spreadsheet editor or text editor. For each note, enter your scores in the three columns using the rubric above. Add any notes in the `comments` column.

### 4. Run the comparison
```bash
python3 eval/compare_annotations.py
```
This prints a side-by-side table of all annotators' scores (flagging disagreements with `*`), agreement statistics, Cohen's kappa, and Krippendorff's alpha. It also saves `eval/annotations/comparison.csv`.

---

## Adding a new annotator
```bash
python3 eval/make_annotation_sheet.py --annotator <name>
```
Once they fill in their sheet, re-run `compare_annotations.py` — it picks up all files in `eval/annotations/` automatically.

## Existing annotations
- `eval/annotations/sona.csv` — Sona's scores (seeded from `single_agent_results_scored.csv`)
- `eval/annotations/michelle.csv` — blank sheet ready to fill in
