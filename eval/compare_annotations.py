"""
eval/compare_annotations.py
Compare annotation sheets from multiple human raters and compute inter-annotator agreement.

Usage:
    python3 eval/compare_annotations.py
    python3 eval/compare_annotations.py --annotations-dir eval/annotations

Output (to stdout + eval/annotations/comparison.csv):
    - Per-note side-by-side table of all annotators' scores
    - Exact-agreement % and mean absolute difference per metric
    - Cohen's quadratic-weighted kappa for each annotator pair
    - Krippendorff's ordinal alpha across all annotators
"""
import argparse
import csv
import itertools
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
METRICS = ["tag_relevance_score", "summary_faithfulness_score", "link_quality_score"]
METRIC_LABELS = {
    "tag_relevance_score":        "Tag Relevance (0–2)",
    "summary_faithfulness_score": "Summary Faithfulness (1–3)",
    "link_quality_score":         "Link Quality (0–2)",
}


def load_annotations(annotations_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all annotator CSV files. Returns {annotator_name: DataFrame indexed by note_id}."""
    sheets = {}
    for f in sorted(annotations_dir.glob("*.csv")):
        if f.name == "comparison.csv":
            continue
        df = pd.read_csv(f, dtype=str)
        df = df.set_index("note_id")
        for col in METRICS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        sheets[f.stem] = df
    return sheets


def side_by_side(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a wide DataFrame: one row per note, columns = annotator × metric."""
    annotators = list(sheets.keys())
    all_note_ids = sorted(set().union(*[set(df.index) for df in sheets.values()]))

    rows = []
    for note_id in all_note_ids:
        row = {"note_id": note_id}
        for ann in annotators:
            df = sheets[ann]
            for m in METRICS:
                val = df.loc[note_id, m] if note_id in df.index else np.nan
                row[f"{ann}__{m}"] = val
        rows.append(row)

    wide = pd.DataFrame(rows).set_index("note_id")
    return wide


def flag_disagreements(wide: pd.DataFrame, annotators: list[str]) -> pd.Series:
    """Return a boolean Series: True where annotators disagree on any metric."""
    disagrees = pd.Series(False, index=wide.index)
    for m in METRICS:
        cols = [f"{a}__{m}" for a in annotators if f"{a}__{m}" in wide.columns]
        if len(cols) < 2:
            continue
        sub = wide[cols].dropna(how="any")
        for idx in sub.index:
            vals = sub.loc[idx].values
            if len(set(vals)) > 1:
                disagrees[idx] = True
    return disagrees


def simple_agreement(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute exact-agreement % and mean absolute difference per metric."""
    annotators = list(sheets.keys())
    results = []
    for m in METRICS:
        cols_data = {}
        for ann in annotators:
            if m in sheets[ann].columns:
                cols_data[ann] = sheets[ann][m]
        if len(cols_data) < 2:
            continue

        combined = pd.DataFrame(cols_data).dropna()
        n = len(combined)
        if n == 0:
            continue

        # Exact agreement (all annotators give same score)
        exact = (combined.nunique(axis=1) == 1).sum() / n * 100

        # Mean pairwise absolute difference
        pairs = list(itertools.combinations(annotators, 2))
        pairwise_diffs = []
        for a1, a2 in pairs:
            if a1 in combined.columns and a2 in combined.columns:
                pairwise_diffs.extend((combined[a1] - combined[a2]).abs().tolist())
        mean_diff = np.mean(pairwise_diffs) if pairwise_diffs else np.nan

        results.append({
            "metric": METRIC_LABELS[m],
            "n_items": n,
            "exact_agreement_%": round(exact, 1),
            "mean_abs_diff": round(mean_diff, 3),
        })
    return pd.DataFrame(results)


def cohen_kappas(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute quadratic-weighted Cohen's kappa for every annotator pair × metric."""
    annotators = list(sheets.keys())
    rows = []
    for a1, a2 in itertools.combinations(annotators, 2):
        for m in METRICS:
            s1 = sheets[a1].get(m) if m in sheets[a1].columns else None
            s2 = sheets[a2].get(m) if m in sheets[a2].columns else None
            if s1 is None or s2 is None:
                continue
            combined = pd.concat([s1.rename("a1"), s2.rename("a2")], axis=1).dropna()
            n = len(combined)
            if n < 2:
                rows.append({"pair": f"{a1} vs {a2}", "metric": METRIC_LABELS[m],
                             "kappa": None, "n_items": n, "note": "too few items"})
                continue
            try:
                kappa = cohen_kappa_score(
                    combined["a1"].astype(int),
                    combined["a2"].astype(int),
                    weights="quadratic"
                )
                rows.append({"pair": f"{a1} vs {a2}", "metric": METRIC_LABELS[m],
                             "kappa": round(kappa, 4), "n_items": n, "note": ""})
            except Exception as e:
                rows.append({"pair": f"{a1} vs {a2}", "metric": METRIC_LABELS[m],
                             "kappa": None, "n_items": n, "note": str(e)})
    return pd.DataFrame(rows)


def krippendorff_alphas(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute Krippendorff's ordinal alpha across all annotators per metric."""
    annotators = list(sheets.keys())
    rows = []
    for m in METRICS:
        cols_data = {}
        for ann in annotators:
            if m in sheets[ann].columns:
                cols_data[ann] = sheets[ann][m]
        if len(cols_data) < 2:
            continue

        combined = pd.DataFrame(cols_data)
        n_complete = combined.dropna().shape[0]
        # krippendorff expects shape (n_raters, n_items); NaN = missing
        matrix = combined.T.values.astype(float)

        try:
            alpha = krippendorff.alpha(
                reliability_data=matrix,
                level_of_measurement="ordinal"
            )
            rows.append({"metric": METRIC_LABELS[m], "alpha": round(alpha, 4),
                         "n_raters": len(cols_data), "n_complete_items": n_complete, "note": ""})
        except Exception as e:
            rows.append({"metric": METRIC_LABELS[m], "alpha": None,
                         "n_raters": len(cols_data), "n_complete_items": n_complete, "note": str(e)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare annotation sheets and compute inter-rater agreement.")
    parser.add_argument("--annotations-dir", default=str(ANNOTATIONS_DIR),
                        help="Directory containing per-annotator CSVs (default: eval/annotations/)")
    args = parser.parse_args()

    ann_dir = Path(args.annotations_dir)
    sheets = load_annotations(ann_dir)

    if len(sheets) == 0:
        print("No annotation sheets found. Run make_annotation_sheet.py first.")
        return
    if len(sheets) == 1:
        print(f"Only one annotator found ({list(sheets.keys())[0]}). Add more sheets to compare.")
        return

    annotators = list(sheets.keys())
    print(f"\nAnnotators: {', '.join(annotators)}")
    print(f"Notes per sheet: {[len(df) for df in sheets.values()]}\n")

    # ── Side-by-side table ──────────────────────────────────────────────
    wide = side_by_side(sheets)
    disagrees = flag_disagreements(wide, annotators)

    print("=" * 70)
    print("PER-NOTE SIDE-BY-SIDE (tag / summary / link)  [* = disagreement]")
    print("=" * 70)
    header = f"{'Note':<35}" + "".join(f"{a:<28}" for a in annotators)
    print(header)
    print("-" * len(header))
    for note_id in wide.index:
        flag = " *" if disagrees[note_id] else "  "
        row_str = f"{(note_id[:33] + '..' if len(note_id) > 35 else note_id):<35}"
        for ann in annotators:
            tag = wide.loc[note_id, f"{ann}__tag_relevance_score"]
            summ = wide.loc[note_id, f"{ann}__summary_faithfulness_score"]
            link = wide.loc[note_id, f"{ann}__link_quality_score"]
            cell = f"{int(tag) if not np.isnan(tag) else '-'} / {int(summ) if not np.isnan(summ) else '-'} / {int(link) if not np.isnan(link) else '-'}"
            row_str += f"{cell:<28}"
        print(row_str + flag)

    # ── Save comparison CSV ─────────────────────────────────────────────
    out_csv = ann_dir / "comparison.csv"
    wide["disagreement"] = disagrees
    wide.to_csv(out_csv)
    print(f"\nSaved side-by-side table to: {out_csv}")

    # ── Simple agreement ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SIMPLE AGREEMENT SUMMARY")
    print("=" * 70)
    sa = simple_agreement(sheets)
    if not sa.empty:
        print(sa.to_string(index=False))

    # ── Cohen's kappa ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COHEN'S QUADRATIC-WEIGHTED KAPPA (per annotator pair)")
    print("  Interpretation: <0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate,")
    print("                  0.6–0.8 substantial, >0.8 almost perfect")
    print("=" * 70)
    ck = cohen_kappas(sheets)
    if not ck.empty:
        print(ck.to_string(index=False))
    else:
        print("Not enough annotator pairs or items to compute.")

    # ── Krippendorff's alpha ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KRIPPENDORFF'S ORDINAL ALPHA (all annotators)")
    print("  Interpretation: <0.667 tentative, 0.667–0.8 acceptable, >0.8 reliable")
    print("=" * 70)
    ka = krippendorff_alphas(sheets)
    if not ka.empty:
        print(ka.to_string(index=False))
    else:
        print("Not enough data to compute.")


if __name__ == "__main__":
    main()
