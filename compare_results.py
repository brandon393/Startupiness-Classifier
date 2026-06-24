#!/usr/bin/env python3
"""
compare_results.py — Deterministic confusion matrix + mismatch report
=======================================================================
Compares manual ground-truth classifications against LLM classifier
output and prints a 4x4 confusion matrix, accuracy, and full mismatch
list. No AI involved in computing or transcribing these numbers —
this script is the single source of truth.

Usage:
    python compare_results.py manual.csv results.jsonl
    python compare_results.py manual.csv results.csv
    python compare_results.py manual.csv results.jsonl --mismatches-out mismatches.csv

Input formats:
    manual.csv   — must have columns: domain, classification
    results.*    — .jsonl (one JSON object per line, with "domain" and
                   "classification" keys) OR .csv with the same two
                   columns. Extra columns/fields are ignored.

Output:
    - Confusion matrix (manual rows x model columns) printed to stdout
    - Overall accuracy
    - Full list of mismatches printed to stdout
    - Optional: mismatches written to a CSV via --mismatches-out
"""

import sys
import json
import argparse
import csv
from collections import defaultdict

CATEGORIES = ["early", "growth", "established", "non-startup"]


def load_classifications(path):
    """
    Load a domain -> classification mapping from either a .csv file
    (columns: domain, classification) or a .jsonl file (one JSON
    object per line with "domain" and "classification" keys).
    """
    mapping = {}
    if path.lower().endswith(".jsonl"):
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping malformed JSON on line {line_num} of {path}: {e}", file=sys.stderr)
                    continue
                domain = row.get("domain")
                cls = row.get("classification")
                if domain is None or cls is None:
                    print(f"⚠️  Skipping line {line_num} of {path}: missing domain or classification", file=sys.stderr)
                    continue
                mapping[domain.strip()] = cls.strip()
    else:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "domain" not in reader.fieldnames or "classification" not in reader.fieldnames:
                print(f"❌  {path} must have 'domain' and 'classification' columns. "
                      f"Found: {reader.fieldnames}", file=sys.stderr)
                sys.exit(1)
            for row in reader:
                domain = row["domain"]
                cls = row["classification"]
                if not domain or not cls:
                    continue
                mapping[domain.strip()] = cls.strip()
    return mapping


def validate_categories(mapping, source_label):
    """Warn about any classification values outside the expected 4 categories."""
    seen = set(mapping.values())
    unexpected = seen - set(CATEGORIES)
    if unexpected:
        print(f"⚠️  {source_label} contains unexpected classification values: {sorted(unexpected)}", file=sys.stderr)


def build_confusion_matrix(manual, model):
    """
    Build a confusion matrix over the domains present in BOTH inputs.
    Returns (matrix, common_domains, manual_only, model_only).
    matrix[manual_label][model_label] = count
    """
    manual_domains = set(manual.keys())
    model_domains = set(model.keys())
    common = manual_domains & model_domains

    matrix = defaultdict(lambda: defaultdict(int))
    for domain in common:
        m_label = manual[domain]
        p_label = model[domain]
        matrix[m_label][p_label] += 1

    return matrix, common, manual_domains - model_domains, model_domains - manual_domains


def print_confusion_matrix(matrix, categories):
    col_width = 14
    header = "Manual\\Model".ljust(16) + "".join(c.ljust(col_width) for c in categories) + "Total"
    print(header)
    print("-" * len(header))
    for m in categories:
        row_total = sum(matrix[m][c] for c in categories)
        row = m.ljust(16) + "".join(str(matrix[m][c]).ljust(col_width) for c in categories) + str(row_total)
        print(row)
    col_totals = [sum(matrix[m][c] for m in categories) for c in categories]
    print("-" * len(header))
    print("Total".ljust(16) + "".join(str(t).ljust(col_width) for t in col_totals) + str(sum(col_totals)))


def main():
    parser = argparse.ArgumentParser(description="Compare manual classifications against LLM classifier output.")
    parser.add_argument("manual_file", help="Path to manual ground-truth file (.csv)")
    parser.add_argument("model_file", help="Path to model results file (.jsonl or .csv)")
    parser.add_argument("--mismatches-out", default=None, help="Optional path to write mismatches as a CSV")
    args = parser.parse_args()

    manual = load_classifications(args.manual_file)
    model = load_classifications(args.model_file)

    validate_categories(manual, args.manual_file)
    validate_categories(model, args.model_file)

    matrix, common, manual_only, model_only = build_confusion_matrix(manual, model)

    if manual_only:
        print(f"⚠️  {len(manual_only)} domain(s) in manual file but not in model results "
              f"(excluded from comparison): {sorted(manual_only)[:10]}{'...' if len(manual_only) > 10 else ''}",
              file=sys.stderr)
    if model_only:
        print(f"⚠️  {len(model_only)} domain(s) in model results but not in manual file "
              f"(excluded from comparison): {sorted(model_only)[:10]}{'...' if len(model_only) > 10 else ''}",
              file=sys.stderr)

    total = len(common)
    if total == 0:
        print("❌  No overlapping domains between the two files. Nothing to compare.", file=sys.stderr)
        sys.exit(1)

    correct = sum(1 for d in common if manual[d] == model[d])
    mismatches = sorted(d for d in common if manual[d] != model[d])

    print(f"\nCompared {total} domains (manual file: {len(manual)}, model file: {len(model)})")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%\n")

    print_confusion_matrix(matrix, CATEGORIES)

    print(f"\nMismatches ({len(mismatches)}):")
    for d in mismatches:
        print(f"  {d:35s} manual={manual[d]:14s} model={model[d]}")

    if args.mismatches_out:
        with open(args.mismatches_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["domain", "manual", "model"])
            for d in mismatches:
                writer.writerow([d, manual[d], model[d]])
        print(f"\nMismatches written to {args.mismatches_out}")


if __name__ == "__main__":
    main()
