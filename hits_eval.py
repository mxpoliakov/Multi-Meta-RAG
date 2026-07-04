"""Hit (predicted source set == ground-truth source set) evaluation for filters.

    python hits_eval.py [predictions.json]  # defaults to the GPT-3.5 baseline
"""

import json
import sys
from collections import defaultdict

DATASET_FILE = "MultiHop-RAG/dataset/MultiHopRAG.json"
GROUND_TRUTH_FILE = "query_metadata_filters_ground_truth.json"
DEFAULT_PREDICTIONS_FILE = "query_metadata_filters.json"


def source_set(filter_dict: dict) -> frozenset:
    source = filter_dict.get("source")
    if isinstance(source, dict) and isinstance(source.get("$in"), list):
        return frozenset(source["$in"])
    return frozenset()


def load_filters(path: str) -> dict[str, frozenset]:
    with open(path) as f:
        entries = json.load(f)
    return {e["query"]: source_set(e["filter"]) for e in entries}


def evaluate(predictions_file: str) -> None:
    with open(DATASET_FILE) as f:
        question_type = {q["query"]: q["question_type"] for q in json.load(f)}

    ground_truth = load_filters(GROUND_TRUTH_FILE)
    predictions = load_filters(predictions_file)

    # hits[type] = [n_hit, n_total]
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for query, gt in ground_truth.items():
        qtype = question_type[query]
        pred = predictions.get(query, frozenset())
        hit = pred == gt
        hits[qtype][0] += int(hit)
        hits[qtype][1] += 1

    print(f"Predictions: {predictions_file}")
    print(f"Ground truth: {GROUND_TRUTH_FILE}\n")
    print(f"{'question_type':<20}{'hits':>10}{'total':>10}{'accuracy':>12}")
    print("-" * 52)

    overall_hit = overall_total = 0
    nonnull_hit = nonnull_total = 0
    for qtype in sorted(hits):
        n_hit, n_total = hits[qtype]
        print(f"{qtype:<20}{n_hit:>10}{n_total:>10}{n_hit / n_total:>11.1%}")
        overall_hit += n_hit
        overall_total += n_total
        if qtype != "null_query":
            nonnull_hit += n_hit
            nonnull_total += n_total

    print("-" * 52)
    print(f"{'ALL (with null)':<20}{overall_hit:>10}{overall_total:>10}"
          f"{overall_hit / overall_total:>11.1%}")
    print(f"{'ALL (without null)':<20}{nonnull_hit:>10}{nonnull_total:>10}"
          f"{nonnull_hit / nonnull_total:>11.1%}")


if __name__ == "__main__":
    predictions_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREDICTIONS_FILE
    evaluate(predictions_file)
