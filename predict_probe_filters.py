"""Write probe source filters and compare hit accuracy against GPT-3.5.

    python predict_probe_filters.py [--model Qwen/Qwen2.5-0.5B]
"""

import json

import numpy as np

import hits_eval
import probe_artifacts
import train_probe as probe

GPT_FILE = "query_metadata_filters.json"


def main() -> None:
    model_name = probe_artifacts.parse_model()
    slug = probe_artifacts.model_slug(model_name)
    results_file = probe_artifacts.results_path(model_name)
    output_file = probe_artifacts.probe_filters_path(model_name)
    with open(results_file) as f:
        best = json.load(f)["best_logreg"]
    pooling, layer = best["pooling"], best["layer"]
    print(f"Model: {model_name}")
    print(f"Best probe: {pooling} pooling, layer {layer}")

    queries, labels, mlb = probe.load_labels()
    all_layers = np.load(probe_artifacts.hidden_states_path(pooling, model_name))
    # Hidden-state index 0 = embeddings, so transformer layers = n_states - 1.
    n_transformer_layers = all_layers.shape[1] - 1
    features = all_layers[:, layer, :]

    # Out-of-fold so no query is scored by a model that trained on it.
    proba = probe.out_of_fold_proba(features, labels, probe.logreg_factory)
    threshold = probe.best_threshold(labels, proba)
    predictions = proba >= threshold

    classes = mlb.classes_
    query_filters = []
    for query, row in zip(queries, predictions):
        sources = sorted(classes[row].tolist())
        # Empty prediction -> empty filter (null-query convention).
        filter_dict = {"source": {"$in": sources}} if sources else {}
        query_filters.append({"query": query, "filter": filter_dict})

    with open(output_file, "w") as f:
        json.dump(query_filters, f, indent=4, sort_keys=True)
    print(f"Wrote {len(query_filters)} probe filters to {output_file}\n")

    print("=" * 52)
    print("PROBE")
    print("=" * 52)
    hits_eval.evaluate(str(output_file))
    print("\n" + "=" * 52)
    print("GPT-3.5")
    print("=" * 52)
    hits_eval.evaluate(GPT_FILE)

    # The probe runs a partial forward pass locally
    depth_fraction = layer / n_transformer_layers
    print("\n" + "=" * 52)
    print("COST / LATENCY")
    print("=" * 52)
    print(
        f"Probe:   local partial forward pass through {layer}/{n_transformer_layers}"
        f" layers ({depth_fraction:.0%}) of {slug} + a linear head."
    )

if __name__ == "__main__":
    main()
