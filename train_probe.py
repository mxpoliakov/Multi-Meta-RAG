"""Sweep every layer/pooling for the source probe, then dump the figure/results.

Class-weighted logistic regression under iterative-stratified 5-fold CV, with a
global threshold tuned on the out-of-fold probabilities; the best layer is also
re-trained as an MLP and compared to a fuzzy string-match baseline.

    python train_probe.py [--model Qwen/Qwen2.5-0.5B]
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

import probe_artifacts

DATASET_FILE = "MultiHop-RAG/dataset/MultiHopRAG.json"
GROUND_TRUTH_FILE = "query_metadata_filters_ground_truth.json"

N_SPLITS = 5
SEED = 0
THRESHOLDS = np.arange(0.05, 0.95, 0.05)


def load_labels() -> tuple[list[str], np.ndarray, MultiLabelBinarizer]:
    # Multi-hot source labels aligned with the hidden-state row order.
    with open(DATASET_FILE) as f:
        queries = [q["query"] for q in json.load(f)]
    with open(GROUND_TRUTH_FILE) as f:
        gt = {e["query"]: e["filter"] for e in json.load(f)}

    label_lists = []
    for q in queries:
        source = gt[q].get("source", {})
        label_lists.append(source.get("$in", []) if isinstance(source, dict) else [])

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(label_lists)
    return queries, labels, mlb


def best_threshold(labels: np.ndarray, proba: np.ndarray) -> float:
    # Global threshold maximising micro F1 on out-of-fold probabilities.
    scores = [
        f1_score(labels, proba >= t, average="micro", zero_division=0)
        for t in THRESHOLDS
    ]
    return float(THRESHOLDS[int(np.argmax(scores))])


def out_of_fold_proba(features: np.ndarray, labels: np.ndarray, factory) -> np.ndarray:
    # Cross-validated probabilities; every row scored by a model that did not
    # train on it, so the numbers transfer straight to prediction.
    splitter = MultilabelStratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )
    proba = np.zeros(labels.shape, dtype=float)
    for train_idx, test_idx in splitter.split(features, labels):
        model = factory()
        model.fit(features[train_idx], labels[train_idx])
        proba[test_idx] = model.predict_proba(features[test_idx])
    return proba


def evaluate(features: np.ndarray, labels: np.ndarray, factory) -> dict:
    proba = out_of_fold_proba(features, labels, factory)
    threshold = best_threshold(labels, proba)
    pred = proba >= threshold
    return {
        "micro_f1": f1_score(labels, pred, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, pred, average="macro", zero_division=0),
        "threshold": threshold,
    }


def logreg_factory():
    return make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(
                class_weight="balanced", solver="liblinear", C=1.0, max_iter=1000
            )
        ),
    )


def mlp_factory():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=300, random_state=SEED
        ),
    )


def string_match_scores(
    queries: list[str], labels: np.ndarray, classes: np.ndarray
) -> dict:
    # Baseline: predict a source when its name is a substring of the query.
    lowered = [q.lower() for q in queries]
    pred = np.zeros(labels.shape, dtype=int)
    for j, name in enumerate(classes):
        needle = name.lower()
        for i, query in enumerate(lowered):
            if needle in query:
                pred[i, j] = 1
    return {
        "micro_f1": f1_score(labels, pred, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, pred, average="macro", zero_division=0),
    }


def plot_sweep(sweep: dict, n_layers: int, model_name: str, figure_file) -> None:
    layers = list(range(n_layers))
    fig, (ax_micro, ax_macro) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for pooling, records in sweep.items():
        ax_micro.plot(layers, [r["micro_f1"] for r in records], marker="o", label=pooling)
        ax_macro.plot(layers, [r["macro_f1"] for r in records], marker="o", label=pooling)
    slug = probe_artifacts.model_slug(model_name)
    for ax, title in [(ax_micro, "micro F1"), (ax_macro, "macro F1")]:
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_ylabel(title)
        ax.set_title(f"Source probe ({slug}) — {title} vs depth")
        ax.grid(alpha=0.3)
        ax.legend(title="pooling")
    fig.tight_layout()
    fig.savefig(figure_file, dpi=150)
    print(f"Saved figure -> {figure_file}")


def main() -> None:
    model_name = probe_artifacts.parse_model()
    queries, labels, mlb = load_labels()
    print(f"Model: {model_name}")
    print(f"{labels.shape[0]} queries, {labels.shape[1]} sources")

    features = {
        "last": np.load(probe_artifacts.hidden_states_path("last", model_name)),
        "mean": np.load(probe_artifacts.hidden_states_path("mean", model_name)),
    }
    n_layers = next(iter(features.values())).shape[1]

    # Layer sweep with the logistic-regression probe, both pooling strategies.
    sweep: dict[str, list[dict]] = {}
    best = {"micro_f1": -1.0}
    for pooling, feats in features.items():
        records = []
        for layer in range(n_layers):
            result = evaluate(feats[:, layer, :], labels, logreg_factory)
            records.append(result)
            print(
                f"[{pooling}] layer {layer:>2}  "
                f"micro {result['micro_f1']:.3f}  macro {result['macro_f1']:.3f}"
            )
            if result["micro_f1"] > best["micro_f1"]:
                best = {**result, "pooling": pooling, "layer": layer}
        sweep[pooling] = records

    figure_file = probe_artifacts.figure_path(model_name)
    plot_sweep(sweep, n_layers, model_name, figure_file)

    print(
        f"\nBest LR probe: {best['pooling']} pooling, layer {best['layer']} "
        f"(micro {best['micro_f1']:.3f}, macro {best['macro_f1']:.3f})"
    )

    # 1-hidden-layer MLP on the best configuration, for capacity comparison.
    best_feats = features[best["pooling"]][:, best["layer"], :]
    mlp = evaluate(best_feats, labels, mlp_factory)
    print(
        f"MLP probe (same config): micro {mlp['micro_f1']:.3f}, "
        f"macro {mlp['macro_f1']:.3f}"
    )

    # Fuzzy string-match baseline over the 49 source names.
    baseline = string_match_scores(queries, labels, mlb.classes_)
    print(
        f"String-match baseline: micro {baseline['micro_f1']:.3f}, "
        f"macro {baseline['macro_f1']:.3f}"
    )

    results_file = probe_artifacts.results_path(model_name)
    with open(results_file, "w") as f:
        json.dump(
            {
                "model": model_name,
                "sweep": sweep,
                "best_logreg": best,
                "mlp": mlp,
                "string_match": baseline,
                "classes": mlb.classes_.tolist(),
            },
            f,
            indent=4,
        )
    print(f"Saved results -> {results_file}")


if __name__ == "__main__":
    main()
