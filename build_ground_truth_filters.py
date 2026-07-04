"""Ground-truth source filters from the gold evidence of each query."""

import json

DATASET_FILE = "MultiHop-RAG/dataset/MultiHopRAG.json"
OUTPUT_FILE = "query_metadata_filters_ground_truth.json"


def build_ground_truth() -> list[dict]:
    with open(DATASET_FILE) as f:
        query_data_list = json.load(f)

    query_filters = []
    for query in query_data_list:
        sources = sorted({e["source"] for e in query["evidence_list"]})
        # Null queries carry no evidence -> empty filter.
        filter_dict = {"source": {"$in": sources}} if sources else {}
        query_filters.append({"query": query["query"], "filter": filter_dict})
    return query_filters


if __name__ == "__main__":
    query_filters = build_ground_truth()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(query_filters, f, indent=4, sort_keys=True)
    print(f"Wrote {len(query_filters)} ground-truth filters to {OUTPUT_FILE}")
