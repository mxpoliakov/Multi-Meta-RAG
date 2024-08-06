"""Mirrors code in MultiHop-RAG/qa_evaluate.py"""
import glob
import importlib
import json
import os
import sys

os.environ["TQDM_DISABLE"] = "1"
sys.stdout = open(os.devnull, 'w')
multihop_rag_qa_evaluate = importlib.import_module("MultiHop-RAG.qa_evaluate")
sys.stdout = sys.__stdout__

output_path = "qa_output"

print(f"Evaluate files in folder: {output_path}")
for filename in glob.glob(os.path.join(output_path, "*.json")):
    with open(filename, "r") as file:
        doc_data = json.load(file)
    if not doc_data:
        continue
    print("-" * 100)
    print("Filename:", filename)
    overall_pred_list = []
    overall_gold_list = []
    type_data = {}
    # Main loop, iterate through document data
    for d in doc_data:
        model_answer = d["model_answer"]
        gold = d["gold_answer"]
        if gold:
            question_type = d["question_type"]
            if question_type not in type_data:
                type_data[question_type] = {"pred_list": [], "gold_list": []}
            type_data[question_type]["pred_list"].append(model_answer)
            type_data[question_type]["gold_list"].append(gold)
            overall_pred_list.append(model_answer)
            overall_gold_list.append(gold)
    for question_type, data in type_data.items():
        precision, recall, f1 = multihop_rag_qa_evaluate.calculate_metrics(
            data["pred_list"], data["gold_list"]
        )
        assert precision == recall == f1
        print(f"Question Type: {question_type}. Accuracy: {precision:.4f}")

    # Calculate overall evaluation metrics
    overall_precision, overall_recall, overall_f1 = (
        multihop_rag_qa_evaluate.calculate_metrics(overall_pred_list, overall_gold_list)
    )
    assert overall_precision == overall_recall == overall_f1
    print(f"Overall Accuracy: {overall_precision:.4f}")
