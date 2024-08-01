import glob
import json
import os
import string

output_path = "qa_output"


def clean_answer(answer):
    return (
        answer.lower()
        .strip()
        .translate(str.maketrans(dict.fromkeys(string.punctuation)))
    )


def get_accuracy(filename):
    with open(filename, "r") as file:
        doc_data = json.load(file)

    correct_answers = 0

    for d in doc_data:
        if clean_answer(d["gold_answer"]) in clean_answer(
            d["model_answer"]
        ) or clean_answer(d["model_answer"]) in clean_answer(d["gold_answer"]):
            correct_answers += 1

    return correct_answers / len(doc_data)


print(f"Evaluate files in folder: {output_path}")
for filename in glob.glob(os.path.join(output_path, "*.json")):
    print(f"For file: {filename} = {round(get_accuracy(filename), 4)}")
