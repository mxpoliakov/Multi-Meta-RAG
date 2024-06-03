import json
import string


def clean_answer(answer):
    return (
        answer.lower()
        .strip()
        .translate(str.maketrans(dict.fromkeys(string.punctuation)))
    )


with open("qa_output/voyage-gpt4.json", "r") as file:
    doc_data = json.load(file)

correct_answers = 0

for d in doc_data:
    if clean_answer(d["gold_answer"]) in clean_answer(
        d["model_answer"]
    ) or clean_answer(d["model_answer"]) in clean_answer(d["gold_answer"]):
        correct_answers += 1

print(correct_answers / len(doc_data))
