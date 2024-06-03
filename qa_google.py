import json
from tqdm import tqdm

import vertexai

from vertexai.language_models import TextGenerationModel

vertexai.init()

model = TextGenerationModel.from_pretrained("text-bison@001")
k_docs = 6
data_file = "output/voyage-02_256_32_with_filtering.json"
save_file = "qa_output/google-palm-voyage-02-filtering.json"

prefix = "You will be provided with questions followed by some context from different sources. Please answer the question based on the context. The answer to the question is a word or entity. If the provided information is insufficient to answer the question, respond 'Insufficient Information'. Answer directly without explanation."

with open(data_file, "r") as file:
    doc_data = json.load(file)


def predict(prompt):
    response = model.predict(prompt, temperature=0.1, max_output_tokens=12)
    return response.text


save_list = []

for d in tqdm(doc_data):
    retrieval_list = d["retrieval_list"][:k_docs]
    context = "--------------".join(e["text"] for e in retrieval_list)
    prompt = f"{prefix}\n\nQuestion:{d['query']}\n\nContext:\n\n{context}"
    response = predict(prompt)
    save = {}
    save["query"] = d["query"]
    save["prompt"] = prompt
    save["model_answer"] = response
    save["gold_answer"] = d["answer"]
    save["question_type"] = d["question_type"]
    save_list.append(save)

    with open(save_file, "w") as f:
        json.dump(save_list, f, indent=4, sort_keys=True)
