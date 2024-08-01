import json
from openai import OpenAI
from retry import retry
from tqdm import tqdm

prefix = "Below is a question followed by some context from different sources. Please answer the question based on the context. The answer to the question is a word or entity. If the provided information is insufficient to answer the question, respond 'Insufficient Information'. Answer directly without explanation."
k_docs = 6

save_file = "qa_output/gpt-4-voyage-02-filtering.json"

with open("output/voyage-02_256_32_with_filtering.json", "r") as f:
    doc_data = json.load(f)

client = OpenAI()


@retry(tries=10, delay=1, backoff=2)
def predict(query):
    completion = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[{"role": "user", "content": query}],
        temperature=0.1,
    )
    response = completion.choices[0].message.content
    return response


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

    with open(save_file, "w") as query_filters_f:
        json.dump(save_list, query_filters_f, indent=4)
