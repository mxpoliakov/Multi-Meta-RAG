from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from retry import retry
from sentence_transformers import CrossEncoder

import json
from tqdm import tqdm

from langchain_voyageai import VoyageAIEmbeddings

device = "cuda"
OUTPUT = "output/voyage-02_256_32_with_filtering.json"
USE_VOYAGE_EMBEDDINGS = True


@retry(tries=10, delay=1, backoff=2)
def similarity_search_with_retry(vector_index, query, filter):
    return vector_index.similarity_search(query=query, k=20, filter=filter)


reranker_model = CrossEncoder(model_name="BAAI/bge-reranker-large", device=device)


def rerank_docs(query, retrieved_docs):
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)


if USE_VOYAGE_EMBEDDINGS:
    embeddings = VoyageAIEmbeddings(model="voyage-2")
else:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": device}
    )

with open("MultiHop-RAG/dataset/MultiHopRAG.json") as query_data_f:
    query_data_list = json.load(query_data_f)

with open("query_metadata_filters.json") as query_filters_f:
    query_filters_list = json.load(query_filters_f)


vector_index = Neo4jVector.from_existing_index(embeddings, index_name="vector")

retrieval_save_list = []
assert len(query_data_list) == len(query_filters_list)


for query_data, query_filters in tqdm(
    zip(query_data_list, query_filters_list), total=len(query_data_list)
):
    assert query_data["query"] == query_filters["query"]
    docs = similarity_search_with_retry(
        vector_index,
        query=query_data["query"],
        filter=query_filters["filter"],
    )

    if docs:
        rerank_docs_result = rerank_docs(query_data["query"], docs)
    else:
        rerank_docs_result = []

    retrieval_list = []
    for doc, score in rerank_docs_result[:10]:
        retrieval_list.append({"text": doc.page_content, "score": float(score)})
    save = {}
    save["query"] = query_data["query"]
    save["answer"] = query_data["answer"]
    save["question_type"] = query_data["question_type"]
    save["retrieval_list"] = retrieval_list
    save["gold_list"] = query_data["evidence_list"]
    retrieval_save_list.append(save)
    with open(OUTPUT, "w") as json_file:
        json.dump(retrieval_save_list, json_file, indent=4)

vector_index._driver.close()
