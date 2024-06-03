from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector

from langchain.schema.document import Document
from llama_index.extractors import BaseExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import MetadataMode
import importlib
from retry import retry
from langchain_voyageai import VoyageAIEmbeddings

USE_VOYAGE_EMBEDDINGS = False

device = "cuda"
batch_size = 500

multihop_rag_util = importlib.import_module("MultiHop-RAG.util")


@retry(tries=5)
def add_embeddings_with_retry(vector_index, embeddings_batch, batch):
    vector_index.add_embeddings(
        [b.page_content for b in batch], embeddings_batch, [b.metadata for b in batch]
    )


class CustomExtractor(BaseExtractor):
    async def aextract(self, nodes):
        metadata_list = [
            {
                "title": (node.metadata["title"]),
                "source": (node.metadata["source"]),
                "published_at": (node.metadata["published_at"]),
            }
            for node in nodes
        ]
        return metadata_list


reader = multihop_rag_util.JSONReader()
data = reader.load_data("MultiHop-RAG/dataset/corpus.json")

text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)

transformations = [text_splitter, CustomExtractor()]
pipeline = IngestionPipeline(transformations=transformations)
nodes = pipeline.run(documents=data)

documents = []
for node in nodes:
    document = Document(
        page_content=node.get_content(metadata_mode=MetadataMode.LLM),
        metadata={
            "published_at": datetime.fromisoformat(
                node.metadata["published_at"]
            ).strftime("%B %-d, %Y"),
            "source": node.metadata["source"],
            "title": node.metadata["title"],
        },
    )
    documents.append(document)


print(documents[0])
print(len(documents))

if USE_VOYAGE_EMBEDDINGS:
    embeddings = VoyageAIEmbeddings(model="voyage-2")
else:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": device},
        show_progress=True,
    )

vector_index = Neo4jVector(embeddings)

for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    embeddings_batch = embeddings.embed_documents([b.page_content for b in batch])
    add_embeddings_with_retry(vector_index, embeddings_batch, batch)

vector_index.create_new_index()
vector_index._driver.close()
