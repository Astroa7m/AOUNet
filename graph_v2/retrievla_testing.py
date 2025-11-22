import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore



embeddings = HuggingFaceEmbeddings()

vector_store = InMemoryVectorStore(embedding=embeddings)

retrieval = vector_store.load("uni_info", embeddings).as_retriever()


docs = retrieval.invoke("TM354 tutor")

for doc in docs:
    print(doc)
    print("?//////////////////////?")

