import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore



embeddings = HuggingFaceEmbeddings()

vector_store = InMemoryVectorStore(embedding=embeddings)

retrieval = vector_store.load("uni_info", embeddings).as_retriever()

query = input("query: ")

while query != "exit":
    answers = retrieval.invoke(query)
    for r in answers:
        print(r)
        print("/" * 300)
    query = input("query: ")
