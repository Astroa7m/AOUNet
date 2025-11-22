import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from graph_v2.loader import splits



embeddings = HuggingFaceEmbeddings()

# Store embeddings in a vector store (InMemoryVectorStore for dev, production DB for scale)
vector_store = InMemoryVectorStore(embedding=embeddings)

# Add all splits to the store
vector_store.add_documents(splits)
vector_store.dump("uni_info")