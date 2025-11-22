from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Q&A JSON
qa_loader = JSONLoader(file_path="../data/json/general_aou_rag_data_cleaned.json", jq_schema=".[]", text_content=False)
qa_docs = qa_loader.load()

csv_loader_1 = CSVLoader(file_path="../data/csv/tutors.csv")
csv_loader_2 = CSVLoader(file_path="../data/csv/modules.csv")
csv_docs_1 = csv_loader_1.load()
csv_docs_2 = csv_loader_2.load()

from pathlib import Path
markdown_docs = []
for md_file in Path("../data/mds").glob("*.md"):
    with open(md_file, encoding="utf-8") as f:
        markdown_docs.append(Document(page_content=f.read(), metadata={"source": str(md_file)}))

all_docs = qa_docs + csv_docs_1 + csv_docs_2 + markdown_docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # characters per chunk
    chunk_overlap=200,        # overlap for context continuity
    add_start_index=True      # track original position
)

splits = text_splitter.split_documents(all_docs)
print(f"Created {len(splits)} chunks from {len(all_docs)} documents")