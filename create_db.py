from document_loader import load_documents
from semantic_chunker import semantic_split_documents
from embedding import load_model
from vectordb import build_vector_database

print("Loading documents...")
docs = load_documents()
print(len(docs))

embedding_model = load_model()

print("Semantic chunking...")
chunks = semantic_split_documents(docs, embedding_model)
print(len(chunks))

print("Building vector database...")
build_vector_database(chunks)
print("Database Created!")
