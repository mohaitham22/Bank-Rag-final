"""
Smoke test for the RAG pipeline.
Run this AFTER scraper.py has created data/json_data/ and AFTER
you've added GENAI_API_KEY to your .env file.

It tests each file one at a time, on a small sample, so you can see
exactly which piece breaks (if any) instead of one confusing error.
"""

import sys


def check(name, fn):
    print(f"\n--- Testing: {name} ---")
    try:
        result = fn()
        print(f"OK: {name}")
        return result
    except Exception as e:
        print(f"FAILED: {name}")
        print(f"Error: {e}")
        sys.exit(1)


# 1. document_loader.py
def test_document_loader():
    from document_loader import load_documents
    docs = load_documents()
    assert len(docs) > 0, "No documents loaded — check data/json_data/ has files"
    print(f"Loaded {len(docs)} documents")
    print("Sample:", docs[0]["text"][:100], "...")
    print("Sample metadata:", docs[0]["metadata"])
    return docs


docs = check("document_loader.py", test_document_loader)


# 2. embedding.py
def test_embedding():
    from embedding import load_model
    model = load_model()
    vec = model.encode(["test sentence"])
    assert len(vec[0]) > 0
    print("Embedding vector length:", len(vec[0]))
    return model


model = check("embedding.py", test_embedding)


# 3. semantic_chunker.py (use only 5 docs so this runs fast)
def test_chunker():
    from semantic_chunker import semantic_split_documents
    sample_docs = docs[:5]
    chunks = semantic_split_documents(sample_docs, model)
    assert len(chunks) > 0
    print(f"Produced {len(chunks)} chunks from {len(sample_docs)} sample documents")
    print("Sample chunk:", chunks[0]["text"][:100], "...")
    return chunks


chunks = check("semantic_chunker.py", test_chunker)


# 4. vectordb.py (builds + queries the real chroma_db — safe, gets overwritten
#    later when you run the full create_db.py on all your data)
def test_vectordb():
    from vectordb import build_vector_database, retrieve_context, format_context
    collection = build_vector_database(chunks)
    results = retrieve_context("What countries do these banks operate in?", collection, top_k=2)
    assert len(results["documents"][0]) > 0, "No chunks retrieved"
    print("Retrieved", len(results["documents"][0]), "chunks")
    print("Top match metadata:", results["metadatas"][0][0])
    print("Top match distance/score:", results["distances"][0][0])
    return format_context(results)


context = check("vectordb.py", test_vectordb)


# 5. llm.py + prompt.txt (needs GENAI_API_KEY in .env)
def test_llm():
    from llm import ask_gemini
    answer = ask_gemini("What countries do these banks operate in?", context)
    print("Gemini answer:", answer)
    return answer


check("llm.py + prompt.txt", test_llm)

print("\nAll steps passed. The full pipeline works end-to-end on a sample.")
print("Now run 'python create_db.py' to build the real database on all your data.")
