import chromadb
from hybrid_search import load_all_chunks, build_bm25_index, dense_search, hybrid_search

# Edit or add your own queries here - the assignment asks for at least 3
QUERIES = [
    "What countries does this bank operate in?",
    "customer service phone number",
    "does this bank offer Islamic banking",
]


def load_collection():
    """
    Loads the existing persisted ChromaDB collection (built by create_db.py).
    """
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection("banks")


def print_results(title, results, score_key):
    """
    Prints a ranked list of results with bank, type, score, and a text preview.
    """
    print(f"\n{title}")
    for i, r in enumerate(results):
        print(f"  {i + 1}. Bank: {r['metadata'].get('bank')} | Type: {r['metadata'].get('type')}")
        print(f"     {score_key}: {r[score_key]:.4f}")
        print(f"     Text: {r['text'][:120]}...")


def main():
    collection = load_collection()

    # Build the BM25 index once from the same chunks already stored in Chroma
    texts, metadatas = load_all_chunks(collection)
    bm25 = build_bm25_index(texts)

    for query in QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        dense_results = dense_search(query, collection, top_k=3)
        print_results("Dense-only results (lower score = closer match)", dense_results, "score")

        fused_results = hybrid_search(query, collection, bm25, texts, metadatas, top_k=3)
        print_results("Hybrid results (higher score = closer match)", fused_results, "fused_score")


if __name__ == "__main__":
    main()
