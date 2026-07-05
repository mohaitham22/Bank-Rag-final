from rank_bm25 import BM25Okapi
from vectordb import retrieve_context


def load_all_chunks(collection):
    """
    Pulls every stored chunk (text + metadata) back out of ChromaDB,
    so a keyword search index can be built from the same data that
    was already embedded and stored — no re-scraping or re-chunking needed.

    Args:
        collection (chromadb.Collection): The "banks" collection.

    Returns:
        tuple: (texts, metadatas)
    """
    data = collection.get(include=["documents", "metadatas"])
    return data["documents"], data["metadatas"]


def build_bm25_index(texts):
    """
    Builds a BM25 keyword search index from the given texts.

    Args:
        texts (list): List of chunk text strings.

    Returns:
        BM25Okapi: The BM25 index, used for sparse (keyword) retrieval.
    """
    tokenized = [t.lower().split() for t in texts]
    return BM25Okapi(tokenized)


def bm25_search(query, bm25, texts, metadatas, top_k=3):
    """
    Runs a BM25 keyword search and returns the top_k matches.

    Args:
        query (str): The user's question.
        bm25 (BM25Okapi): Index built by build_bm25_index().
        texts (list): The same texts the index was built from.
        metadatas (list): Metadata aligned with texts.
        top_k (int): How many results to return.

    Returns:
        list: Dicts with "text", "metadata", "score", "rank".
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {"text": texts[i], "metadata": metadatas[i], "score": scores[i], "rank": rank + 1}
        for rank, i in enumerate(ranked)
    ]


def dense_search(query, collection, top_k=3):
    """
    Runs a dense (embedding-based) search and returns results in the same
    shape as bm25_search(), so the two can be fused together.

    Args:
        query (str): The user's question.
        collection (chromadb.Collection): The "banks" collection.
        top_k (int): How many results to return.

    Returns:
        list: Dicts with "text", "metadata", "score", "rank".
    """
    results = retrieve_context(query, collection, top_k=top_k)

    return [
        {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": results["distances"][0][i],
            "rank": i + 1
        }
        for i in range(len(results["documents"][0]))
    ]


def reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=3):
    """
    Combines dense and sparse (BM25) results using Reciprocal Rank Fusion (RRF).

    Each result contributes 1 / (k + rank) to a combined score. A chunk near
    the top of either list gets boosted; a chunk found near the top of BOTH
    lists gets boosted even more. This avoids having to compare embedding
    distances and BM25 scores directly, since they're on different scales.

    Args:
        dense_results (list): Output of dense_search().
        sparse_results (list): Output of bm25_search().
        k (int): RRF constant controlling how much rank position matters.
        top_k (int): Number of fused results to return.

    Returns:
        list: Dicts with "text", "metadata", "fused_score", sorted best first.
    """
    scores = {}
    lookup = {}

    for result_list in [dense_results, sparse_results]:
        for r in result_list:
            key = r["text"]
            lookup[key] = r
            scores[key] = scores.get(key, 0) + 1 / (k + r["rank"])

    ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)[:top_k]

    return [
        {"text": key, "metadata": lookup[key]["metadata"], "fused_score": scores[key]}
        for key in ranked_keys
    ]


def hybrid_search(query, collection, bm25, texts, metadatas, top_k=3):
    """
    Runs both dense and BM25 search, then fuses the results with RRF.

    Args:
        query (str): The user's question.
        collection (chromadb.Collection): The "banks" collection.
        bm25 (BM25Okapi): Index built by build_bm25_index().
        texts (list): Texts the BM25 index was built from.
        metadatas (list): Metadata aligned with texts.
        top_k (int): Number of fused results to return.

    Returns:
        list: Fused top_k results with "text", "metadata", "fused_score".
    """
    dense_results = dense_search(query, collection, top_k=top_k)
    sparse_results = bm25_search(query, bm25, texts, metadatas, top_k=top_k)
    return reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)
