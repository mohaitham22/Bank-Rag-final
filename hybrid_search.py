from rank_bm25 import BM25Okapi
from vectordb import retrieve_context


def load_all_chunks(collection):
 
    data = collection.get(include=["documents", "metadatas"])
    return data["documents"], data["metadatas"]


def build_bm25_index(texts):

    tokenized = [t.lower().split() for t in texts]
    return BM25Okapi(tokenized)


def bm25_search(query, bm25, texts, metadatas, top_k=3):

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {"text": texts[i], "metadata": metadatas[i], "score": scores[i], "rank": rank + 1}
        for rank, i in enumerate(ranked)
    ]


def dense_search(query, collection, top_k=3):

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

    dense_results = dense_search(query, collection, top_k=top_k)
    sparse_results = bm25_search(query, bm25, texts, metadatas, top_k=top_k)
    return reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)
