import chromadb
from embedding import load_model

embedding_model = load_model()


def build_vector_database(chunks):
    """
    Builds a persistent vector database using ChromaDB.

    Args:
        chunks (list): List of dicts with "text" and "metadata" keys.

    Returns:
        chromadb.Collection: The collection storing the chunk embeddings.
    """
    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection("banks")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    embeddings = embedding_model.encode(texts)

    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return collection


def retrieve_context(query, collection, top_k=3, where=None):
    """
    Retrieves the most relevant chunks from the vector database for a query.

    Args:
        query (str): The user's question.
        collection (chromadb.Collection): The ChromaDB collection to search in.
        top_k (int): The number of top results to retrieve.
        where (dict, optional): A ChromaDB metadata filter, e.g. {"country": "Egypt"}.
            Pass None (default) to search with no filter.

    Returns:
        dict: Raw ChromaDB results, including documents, metadatas, and distances.
    """
    query_embedding = embedding_model.encode([query])

    query_kwargs = {
        "query_embeddings": query_embedding.tolist(),
        "n_results": top_k
    }

    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)
    return results


def format_context(results):
    """
    Joins the retrieved chunk texts into one context string for the LLM.

    Args:
        results (dict): The output of retrieve_context().

    Returns:
        str: The combined context text.
    """
    return "\n\n".join(results["documents"][0])
