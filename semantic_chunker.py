import re
import numpy as np


def split_sentences(text):
    """
    Splits text into a list of clean sentences.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def cosine_distance(a, b):
    """
    Returns the cosine distance between two vectors.
    0 means identical meaning, 1 means unrelated meaning.
    """
    sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    return 1.0 - sim


def semantic_split_documents(
    documents,
    embedding_model,
    threshold_percentile=90,
    min_chars=250,
    max_chars=1800,
):

    new_docs = []

    for doc in documents:

        sentences = split_sentences(doc["text"])

        if len(sentences) <= 1:
            new_docs.append(doc)
            continue

        vectors = embedding_model.encode(sentences)

        distances = [
            cosine_distance(vectors[i], vectors[i + 1])
            for i in range(len(sentences) - 1)
        ]

        threshold = float(np.percentile(distances, threshold_percentile))

        chunks = []
        current = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):

            sentence = sentences[i]
            gap = distances[i - 1]
            new_len = current_len + 1 + len(sentence)

            is_breakpoint = gap > threshold and current_len >= min_chars

            if is_breakpoint or new_len > max_chars:
                chunks.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len = new_len

        if current:
            chunks.append(" ".join(current))

        for chunk in chunks:
            new_docs.append({"text": chunk, "metadata": dict(doc["metadata"])})

    return new_docs
