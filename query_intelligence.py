import json
from llm import gemini_model


def rewrite_query(query):
    """
    Rewrites the user's question to be clearer and more specific for search.

    Args:
        query (str): The original user question.

    Returns:
        str: The rewritten question.
    """
    prompt = f"""Rewrite the following user question about banks to make it
clearer, more specific, and easier to search for. Keep it in English.
Only return the rewritten question, nothing else.

Original question: {query}

Rewritten question:"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()


def classify_query(query):
    """
    Classifies the question into one of the bank domain categories.

    Args:
        query (str): The user question (ideally the rewritten one).

    Returns:
        str: One of "factual_lookup", "comparison", "location", "faq", "out_of_scope".
    """
    prompt = f"""Classify the following question about banks into exactly
one of these categories:

- factual_lookup: asking for a specific fact about one bank (description, about, contact info, ratings)
- comparison: comparing two or more banks
- location: asking about a bank's country or location
- faq: asking something in a typical FAQ style (how do I, what is, can I)
- out_of_scope: not related to banks at all

Question: {query}

Return only the category name, nothing else.

Category:"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip().lower()


def extract_filters(query):
    """
    Extracts structured search filters implied by the question.

    Args:
        query (str): The user question (ideally the rewritten one).

    Returns:
        dict: Keys "bank", "country", "type" - each a string or None.
    """
    prompt = f"""Extract search filters from the following question about banks.
Return ONLY a valid JSON object with these exact keys: "bank", "country", "type".
Use null for any filter that isn't mentioned or clearly implied.
"type" must be one of: "description", "about", "faq", or null.

Question: {query}

JSON:"""

    response = gemini_model.generate_content(prompt)
    text = response.text.strip()

    # Gemini sometimes wraps JSON in ```json fences - strip them if present
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"bank": None, "country": None, "type": None}


def build_where_filter(filters):
    """
    Converts extracted filters into a ChromaDB metadata "where" filter,
    skipping any filter that wasn't detected.

    Args:
        filters (dict): Output of extract_filters().

    Returns:
        dict or None: A ChromaDB-compatible where filter, or None if empty.
    """
    where = {}

    for key in ["bank", "country", "type"]:
        value = filters.get(key)
        if value:
            where[key] = value

    if not where:
        return None

    if len(where) == 1:
        return where

    return {"$and": [{k: v} for k, v in where.items()]}


def process_query(query):
    """
    Runs the full query intelligence pipeline: rewrite -> classify -> extract filters.

    Args:
        query (str): The original user question.

    Returns:
        dict: {
            "original_query": str,
            "rewritten_query": str,
            "query_class": str,
            "filters": dict,
            "where": dict or None
        }
    """
    rewritten = rewrite_query(query)
    query_class = classify_query(rewritten)
    filters = extract_filters(rewritten)
    where = build_where_filter(filters)

    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "query_class": query_class,
        "filters": filters,
        "where": where
    }
