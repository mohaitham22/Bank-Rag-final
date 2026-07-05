import os
import json

DATA_DIR = "data/json_data"


def load_documents():
    """
    Loads bank JSON files and turns them into plain text documents with metadata.

    Returns:
        list: A list of dicts, each with "text" and "metadata" keys.
    """
    docs = []

    for file in os.listdir(DATA_DIR):

        if not file.endswith(".json"):
            continue

        with open(os.path.join(DATA_DIR, file), encoding="utf8") as f:
            bank = json.load(f)

        metadata = {
            "bank": bank["bank_name"],
            "country": bank["country"],
            "url": bank["url"],
            "website": bank["website"]
        }

        if bank["description"]:
            docs.append({
                "text": f"Bank Name: {bank['bank_name']}\n\nDescription:\n\n{bank['description']}",
                "metadata": {**metadata, "type": "description"}
            })

        if bank["about"]:
            docs.append({
                "text": f"Bank Name: {bank['bank_name']}\n\nAbout:\n\n{bank['about']}",
                "metadata": {**metadata, "type": "about"}
            })

        for faq in bank["faq"]:
            docs.append({
                "text": f"Bank Name: {bank['bank_name']}\n\nQuestion:\n{faq['question']}\n\nAnswer:\n{faq['answer']}",
                "metadata": {**metadata, "type": "faq", "question": faq["question"]}
            })

    return docs
