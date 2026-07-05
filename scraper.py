import os
import json
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

RAW_HTML_DIR = "data/raw_html"
JSON_DIR = "data/json_data"

os.makedirs(RAW_HTML_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)


def clean(text):
    if not text:
        return ""
    return " ".join(text.split())


def extract_json_ld(soup):
    data = []

    for script in soup.find_all("script", type="application/ld+json"):

        if not script.string:
            continue

        try:
            obj = json.loads(script.string)
            data.append(obj)
        except Exception:
            pass

    return data


def extract_data(soup, url):

    bank = {
        "url": url,
        "title": "",
        "bank_name": "",
        "description": "",
        "country": "",
        "website": "",
        "faq": [],
        "about": "",
        "ratings": {},
        "all_text": ""
    }

    if soup.title:
        bank["title"] = clean(soup.title.text)

    h1 = soup.find("h1")
    if h1:
        bank["bank_name"] = clean(h1.text)

    meta = soup.find("meta", attrs={"name": "description"})
    if meta:
        bank["description"] = meta.get("content", "")

    json_blocks = extract_json_ld(soup)

    for block in json_blocks:

        if block.get("@type") == "FAQPage":

            for item in block.get("mainEntity", []):

                bank["faq"].append({
                    "question": clean(item.get("name", "")),
                    "answer": clean(item.get("acceptedAnswer", {}).get("text", ""))
                })

    country = soup.find("a", href=lambda x: x and "/country/" in x)

    if country:
        bank["country"] = clean(country.text)

    website = soup.find("a", href=lambda x: x and x.startswith("http"))

    if website:
        bank["website"] = website["href"]

    about_heading = soup.find(
        lambda tag:
        tag.name in ["h2", "h3"] and
        "About" in tag.get_text()
    )

    if about_heading:

        text = []

        node = about_heading.find_next_sibling()

        while node:

            if node.name in ["h2", "h3"]:
                break

            text.append(node.get_text(" ", strip=True))

            node = node.find_next_sibling()

        bank["about"] = clean(" ".join(text))

    scores = soup.find(id="scores")

    if scores:
        bank["ratings"]["raw"] = scores.get_text("\n", strip=True)

    body = soup.find("main")

    if body:
        bank["all_text"] = clean(body.get_text(" ", strip=True))

    return bank


with open("data/bank_urls.txt") as f:
    urls = [x.strip() for x in f if x.strip()]

for url in urls:

    print(url)

    html = requests.get(url, headers=HEADERS).text

    slug = url.split("/")[-1]

    with open(f"{RAW_HTML_DIR}/{slug}.html", "w", encoding="utf8") as f:
        f.write(html)

    soup = BeautifulSoup(html, "html.parser")

    bank = extract_data(soup, url)

    with open(f"{JSON_DIR}/{slug}.json", "w", encoding="utf8") as f:
        json.dump(bank, f, indent=4, ensure_ascii=False)