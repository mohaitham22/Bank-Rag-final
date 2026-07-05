# Bank RAG Assistant — Written Report

Route Graduation Project — AI Engineering Assessment

---

## 1. Dataset Description

>> TODO: Fill in the following based on your actual scraping:
- **Website/source used:** [name of the site you scraped bank pages from]
- **Source URLs:** [list them, or point to `data/bank_urls.txt`]
- **Type of data collected:** HTML pages containing bank name, description,
  about section, FAQ (via JSON-LD structured data), country, website link,
  and rating/score information.
- **Number of pages/records collected:** [e.g. "62 bank profile pages"]
- **Final dataset format:** Raw HTML saved in `data/raw_html/`, cleaned and
  structured into one JSON file per bank in `data/json_data/`, with fields:
  `url`, `title`, `bank_name`, `description`, `country`, `website`, `faq`,
  `about`, `ratings`, `all_text`.

## 2. Data Collection Method

The dataset was collected using a custom Python scraper (`scraper.py`) built
with `requests` and `BeautifulSoup`.

**Tools/libraries used:** `requests` for HTTP requests, `beautifulsoup4` for
HTML parsing.

**Method:**
- A list of target URLs was read from `data/bank_urls.txt`.
- Each page's raw HTML was fetched and saved to `data/raw_html/` for reproducibility.
- Structured fields were then extracted from each page:
  - Page title and meta description
  - Bank name from the page's `<h1>`
  - FAQ content parsed from embedded `application/ld+json` (`FAQPage`) blocks
  - Country, extracted from links matching a `/country/` URL pattern
  - The bank's official website, taken from the first outbound link
  - An "About" section, collected by walking sibling elements following an
    `<h2>`/`<h3>` heading containing the word "About"
  - Any ratings/scores block (`id="scores"`), stored as raw text
  - All visible text inside the page's `<main>` tag, as a fallback full-text field

**Cleaning steps:** All extracted text was whitespace-normalized (collapsing
repeated spaces/newlines) before being stored. Each page's structured result
was saved as its own JSON file, so any failed or partial extraction is
isolated to a single record rather than corrupting the whole dataset.

>> TODO: **Challenges faced during data collection:** [e.g. inconsistent page
layouts across banks, missing FAQ blocks on some pages, rate limiting, pages
without an "About" heading, etc. — describe what you actually ran into.]

## 3. Chunking Strategy

**Chosen method: Semantic chunking based on embedding distance between sentences.**

Each document (a bank's description, about section, or a single FAQ entry)
is first split into sentences. Consecutive sentences are embedded and the
cosine distance between each adjacent pair is measured. A chunk boundary is
placed wherever the distance jumps above the 90th percentile of all observed
gaps in that document — i.e. wherever the topic shifts noticeably — as long
as the chunk built so far has reached a minimum size (250 characters). A hard
maximum of 1800 characters is also enforced so no chunk becomes too large for
retrieval, regardless of whether a natural meaning-break was found.

**Why this strategy was chosen:** The dataset is not uniform — a bank
`description` might be a single short paragraph, while an `about` section or
`all_text` field can span several unrelated topics (history, services,
branches, leadership) with no consistent structure like headings or bullet
points. A fixed-size chunker (e.g. every 500 characters) risks cutting a
sentence — or a fact — in half. Semantic chunking instead tries to respect
where the *meaning* actually changes.

**What was observed in the data that influenced the decision:** Bank
`about`/`all_text` fields varied a lot in length and structure page to page,
while FAQ entries were already naturally short, self-contained
question/answer pairs. This mix made a single fixed rule (like "always split
every N sentences") a poor fit — semantic chunking adapts per-document instead.

**Other strategies considered and rejected:**
- *Fixed character/token chunking:* simplest to implement, but risks
  splitting a sentence or a fact mid-way, especially in longer `about`/`all_text` fields.
- *One chunk per document field (no splitting at all):* simpler, but some
  `about`/`all_text` fields are long enough that important details near the
  end would get diluted in a single large embedding, hurting retrieval precision.
- *Paragraph-based splitting:* rejected because the scraped text was
  whitespace-normalized into a single line during cleaning, so paragraph
  boundaries were not preserved in the extracted data.

**Weaknesses of the chosen strategy:**
- It depends on an embedding model's notion of "similar meaning," which can
  occasionally miss a real topic shift or invent one where there isn't a strong shift.
- It requires embedding every sentence during chunking (not just every chunk),
  which is slightly more compute-intensive than fixed-size splitting.
- Very short documents (FAQ entries, single-sentence descriptions) skip the
  process entirely and are kept as one chunk, which is intentional but means
  the semantic logic doesn't apply uniformly across the whole dataset.

**How weaknesses were reduced:** A minimum chunk size (250 characters)
prevents the algorithm from creating too many tiny, low-context chunks from
minor distance fluctuations. A maximum chunk size (1800 characters) prevents
the opposite failure mode — one giant chunk if no strong semantic break is found.
Each chunk also keeps the original document's metadata (`bank`, `country`,
`url`, `website`, `type`), so retrieval can always trace a chunk back to its
exact source and bank, regardless of chunk boundaries.

## 4. Embedding and Vector Store

**Embedding model used:** `all-MiniLM-L6-v2` (via `sentence-transformers`).

**Why it was selected:**
- **Language support:** Strong general-purpose English performance, which
  matches the dataset (English-language bank pages).
- **Accuracy:** A well-established, widely benchmarked model for semantic
  similarity and retrieval tasks relative to its size.
- **Speed:** One of the fastest sentence-transformer models available,
  important since chunking itself requires embedding every sentence in the
  dataset, not just every chunk.
- **Cost:** Runs locally, fully free, no API cost per embedding call.
- **Embedding dimension:** 384 dimensions — compact, keeping the vector
  database small and queries fast, which is a good fit for this project's
  dataset size.
- **Suitability for the dataset:** Bank profile text (descriptions, FAQs,
  short "about" paragraphs) is short-to-medium length, plain English text —
  exactly the kind of input this model was designed and benchmarked on.

**Vector store used:** ChromaDB (`PersistentClient`, stored in `chroma_db/`).

**Why it was selected:**
- Simple, embedded (no separate server to run), and persists to disk between runs.
- Native support for storing metadata alongside each chunk and filtering
  search results using that metadata (`where` filters), which the project's
  Level 2 filtered retrieval depends on.
- Straightforward Python API that keeps the codebase simple, without needing
  a heavier framework.

## 5. Gemini API Usage

**Where Gemini was used:**
- Final answer generation (`llm.py`)
- Query rewriting (`query_intelligence.py`)
- Query classification (`query_intelligence.py`)
- Structured filter extraction (`query_intelligence.py`)

**Prompt design:** The generation prompt (`prompt.txt`) explicitly instructs
the model to act as a banking information assistant, to answer *only* using
the retrieved context, to use the bank's name when relevant, and to respond
with a fixed fallback sentence ("I could not find the answer in the
document.") when the answer isn't present in the retrieved chunks.

**Grounding strategy:** Only the chunks retrieved for a given query
(top-k from either dense or hybrid search) are passed into the prompt as
context. The model is never given the full dataset — only what was retrieved
for that specific question — which keeps the answer traceable back to
specific sources.

**How hallucination was reduced:**
- The prompt explicitly forbids using outside knowledge.
- A required fallback phrase is defined for when context is insufficient,
  giving the model an explicit "safe" way out instead of guessing.
- The Streamlit GUI always displays the exact retrieved chunks and their
  metadata alongside the answer, so any ungrounded claim is easy to catch by inspection.

## 6. Query Intelligence

**Query rewriting approach:** Before classification or retrieval, the raw
user question is passed to Gemini with an instruction to rewrite it to be
clearer, more specific, and in English — without changing its meaning.

**Query classification approach:** The rewritten query is classified by
Gemini into exactly one of five categories tailored to the bank domain:

| Category | Meaning |
|---|---|
| `factual_lookup` | Asking for a specific fact about one bank |
| `comparison` | Comparing two or more banks |
| `location` | Asking about a bank's country/location |
| `faq` | A typical FAQ-style question |
| `out_of_scope` | Not related to banks at all |

**Filter extraction approach:** Gemini is prompted to return a strict JSON
object with `bank`, `country`, and `type` fields (each `null` if not implied
by the question). This is converted into a ChromaDB `where` filter, so
detected filters narrow the search to only matching metadata.

**Examples for each query category:**

>> TODO: Run a few real questions through `app.py` or `query_intelligence.py`
directly and paste one real example (original query -> rewritten query ->
class -> filters) for each of the 5 categories above. Example format:

```
Category: location
Original: "where does adib operate"
Rewritten: "What countries does ADIB (Abu Dhabi Islamic Bank) operate in?"
Filters: {"bank": "ADIB", "country": null, "type": null}
```

## 7. Hybrid Search

**Explanation of hybrid search:** Dense retrieval (embeddings) is strong at
understanding meaning and paraphrasing, but can miss exact terms (a specific
bank name, a rare word, an acronym) if they weren't well represented during
training. Sparse retrieval (keyword-based) is strong at exact term matching,
but can't understand that two different phrasings mean the same thing.
Hybrid search runs both and combines their results, aiming to get the best of both.

**Dense retrieval method:** ChromaDB similarity search over `all-MiniLM-L6-v2`
sentence embeddings.

**Sparse retrieval method:** BM25 (`rank-bm25`), built directly from the same
chunk texts already stored in ChromaDB (pulled back out via `collection.get()`),
so no separate index needs to be kept in sync.

**Fusion method:** Reciprocal Rank Fusion (RRF). Each chunk's final score is
`1 / (k + rank)` summed across both result lists (with `k = 60`), so a chunk
ranked highly in either list is boosted, and a chunk ranked highly in *both*
lists is boosted the most. RRF was chosen over a weighted-score fusion because
dense distance and BM25 scores are on incompatible numeric scales — RRF avoids
needing to normalize or tune a weight between them.

**Comparison between dense-only and hybrid search:**

>> TODO: Run `python compare_search.py` (edit the `QUERIES` list at the top
to real questions about your dataset first) and paste the actual output here
for at least 3 queries. For each one, note:
- Which chunks came back for dense-only vs. hybrid
- Which one you judged as better, and why (e.g. hybrid surfaced a chunk
  containing an exact bank name/term that dense-only ranked lower or missed)

## 8. Streamlit GUI

>> TODO: Add 2-3 screenshots of `app.py` in use here (question asked,
retrieved chunks expanded, final answer shown).

**Explanation of interface components:**
- A search mode toggle (Dense only / Hybrid) to demonstrate both retrieval methods
- A text input and "Ask" button for the user's question
- A "Query Understanding" panel showing the original query, rewritten query,
  detected query class, and extracted filters
- Expandable cards for each retrieved chunk, showing its similarity/fused
  score and full metadata
- The final Gemini-generated answer
- A de-duplicated list of sources used

**How the GUI connects to the pipeline:** `app.py` imports directly from the
same modules used everywhere else in the project (`vectordb.py`,
`hybrid_search.py`, `query_intelligence.py`, `llm.py`) — it does not
reimplement any retrieval or generation logic itself, it only calls it and
displays the results.

## 9. Limitations

>> TODO: Review and adjust these based on what you actually observed:
- **Small dataset size:** [fill in your actual number of banks/chunks — currently ~354 chunks]
- **Website structure inconsistency:** Not every bank page had all fields
  (some lacked an "About" section or FAQ block), so some chunks are sparser than others.
- **Metadata limitations:** Country and website extraction rely on simple
  link-pattern heuristics (`/country/` in the URL, first outbound link), which
  may occasionally mislabel a page.
- **Retrieval mistakes:** Both dense and hybrid search can still return
  loosely related chunks for ambiguous or very short questions.
- **Gemini limitations:** Filter extraction occasionally returns a bank name
  spelled slightly differently than it appears in the stored metadata, which
  can cause a metadata filter to return zero results even though a relevant
  chunk exists.
- **Query ambiguity:** Very short or vague questions (e.g. "tell me about it")
  are hard to classify and rewrite meaningfully without more conversation context.

## 10. Future Work

- Expand scraping coverage to more banks and additional page types (e.g. product/loan pages)
- Improve metadata extraction reliability (e.g. a proper country list instead of URL-pattern matching)
- Experiment with different chunk sizes/thresholds and measure retrieval accuracy directly
- Tune BM25/dense fusion weighting instead of using unweighted RRF
- Build a larger, labeled set of evaluation queries to measure retrieval and answer quality systematically
- Add authentication and persistent chat history if the tool moves beyond a demo
