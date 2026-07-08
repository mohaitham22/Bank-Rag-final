# Bank RAG Assistant

An AI-powered question-answering assistant that answers natural-language
questions about a collected dataset of bank profile pages, using a
Retrieval-Augmented Generation (RAG) pipeline with hybrid search and
Google Gemini for answer generation.

Built for the Route Graduation Project (AI Engineering Assessment).

## Features

- Custom-scraped dataset of bank profile pages (description, about, FAQ, ratings, country)
- Semantic chunking based on meaning shifts between sentences
- Dense retrieval using sentence-transformer embeddings + ChromaDB
- Sparse (keyword) retrieval using BM25
- Hybrid search combining both via Reciprocal Rank Fusion
- Query intelligence: rewriting, classification, and structured filter extraction (via Gemini)
- Grounded answer generation using the Gemini API
- Streamlit GUI to test and demo the full pipeline

## Project Structure

```
.
├── data/
│   ├── raw_html/            # Raw scraped HTML pages
│   └── json_data/           # Cleaned, structured bank data (per page)
├── chroma_db/                # Persisted vector database (created by create_db.py)
├── scraper.py                 # Step 1: scrape bank pages -> data/json_data
├── document_loader.py         # Step 2: load JSON files into text + metadata
├── semantic_chunker.py        # Step 3: split text into semantic chunks
├── embedding.py                # Loads the sentence-transformer embedding model
├── vectordb.py                 # Builds/queries the ChromaDB vector store
├── create_db.py                # Runs steps 2-4: load -> chunk -> embed -> store
├── hybrid_search.py            # BM25 + dense fusion (hybrid search)
├── query_intelligence.py       # Query rewriting, classification, filter extraction
├── llm.py                       # Gemini API wrapper for grounded answer generation
├── prompt.txt                   # Prompt template used by llm.py
├── app.py                        # Streamlit GUI - the main way to use the project
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Clone the repository and create a virtual environment (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your Gemini API key:
   ```
   GENAI_API_KEY=your_key_here
   ```

## Running the Pipeline

Run these once, in order, to collect data and build the database:

```bash
python scraper.py       # Scrapes bank pages into data/json_data/
python create_db.py     # Loads, chunks, embeds, and stores everything in chroma_db/
```

Then launch the app:

```bash
streamlit run app.py
```

## Notes

- The vector database (`chroma_db/`) is created locally and is not committed to GitHub.
- Real API keys must never be committed - only `.env.example` is tracked.
- The core RAG pipeline (retrieval, chunking, hybrid search, Gemini prompts) was written
  by the student; AI tools were used to help generate the Streamlit GUI code.
