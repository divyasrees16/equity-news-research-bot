# 📈 Equity News Research Bot

An AI-assisted research tool that reads news articles you provide and answers your questions about them — with the **exact source** for every answer.

Paste a few article links, ask something like *"What was the Q2 revenue?"*, and the bot finds the most relevant passages across all articles, highlights the best sentence, and links back to where it came from.


## ✨ Features

- **Multi-article research** – load up to 10 article URLs at once and query them together
- **Passage-level retrieval** – articles are split into overlapping passages, so answers are precise rather than "the whole page"
- **Cited answers** – every answer shows its source article and a relevance score
- **Clean text extraction** – ignores navigation menus, scripts and footers
- **Robust fetching** – timeouts, friendly error messages for blocked/paywalled pages
- **Tested** – unit tests for parsing, chunking and retrieval

## 🧠 How it works

1. **Fetch** – downloads each page and keeps only the readable paragraph text (`requests` + `BeautifulSoup`).
2. **Chunk** – splits text into 3-sentence passages with 1-sentence overlap (`NLTK`).
3. **Index** – builds a TF-IDF matrix over all passages (`scikit-learn`).
4. **Retrieve** – ranks passages by cosine similarity to your question and picks the best-matching sentence.

## 🛠️ Tech stack

Python · Streamlit · scikit-learn · NLTK · BeautifulSoup · Requests

## 📁 Project structure

```
equity-news-research-bot/
├── main.py                 # Streamlit user interface
├── research.py             # Core logic: fetch, chunk, index, search
├── requirements.txt        # Runtime dependencies
└── .gitignore
```

## 🚀 Getting started

```bash
# 1. Clone
git clone https://github.com/divyasrees16/equity-news-research-bot.git
cd equity-news-research-bot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run main.py
```

Then open the local URL Streamlit prints (http://localhost:8501).

