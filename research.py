"""Core logic for the Equity News Research Bot.

Kept separate from the Streamlit UI (main.py) so it can be tested and reused.
Pipeline: fetch article -> clean text -> split into passages -> TF-IDF index
-> retrieve the passages (and best sentence) that answer a question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import nltk
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EquityNewsResearchBot/1.0)"}
REQUEST_TIMEOUT = 15
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe"]


class FetchError(Exception):
    """Raised when an article cannot be downloaded or contains no usable text."""


@dataclass
class Article:
    url: str
    title: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Passage:
    text: str
    source: str
    title: str
    score: float = 0.0


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_nltk_ready = False


def _prepare_nltk() -> None:
    """Download tokenizer data once (newer NLTK versions need 'punkt_tab')."""
    global _nltk_ready
    if _nltk_ready:
        return
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:  # offline etc. - regex fallback below still works
            pass
    _nltk_ready = True


def split_sentences(text: str) -> list[str]:
    _prepare_nltk()
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 2]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, sentences_per_chunk: int = 3, overlap: int = 1) -> list[str]:
    """Split text into overlapping multi-sentence passages."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    step = max(1, sentences_per_chunk - overlap)
    chunks = []
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start:start + sentences_per_chunk])
        if len(chunk.split()) >= 8:  # drop tiny fragments (menus, captions)
            chunks.append(chunk)
        if start + sentences_per_chunk >= len(sentences):
            break
    return chunks


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def parse_article(url: str, html: bytes | str) -> Article:
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.get_text()) if soup.title else url
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    paragraphs = [clean_text(p.get_text()) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p.split()) >= 5]
    text = " ".join(paragraphs) if paragraphs else clean_text(soup.get_text())
    if len(text.split()) < 30:
        raise FetchError("Page has too little readable text (it may be paywalled or JavaScript-rendered).")
    return Article(url=url, title=title, text=text)


def fetch_article(url: str) -> Article:
    if not re.match(r"^https?://", url):
        raise FetchError("URL must start with http:// or https://")
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise FetchError(str(exc)) from exc
    return parse_article(url, response.content)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class SearchIndex:
    """TF-IDF index over passages from all loaded articles."""

    def __init__(self, articles: list[Article]):
        self.passages: list[Passage] = []
        for art in articles:
            for chunk in chunk_text(art.text):
                self.passages.append(Passage(text=chunk, source=art.url, title=art.title))
        if not self.passages:
            raise ValueError("No passages could be built from the supplied articles.")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([p.text for p in self.passages])

    def search(self, query: str, top_k: int = 3) -> list[Passage]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            Passage(self.passages[i].text, self.passages[i].source, self.passages[i].title, float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]


def best_sentence(passage: str, query: str) -> str:
    """Pick the single sentence in a passage that best matches the query."""
    sentences = split_sentences(passage)
    if len(sentences) <= 1:
        return passage
    try:
        vec = TfidfVectorizer(stop_words="english").fit(sentences + [query])
        sims = cosine_similarity(vec.transform([query]), vec.transform(sentences)).flatten()
    except ValueError:  # query made only of stop words
        return sentences[0]
    return sentences[int(sims.argmax())]
