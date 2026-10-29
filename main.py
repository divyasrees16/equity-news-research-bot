"""Streamlit UI for the Equity News Research Bot.  Run: streamlit run main.py"""
import streamlit as st

from research import Article, FetchError, SearchIndex, best_sentence, fetch_article

MAX_URLS = 10

st.set_page_config(page_title="Equity News Research Bot", page_icon="📈", layout="wide")
st.title("📈 Equity News Research Bot")
st.caption("Paste news article links, then ask questions. Answers are pulled from the articles and cited.")

# ---- Sidebar: input ---------------------------------------------------------
st.sidebar.header("News article URLs")
raw_urls = st.sidebar.text_area(
    f"One URL per line (max {MAX_URLS})",
    height=180,
    placeholder="https://example.com/news/article-1\nhttps://example.com/news/article-2",
)
process_clicked = st.sidebar.button("Process URLs", type="primary")

if process_clicked:
    urls = list(dict.fromkeys(u.strip() for u in raw_urls.splitlines() if u.strip()))[:MAX_URLS]
    if not urls:
        st.sidebar.warning("Please enter at least one URL.")
    else:
        articles: list[Article] = []
        with st.spinner("Fetching and indexing articles..."):
            for url in urls:
                try:
                    articles.append(fetch_article(url))
                except FetchError as exc:
                    st.sidebar.error(f"Could not load {url}\n\n{exc}")
        if articles:
            st.session_state["articles"] = articles
            st.session_state["index"] = SearchIndex(articles)
            st.sidebar.success(f"Loaded {len(articles)} article(s).")

# ---- Main: loaded articles --------------------------------------------------
articles = st.session_state.get("articles", [])
index = st.session_state.get("index")

if articles:
    with st.expander(f"Loaded articles ({len(articles)})"):
        for art in articles:
            st.markdown(f"- [{art.title}]({art.url}) — {art.word_count:,} words")

# ---- Main: Q&A --------------------------------------------------------------
query = st.text_input("Ask a question about the articles", placeholder="e.g. What did the company report for Q2 revenue?")
top_k = st.slider("Supporting passages to show", 1, 5, 3)

if query:
    if index is None:
        st.info("Add article URLs in the sidebar and click **Process URLs** first.")
    else:
        results = index.search(query, top_k=top_k)
        if not results:
            st.warning("No relevant passage found. Try rephrasing, or add more articles.")
        else:
            top = results[0]
            st.subheader("Answer")
            st.success(best_sentence(top.text, query))
            st.markdown(f"**Source:** [{top.title}]({top.source})  ·  relevance {top.score:.2f}")

            st.subheader("Supporting passages")
            for i, p in enumerate(results, 1):
                with st.expander(f"{i}. {p.title[:80]}  (relevance {p.score:.2f})"):
                    st.write(p.text)
                    st.markdown(f"[Open source]({p.source})")
