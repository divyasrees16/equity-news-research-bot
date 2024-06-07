import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk

# Download NLTK punkt tokenizer data
nltk.download('punkt')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #2c3e50;
    }
    .sidebar-title {
        color: #2980b9;
    }
    .text {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
    }
    .stTextInput>div>div>input {
        border: 1px solid #2980b9;
    }
    .sidebar .sidebar-content {
        background-color: #f1f1f1;
    }
    .main-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main-title">Equity News Research Tool</h1>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 class="sidebar-title">News Article URLs</h2>', unsafe_allow_html=True)

# Function to fetch and parse URL content
def fetch_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}")
        return None

# Function to extract key points from a document
def extract_key_points(text, num_points=5):
    sentences = sent_tokenize(text)
    return sentences[:num_points]

# Function to find the best sentence matching the query
def find_best_sentence(context, query):
    sentences = sent_tokenize(context)
    vectorizer = TfidfVectorizer().fit(sentences)
    query_vec = vectorizer.transform([query])
    sentence_vecs = vectorizer.transform(sentences)
    similarities = cosine_similarity(query_vec, sentence_vecs).flatten()
    best_index = similarities.argmax()
    return sentences[best_index]

# Get URLs from user input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vector_store.pkl"

main_placeholder = st.empty()

if process_url_clicked and urls:
    main_placeholder.text("Data Loading...Started...✅✅✅")

    # Fetch and create documents
    data = []
    for url in urls:
        content = fetch_url_content(url)
        if content:
            data.append({"content": content, "source": url})
            st.write(f"Fetched content from {url}")
        else:
            st.write(f"Failed to fetch content from {url}")

    st.write(f"Loaded {len(data)} documents.")

    # Convert documents to TF-IDF vectors
    main_placeholder.text("Creating TF-IDF vectors...Started...✅✅✅")
    documents = [doc["content"] for doc in data]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    # Save the vector store to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump((X, data, vectorizer), f)
    main_placeholder.text("Data Processing Complete. You can now ask questions! 🎉")

# Query input and retrieval
query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            X, data, vectorizer = pickle.load(f)
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, X).flatten()
            top_index = similarities.argmax()  # Get the top matching document

            st.header("Answer")
            context = data[top_index]['content']
            source = data[top_index]['source']
            
            # Extract the answer
            answer = find_best_sentence(context, query)
            
            # Display the answer and key points
            st.write(f"**Source:** {source}")
            st.write(f"**Answer:** {answer}")
    else:
        st.error("Vector store not found. Please process URLs first.")
