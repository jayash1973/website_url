import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from together import Together
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException
from googletrans import Translator
import plotly.express as px
import networkx as nx
from collections import Counter
import re
import base64
from io import BytesIO
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Set up Together API client
together_api_key = ""
client = Together(api_key=together_api_key)

# Initialize sentiment analyzer, translator, and other tools
sia = SentimentIntensityAnalyzer()
translator = Translator()

# Set page config
st.set_page_config(page_title="Advanced Website Analyzer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to improve UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #808080;
        border-radius: 5px 5px 0px 0px;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.text for p in soup.find_all('p')])
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")
        return None

@st.cache_data
def get_summary(text):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Please provide a concise summary of the following text:\n\n{text[:4000]}"}
        ],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content

def chat_with_ai(question, context):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
            {"role": "user", "content": f"Context: {context[:4000]}\n\nQuestion: {question}"}
        ],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content

@st.cache_data
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

@st.cache_data
def perform_sentiment_analysis(text):
    sentences = nltk.sent_tokenize(text)
    sentiments = [sia.polarity_scores(sentence)['compound'] for sentence in sentences]
    return pd.DataFrame({'sentence': sentences, 'sentiment': sentiments})

@st.cache_data
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "Unknown"

@st.cache_data
def translate_text(text, target='en'):
    try:
        return translator.translate(text, dest=target).text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

@st.cache_data
def generate_topic_graph(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_pairs = list(zip(words[:-1], words[1:]))
    word_counts = Counter(word_pairs)
    
    G = nx.Graph()
    for (word1, word2), count in word_counts.most_common(50):
        G.add_edge(word1, word2, weight=count)
    
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = px.line(x=edge_x, y=edge_y).data[0]
    edge_trace.line.width = 0.5
    edge_trace.line.color = '#888'

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = px.scatter(x=node_x, y=node_y, text=list(G.nodes())).data[0]
    node_trace.marker.size = 10
    node_trace.marker.color = '#007bff'
    node_trace.text = list(G.nodes())
    node_trace.textposition = 'top center'

    fig = px.scatter(x=node_x, y=node_y, text=list(G.nodes()))
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    fig.update_layout(showlegend=False, title="Topic Relationship Graph")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv" class="streamlit-button">Download CSV file</a>'
    return href

@st.cache_data
def get_summary(text, max_tokens=2024):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Please provide a concise summary of the following text:\n\n{text[:4000]}"}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content

async def crawl_website(url, max_pages=10):
    visited = set()
    to_visit = [url]
    results = []

    async with aiohttp.ClientSession() as session:
        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue

            try:
                async with session.get(current_url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        text = ' '.join([p.text for p in soup.find_all('p')])
                        summary = get_summary(text)
                        results.append({'url': current_url, 'text': text, 'summary': summary})
                        visited.add(current_url)

                        for link in soup.find_all('a', href=True):
                            new_url = urljoin(current_url, link['href'])
                            if urlparse(new_url).netloc == urlparse(url).netloc and new_url not in visited:
                                to_visit.append(new_url)
            except Exception as e:
                st.warning(f"Error crawling {current_url}: {e}")

    return results

def main():
    st.title("🌐 Advanced Website Analyzer")
    st.write("Enter a website URL to analyze its content, get insights, and interact with AI!")

    url = st.text_input("Enter website URL:")
    
    if url:
        tabs = st.tabs(["Summary", "Chat", "Statistics", "Sentiment", "Language", "Topics", "Web Crawler"])
        
        with st.spinner("Analyzing website..."):
            text = extract_text_from_url(url)
            
            if text:
                with tabs[0]:
                    st.subheader("📝 Website Summary")
                    summary = get_summary(text)
                    st.write(summary)
                
                with tabs[1]:
                    st.subheader("💬 Chat with AI about the website")
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []

                    for message in st.session_state.chat_history:
                        st.write(f"{'You' if message['role'] == 'user' else 'AI'}: {message['content']}")

                    question = st.text_input("Ask a question about the website content:")
                    if question:
                        answer = chat_with_ai(question, text)
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.write("AI:", answer)
                
                with tabs[2]:
                    st.subheader("📊 Text Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        word_count = len(text.split())
                        st.metric("Word count", word_count)
                        
                        st.subheader("🔠 Word Cloud")
                        word_cloud_fig = generate_word_cloud(text)
                        st.pyplot(word_cloud_fig)
                    
                    with col2:
                        st.subheader("📈 Top Words")
                        word_freq = pd.Series(text.split()).value_counts().head(10)
                        fig = px.bar(x=word_freq.index, y=word_freq.values)
                        fig.update_layout(xaxis_title="Word", yaxis_title="Frequency")
                        st.plotly_chart(fig)
                
                with tabs[3]:
                    st.subheader("😊 Sentiment Analysis")
                    sentiment_df = perform_sentiment_analysis(text)
                    fig = px.histogram(sentiment_df, x='sentiment', nbins=20)
                    fig.update_layout(xaxis_title="Sentiment Score", yaxis_title="Frequency")
                    st.plotly_chart(fig)
                    
                    st.write("Sentiment Distribution")
                    st.dataframe(sentiment_df)
                    st.markdown(get_table_download_link(sentiment_df), unsafe_allow_html=True)
                
                with tabs[4]:
                    st.subheader("🌍 Language Detection and Translation")
                    detected_lang = detect_language(text)
                    st.write(f"Detected language: {detected_lang}")
                    
                    target_lang = st.selectbox("Select target language for translation:", ['en', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'zh-cn'])
                    if st.button("Translate"):
                        translated_text = translate_text(text[:1000], target_lang)
                        st.write("Translated text (first 1000 characters):")
                        st.write(translated_text)
                
                with tabs[5]:
                    st.subheader("🕸️ Topic Relationship Graph")
                    topic_graph = generate_topic_graph(text)
                    st.plotly_chart(topic_graph)
                
                with tabs[6]:
                    st.subheader("🕷️ Web Crawler")
                    max_pages = st.slider("Maximum number of pages to crawl", min_value=1, max_value=50, value=10)
                    
                    if st.button("Start Crawling"):
                        with st.spinner("Crawling website..."):
                            crawl_results = asyncio.run(crawl_website(url, max_pages))
                            st.session_state.crawl_results = crawl_results
                            st.success(f"Crawled {len(crawl_results)} pages.")

                    if hasattr(st.session_state, 'crawl_results'):
                        for i, result in enumerate(st.session_state.crawl_results):
                            with st.expander(f"Page {i+1}: {result['url']}"):
                                if st.button(f"Summarize Page {i+1}", key=f"summarize_{i}"):
                                    with st.spinner("Generating summary..."):
                                        summary = get_summary(result['text'], max_tokens=2024)
                                        st.write("Summary:")
                                        st.write(summary)
                                st.write("Full text:")
                                st.write(result['text'][:500] + "...")
            else:
                st.error("Failed to extract text from the given URL. Please try another website.")


if __name__ == "__main__":
    main()
