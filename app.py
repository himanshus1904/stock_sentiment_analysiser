import streamlit as st
import praw
import torch
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
import time
from config import CONFIG

# Streamlit App Title
st.title("📈 Reddit Stock Sentiment Analyzer")
st.write("Analyze stock sentiment from Reddit posts and comments.")

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=st.secrets["REDDIT_CLIENT_ID"],
    client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
    user_agent=st.secrets["REDDIT_USER_AGENT"]
)

# Load FinBERT model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model"]["name"])
    model = BertForSequenceClassification.from_pretrained(CONFIG["model"]["name"])
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Function to fetch Reddit data
def get_reddit_data(ticker):
    posts = []
    for sub in CONFIG["reddit"]["subreddit"]:
        try:
            subreddit = reddit.subreddit(sub)
            if not subreddit.display_name:
                st.warning(f"Skipping invalid subreddit: {sub}")
                continue
        except Exception as e:
            st.error(f"Error accessing subreddit {sub}: {e}")
            continue

        search_query = f'{ticker}'
        for post in subreddit.search(
            search_query, 
            sort='relevance', 
            time_filter=CONFIG["reddit"]["time_filter"],
            limit=CONFIG["reddit"]["post_limit"]
        ):
            try:
                post.comments.replace_more(limit=0)
                comments = [c.body.strip() for c in post.comments.list()[:CONFIG["reddit"]["comment_limit"]]]
                
                posts.append({
                    "subreddit": sub,
                    "title": post.title.strip(),
                    "text": post.selftext.strip(),
                    "comments": comments,
                    "created": post.created_utc,
                    "score": post.score
                })
                time.sleep(1)  # Rate limiting
            except Exception as e:
                st.error(f"Error processing post: {e}")
    
    return posts

# Function to analyze sentiment in batches
def analyze_batch(texts):
    if not texts:
        return []
    
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncuration=True, 
        max_length=256
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = softmax(outputs.logits.numpy(), axis=1)
    return [(CONFIG["sentiment_labels"][p.argmax()], p.max()) for p in probs]

# Function to process data and generate CSV
def process_data(ticker, posts):
    all_texts = []
    metadata = []
    
    for post in posts:
        if post["title"]:
            all_texts.append(post["title"])
            metadata.append({
                "type": "title",
                "subreddit": post["subreddit"],
                "created": post["created"],
                "score": post["score"]
            })
        
        if post["text"]:
            all_texts.append(post["text"])
            metadata.append({
                "type": "post",
                "subreddit": post["subreddit"],
                "created": post["created"],
                "score": post["score"]
            })
        
        for comment in post["comments"]:
            if comment:
                all_texts.append(comment)
                metadata.append({
                    "type": "comment",
                    "subreddit": post["subreddit"],
                    "created": post["created"],
                    "score": None
                })
    
    results = []
    for i in tqdm(range(0, len(all_texts), CONFIG["model"]["batch_size"]),
                  desc="Analyzing sentiment"):
        batch = all_texts[i:i+CONFIG["model"]["batch_size"]]
        results.extend(analyze_batch(batch))
    
    df = pd.DataFrame([{
        "ticker": ticker,
        "text": text,
        "sentiment": result[0],
        "confidence": result[1],
        **meta
    } for text, result, meta in zip(all_texts, results, metadata)])
    
    return df

# Function to generate report
def generate_report(df):
    if df.empty:
        return {"error": "No data found"}
    
    total = len(df)
    report = {
        "Positive": len(df[df["sentiment"] == "Positive"]) / total,
        "Neutral": len(df[df["sentiment"] == "Neutral"]) / total,
        "Negative": len(df[df["sentiment"] == "Negative"]) / total
    }
    
    if report["Positive"] > 0.7 and df["confidence"].mean() > 0.75:
        rating = "Strong Buy"
    elif report["Positive"] > 0.55:
        rating = "Buy"
    elif report["Negative"] > 0.7 and df["confidence"].mean() > 0.75:
        rating = "Strong Sell"
    elif report["Negative"] > 0.55:
        rating = "Sell"
    else:
        rating = "Hold"
    
    return {
        "rating": rating,
        "confidence": df["confidence"].mean(),
        "sample_size": total,
        **report
    }

# Streamlit UI
ticker = st.text_input("Enter stock ticker (e.g., RELIANCE, TCS):").strip().upper()

if st.button("Analyze"):
    if not ticker:
        st.error("Please enter a stock ticker.")
    else:
        with st.spinner("Fetching Reddit data..."):
            posts = get_reddit_data(ticker)
        
        if not posts:
            st.warning("No data found for the given ticker.")
        else:
            with st.spinner("Analyzing sentiment..."):
                df = process_data(ticker, posts)
                report = generate_report(df)
            
            st.success("Analysis complete!")
            
            # Display report
            st.subheader("📊 Sentiment Report")
            st.write(pd.DataFrame([report]).T)
            
            # Display sample data
            st.subheader("Sample Data")
            st.write(df.head())
            
            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{ticker}_sentiment_analysis.csv",
                mime="text/csv"
            )
