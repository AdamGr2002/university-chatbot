from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

# Database connection
def get_db():
    conn = sqlite3.connect('chatbot.db')
    conn.row_factory = sqlite3.Row
    return conn

class Message(BaseModel):
    user_input: str

# NLP processing function
def process_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Intent matching function
def match_intent(processed_text):
    db = get_db()
    cursor = db.cursor()
    
    # Get all scraped content
    cursor.execute("SELECT question, answer FROM faq")
    content = cursor.fetchall()
    
    # Combine question and answer for better matching
    documents = [f"{row['question']} {row['answer']}" for row in content]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents + [" ".join(processed_text)])
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    best_match_index = cosine_similarities.argmax()
    
    db.close()
    
    if cosine_similarities[best_match_index] > 0.3:  # Adjust threshold as needed
        return content[best_match_index]['answer']
    else:
        return None

@app.post("/chat")
async def chat(message: Message):
    processed_text = process_text(message.user_input)
    intent_response = match_intent(processed_text)
    
    if intent_response:
        bot_response = intent_response
    else:
        bot_response = "I'm sorry, I don't have information about that. Can you please rephrase or ask something else?"
    
    # Save to database
    db = get_db()
    cursor = db.cursor()
    cursor.execute("INSERT INTO messages (user_input, bot_response) VALUES (?, ?)",
                   (message.user_input, bot_response))
    db.commit()
    db.close()
    
    return {"response": bot_response}

@app.get("/chat_history")
async def get_chat_history():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT user_input, bot_response FROM messages")
    history = cursor.fetchall()
    db.close()
    return {"history": [{"user_input": row['user_input'], "bot_response": row['bot_response']} for row in history]}