import nltk
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import os
import sqlite3
import traceback
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set NLTK data path and download resources
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
    # Add this line to download the French punctuation resource
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

class Message(BaseModel):
    user_input: str

# NLP processing function
def process_text(text):
    try:
        # Tokenize
        tokens = word_tokenize(text.lower(), language='french')
        
        # Remove punctuation and numbers
        tokens = [token for token in tokens if token not in string.punctuation and not token.isdigit()]
        
        # Remove stopwords
        stop_words = set(stopwords.words('french'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Stem words
        stemmer = FrenchStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        logger.info(f"Processed tokens: {tokens}")
        return tokens
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing text")

# Intent matching function
def match_intent(processed_text, context):
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute("SELECT question, answer FROM faq")
    content = cursor.fetchall()
    
    logger.info(f"Number of entries in database: {len(content)}")
    
    if len(content) == 0:
        logger.error("Database is empty!")
        return None
    
    documents = [f"{row['question']} {row['answer']}" for row in content]
    processed_documents = [' '.join(process_text(doc)) for doc in documents]
    
    query = f"{context} {' '.join(processed_text)}"
    processed_query = ' '.join(process_text(query))
    
    logger.info(f"Processed query: {processed_query}")
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(processed_documents + [processed_query])
    
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Log all similarities
    for idx, sim in enumerate(cosine_similarities):
        logger.info(f"Document {idx}: Similarity {sim}")
    
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    
    responses = []
    for idx in top_indices:
        similarity = cosine_similarities[idx]
        logger.info(f"Match {idx}: Similarity {similarity}")
        logger.info(f"Question: {content[idx]['question']}")
        logger.info(f"Answer: {content[idx]['answer'][:100]}...")  # Log first 100 chars of answer
        
        if similarity > 0.01:  # Lowered threshold significantly for debugging
            responses.append((content[idx]['answer'], similarity))
    
    if responses:
        sorted_responses = sorted(responses, key=lambda x: x[1], reverse=True)
        logger.info(f"Number of responses: {len(sorted_responses)}")
        
        final_response = sorted_responses[0][0]  # Just take the best match for now
        return final_response
    else:
        logger.info("No matches found above threshold")
        return None

@app.post("/chat")
async def chat(message: Message):
    try:
        logger.info(f"Received message: {message.user_input}")
        
        processed_text = process_text(message.user_input)
        logger.info(f"Processed text: {processed_text}")
        
        context = conversation_history.get_context()
        logger.info(f"Context: {context}")
        
        response = match_intent(processed_text, context)
        if response:
            logger.info(f"Response from match_intent: {response[:100]}...")
            conversation_history.add(message.user_input, response)
            return {"message": response, "method": "intent_matching"}
        else:
            logger.info("No response found")
            return {"message": "Je suis désolé, je n'ai pas de réponse à cette question.", "method": "default"}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history")
async def get_chat_history():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT user_input, bot_response FROM messages")
    history = cursor.fetchall()
    db.close()
    return {"history": [{"user_input": row['user_input'], "bot_response": row['bot_response']} for row in history]}

from collections import deque

class ConversationHistory:
    def __init__(self, max_length=5):
        self.history = deque(maxlen=max_length)

    def add(self, user_input, bot_response):
        self.history.append((user_input, bot_response))

    def get_context(self):
        return " ".join([f"{u} {b}" for u, b in self.history])

conversation_history = ConversationHistory()

def get_db():
    conn = sqlite3.connect('chatbot.db')
    conn.row_factory = sqlite3.Row
    return conn

def direct_db_query(query):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT answer FROM faq WHERE question LIKE ?", ('%' + query + '%',))
    result = cursor.fetchone()
    db.close()
    return result['answer'] if result else None

@app.get("/check_db")
async def check_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM faq")
    rows = cursor.fetchall()
    db.close()
    return {"db_content": [dict(row) for row in rows]}

