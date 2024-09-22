import nltk
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
import jwt
from jwt import PyJWTError
from scraper import get_answer as scraper_get_answer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBearer()

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
    chat_id: int | None = None

class UserMessage(BaseModel):
    user_id: str
    message: str
    is_user: bool
def get_db():
    conn = sqlite3.connect('chatbot.db')
    conn.row_factory = sqlite3.Row
    return conn
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

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
    
    top_indices = cosine_similarities.argsort()[-3:][::-1]  # Get top 3 matches
    
    responses = []
    for idx in top_indices:
        similarity = cosine_similarities[idx]
        logger.info(f"Match {idx}: Similarity {similarity}")
        logger.info(f"Question: {content[idx]['question']}")
        logger.info(f"Answer: {content[idx]['answer'][:100]}...")
        
        if similarity > 0.1:  # Lowered threshold
            responses.append((content[idx]['answer'], similarity))

    if responses:
        # Extract the most relevant sentence or paragraph
        best_response = responses[0][0]
        sentences = best_response.split('.')
        relevant_sentence = next((s for s in sentences if 'ISET\'COM' in s), sentences[0])
        return relevant_sentence.strip() + '.'
    else:
        logger.info("No matches found above threshold")
        return None


@app.get("/user_messages")
async def get_user_messages(user: dict = Depends(verify_token)):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT message, is_user FROM user_messages WHERE user_id = ? ORDER BY id", (user['sub'],))
    messages = cursor.fetchall()
    db.close()
    return {"messages": [{"text": row['message'], "isUser": row['is_user']} for row in messages]}

def save_user_message(user_id: str, message: str, is_user: bool):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("INSERT INTO user_messages (user_id, message, is_user) VALUES (?, ?, ?)", (user_id, message, is_user))
    db.commit()
    db.close()

# Add this function to create the user_messages table
def create_user_messages_table():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        message TEXT NOT NULL,
        is_user BOOLEAN NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    db.commit()
    db.close()

# Call this function when your app starts
create_user_messages_table()

@app.get("/chat_history")
async def get_chat_history(user: dict = Depends(verify_token)):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, title, timestamp FROM chats WHERE user_id = ? ORDER BY timestamp DESC", (user['sub'],))
    history = cursor.fetchall()
    db.close()
    return {"history": [{"id": row['id'], "title": row['title'], "timestamp": row['timestamp']} for row in history]}

@app.get("/chat/{chat_id}")
async def get_chat(chat_id: int, user: dict = Depends(verify_token)):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT message, is_user FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,))
    messages = cursor.fetchall()
    db.close()
    return {"messages": [{"text": row['message'], "isUser": row['is_user']} for row in messages]}

@app.post("/chat")
async def chat(message: Message, user: dict = Depends(verify_token)):
    try:
        logger.info(f"Received message from user {user['sub']}: {message.user_input}")
        
        processed_text = process_text(message.user_input)
        logger.info(f"Processed text: {processed_text}")
        
        context = conversation_history.get_context()
        logger.info(f"Context: {context}")
        
        response, similarity = scraper_get_answer(message.user_input)
        logger.info(f"Response from get_answer: {response} (Similarity: {similarity})")
        
        db = get_db()
        cursor = db.cursor()
        
        # Create a new chat if chat_id is not provided
        if message.chat_id is None:
            cursor.execute("INSERT INTO chats (user_id, title) VALUES (?, ?)", (user['sub'], message.user_input[:30]))
            chat_id = cursor.lastrowid
        else:
            chat_id = message.chat_id
        
        # Save the user message
        cursor.execute("INSERT INTO messages (chat_id, message, is_user) VALUES (?, ?, ?)", (chat_id, message.user_input, True))
        
        if similarity > 60:  # Adjust this threshold as needed
            summarized_response = summarize_response(response)
            logger.info(f"Summarized response: {summarized_response}")
            conversation_history.add(message.user_input, summarized_response)
            
            # Save the bot response
            cursor.execute("INSERT INTO messages (chat_id, message, is_user) VALUES (?, ?, ?)", (chat_id, summarized_response, False))
            
            db.commit()
            db.close()
            return {"message": summarized_response, "method": "intent_matching", "similarity": similarity, "chat_id": chat_id}
        else:
            logger.info("No good match found, using default response")
            default_response = "Je suis désolé, je n'ai pas de réponse précise à cette question. Pouvez-vous reformuler ou poser une autre question ?"
            conversation_history.add(message.user_input, default_response)
            
            # Save the bot response
            cursor.execute("INSERT INTO messages (chat_id, message, is_user) VALUES (?, ?, ?)", (chat_id, default_response, False))
            
            db.commit()
            db.close()
            return {"message": default_response, "method": "default", "similarity": similarity, "chat_id": chat_id}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

from collections import deque

class ConversationHistory:
    def __init__(self, max_length=5):
        self.history = deque(maxlen=max_length)

    def add(self, user_input, bot_response):
        self.history.append((user_input, bot_response))
        logger.info(f"Added to history: User: {user_input}, Bot: {bot_response}")

    def get_context(self):
        context = " ".join([f"{u} {b}" for u, b in self.history])
        logger.info(f"Current context: {context}")
        return context

conversation_history = ConversationHistory()

def direct_db_query(query):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT answer FROM faq WHERE question LIKE ? LIMIT 1", ('%' + query + '%',))
    result = cursor.fetchone()
    db.close()
    if result:
        sentences = result['answer'].split('.')
        relevant_sentence = next((s for s in sentences if query.lower() in s.lower()), sentences[0])
        return relevant_sentence.strip() + '.'
    return None

@app.get("/check_db")
async def check_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM faq")
    rows = cursor.fetchall()
    db.close()
    return {"db_content": [dict(row) for row in rows]}

@app.get("/debug")
async def debug():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM faq LIMIT 10")
    faq_sample = cursor.fetchall()
    cursor.execute("SELECT * FROM user_messages ORDER BY id DESC LIMIT 10")
    recent_messages = cursor.fetchall()
    db.close()
    
    return {
        "faq_sample": [dict(row) for row in faq_sample],
        "recent_messages": [dict(row) for row in recent_messages],
        "conversation_history": list(conversation_history.history)
    }

def summarize_response(response, max_length=800):
    if len(response) <= max_length:
        return response
    sentences = response.split('.')
    summary = ''
    for sentence in sentences:
        if len(summary) + len(sentence) <= max_length:
            summary += sentence + '.'
        else:
            break
    return summary.strip()

def create_tables():
    db = get_db()
    cursor = db.cursor()
    
    # Create chats table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        is_user BOOLEAN NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chat_id) REFERENCES chats (id)
    )
    """)
    
    db.commit()
    db.close()

# Call this function when your app starts
create_tables()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

