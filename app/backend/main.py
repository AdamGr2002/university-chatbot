from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from typing import List

app = FastAPI()

# Database setup
conn = sqlite3.connect('chatbot.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages
    (id INTEGER PRIMARY KEY, user_input TEXT, bot_response TEXT)
''')
conn.commit()

class Message(BaseModel):
    user_input: str

@app.post("/chat")
async def chat(message: Message):
    # TODO: Implement actual NLP processing here
    bot_response = f"You said: {message.user_input}. This is a placeholder response."
    
    # Save to database
    cursor.execute("INSERT INTO messages (user_input, bot_response) VALUES (?, ?)",
                   (message.user_input, bot_response))
    conn.commit()
    
    return {"response": bot_response}

@app.get("/chat_history")
async def get_chat_history():
    cursor.execute("SELECT user_input, bot_response FROM messages")
    history = cursor.fetchall()
    return {"history": [{"user_input": row[0], "bot_response": row[1]} for row in history]}