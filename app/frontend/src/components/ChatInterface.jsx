import React, { useState,useEffect } from 'react';
import axios from 'axios';
import './ChatInterface.css'; 



function ChatInterface() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);

  const handleSend = async () => {
    if (input.trim() === '') return;

    try {
      // Add user message to the chat
      const userMessage = { text: input, isUser: true };
      setMessages([...messages, userMessage]);

      // Make API call to backend
      const response = await axios.post('http://localhost:8000/chat', { user_input: input });
      
      // Add bot response to the chat

      const botMessage = { text: response.data.message, isUser: false };
      setMessages(prevMessages => [...prevMessages, botMessage]);


      setInput('');
    } catch (error) {
      console.error('Error sending message:', error);
      // Optionally, add an error message to the chat
      setMessages(prevMessages => [...prevMessages, { text: 'Error: Unable to get response', isUser: false }]);
    }
  };

  const SupportInfo = () => (
    <div className="support-info">
      <h3>Need more help?</h3>
      <p>Contact our support team at support@isetcom.tn or visit our website www.isetcom.tn for more information.</p>
    </div>
  );

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>Iset'Com AI Assistant</h2>
      </div>
      <div className="messages-container">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.isUser ? 'user-message' : 'bot-message'}`}>
            <span className="message-sender">{msg.isUser ? 'You' : 'AI'}</span>
            <p>{msg.text}</p>
          </div>
        ))}
      </div>
      {messages.length > 0 && messages[messages.length - 1].text && messages[messages.length - 1].text.includes("I'm sorry, I don't have information about that") && (
        <SupportInfo />
      )}
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask your question..."
          className="chat-input"

        />
        <button onClick={handleSend} className="send-button">Send</button>
      </div>
    </div>
  );
}

export default ChatInterface;