import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ChatInterface.css';
import { useUser, SignIn, SignOutButton } from '@clerk/clerk-react';

function ChatInterface() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const { isSignedIn, user } = useUser();

  useEffect(() => {
    if (isSignedIn) {
      fetchUserMessages();
    }
  }, [isSignedIn]);

  const fetchUserMessages = async () => {
    try {
      const token = await user.getToken();
      const response = await axios.get('http://localhost:8000/user_messages', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessages(response.data.messages);
    } catch (error) {
      console.error('Error fetching user messages:', error);
    }
  };

  const handleSend = async () => {
    if (input.trim() === '' || !isSignedIn) return;

    try {
      const userMessage = { text: input, isUser: true };
      setMessages([...messages, userMessage]);

      const token = await user.getToken();
      const response = await axios.post('http://localhost:8000/chat', 
        { user_input: input },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      
      const botMessage = { text: response.data.message, isUser: false };
      setMessages(prevMessages => [...prevMessages, botMessage]);

      setInput('');
    } catch (error) {
      console.error('Error sending message:', error);
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
        {isSignedIn ? (
          <div>
            <p>Welcome, {user.firstName}!</p>
            <SignOutButton />
          </div>
        ) : (
          <SignIn />
        )}
      </div>
      {isSignedIn ? (
        <>
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
        </>
      ) : (
        <p>Please sign in to use the chat interface.</p>
      )}
    </div>
  );
}

export default ChatInterface;