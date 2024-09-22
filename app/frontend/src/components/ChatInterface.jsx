import React, { useState, useEffect } from 'react';
import { useUser, useAuth } from '@clerk/clerk-react';
import axios from 'axios';
import ChatHistorySidebar from './ChatHistorySidebar';
import SignOutButton from './SignOutButton';
import SignIn from './SignIn';

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);

  const { isSignedIn, user } = useUser();
  const { getToken } = useAuth();

  useEffect(() => {
    if (isSignedIn) {
      fetchUserMessages();
      fetchChatHistory();
    }
  }, [isSignedIn]);

  const fetchUserMessages = async () => {
    try {
      const token = await getToken();
      const response = await axios.get('http://localhost:8000/user_messages', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessages(response.data.messages);
    } catch (error) {
      console.error('Error fetching user messages:', error);
    }
  };

  const fetchChatHistory = async () => {
    try {
      const token = await getToken();
      const response = await axios.get('http://localhost:8000/chat_history', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setChatHistory(response.data.history);

      // If there's a current chat, make sure it's selected in the UI
      if (currentChatId) {
        const today = new Date().toISOString().split('T')[0];
        const todayChats = response.data.history[today] || [];
        const currentChat = todayChats.find(chat => chat.id === currentChatId);
        if (currentChat) {
          onSelectChat(currentChat.id);
        }
      }
    } catch (error) {
      console.error('Error fetching chat history:', error);
    }
  };

  const handleSend = async () => {
    if (input.trim() === '' || !isSignedIn) return;

    try {
      const userMessage = { text: input, isUser: true };
      setMessages([...messages, userMessage]);

      const token = await getToken();
      const response = await axios.post('http://localhost:8000/chat', 
        { user_input: input, chat_id: currentChatId },  // Include chat_id if available
        { headers: { Authorization: `Bearer ${token}` } }
      );
      
      const botMessage = { text: response.data.message, isUser: false };
      setMessages(prevMessages => [...prevMessages, botMessage]);

      setInput('');
      
      // Update currentChatId if a new chat was created
      if (!currentChatId) {
        setCurrentChatId(response.data.chat_id);
        // Fetch updated chat history
        await fetchChatHistory();
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prevMessages => [...prevMessages, { text: 'Error: Unable to get response', isUser: false }]);
    }
  };

  const handleSelectChat = async (chatId) => {
    setCurrentChatId(chatId);
    try {
      const token = await getToken();
      const response = await axios.get(`http://localhost:8000/chat/${chatId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessages(response.data.messages);
    } catch (error) {
      console.error('Error fetching chat:', error);
    }
  };

  const onSelectChat = async (chatId) => {
    setCurrentChatId(chatId);
    try {
      const token = await getToken();
      const response = await axios.get(`http://localhost:8000/chat/${chatId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessages(response.data.messages);
    } catch (error) {
      console.error('Error fetching chat messages:', error);
    }
  };

  const SupportInfo = () => (
    <div className="support-info">
      <h3>Need more help?</h3>
      <p>Contact our support team at support@isetcom.tn or visit our website www.isetcom.tn for more information.</p>
    </div>
  );

  return (
    <div className="flex h-screen bg-gray-100">
      <div className="w-80 bg-indigo-900 text-white">
        <h1 className="text-2xl font-bold p-6 bg-indigo-800">ISETCOM Chatbot</h1>
        <ChatHistorySidebar chatHistory={chatHistory} onSelectChat={handleSelectChat} />
      </div>
      <div className="flex-1 flex flex-col">
        <div className="bg-white shadow-md p-6">
          {isSignedIn ? (
            <div className="flex justify-between items-center">
              <p className="text-xl text-indigo-900">Welcome, {user.firstName}!</p>
              <SignOutButton />
            </div>
          ) : (
            <SignIn />
          )}
        </div>
        {isSignedIn ? (
          <>
            <div className="flex-1 overflow-y-auto p-6 bg-white">
              {messages.map((msg, index) => (
                <div key={index} className={`mb-4 ${msg.isUser ? 'text-right' : 'text-left'}`}>
                  <div className={`inline-block p-3 rounded-lg max-w-md ${
                    msg.isUser ? 'bg-indigo-100 text-indigo-900' : 'bg-gray-200 text-gray-900'
                  }`}>
                    <span className="font-bold text-sm mb-1 block">{msg.isUser ? 'You' : 'AI Assistant'}</span>
                    <p className="text-sm">{msg.text}</p>
                  </div>
                </div>
              ))}
            </div>
            {messages.length > 0 && messages[messages.length - 1].text && messages[messages.length - 1].text.includes("I'm sorry, I don't have information about that") && (
              <SupportInfo />
            )}
            <div className="p-6 bg-white border-t border-gray-200">
              <div className="flex">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Ask your question..."
                  className="flex-1 p-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <button 
                  onClick={handleSend} 
                  className="bg-indigo-600 text-white px-6 py-3 rounded-r-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  Send
                </button>
              </div>
            </div>
          </>
        ) : (
          <p className="text-center mt-8 text-xl text-indigo-900">Please sign in to use the chat interface.</p>
        )}
      </div>
    </div>
  );
}

export default ChatInterface;