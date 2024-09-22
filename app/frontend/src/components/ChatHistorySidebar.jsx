import React from 'react';

function ChatHistorySidebar({ chatHistory, onSelectChat }) {
  return (
    <div className="overflow-y-auto h-full">
      <h3 className="text-xl font-semibold p-6 bg-indigo-800">Chat History</h3>
      {Object.entries(chatHistory).map(([date, chats]) => (
        <div key={date} className="mb-4">
          <h4 className="text-sm font-medium text-indigo-300 px-6 py-2">{date}</h4>
          {chats.map((chat) => (
            <div
              key={chat.id}
              className="px-6 py-2 hover:bg-indigo-800 cursor-pointer transition duration-150 ease-in-out"
              onClick={() => onSelectChat(chat.id)}
            >
              {chat.title || 'Untitled Chat'}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

export default ChatHistorySidebar;
