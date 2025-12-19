import { useState, useRef, useEffect } from 'react';
import { query } from '../api';
import './ChatWindow.css';

function ChatWindow({ selectedSources, useAllDocs }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput('');

        // Add user message
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
        setIsLoading(true);

        try {
            const filterSources = useAllDocs ? null : selectedSources;
            const result = await query(userMessage, filterSources);

            // Add assistant message
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: result.answer,
                sources: result.sources
            }]);
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Error: ${error.message}`,
                isError: true
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="chat-window">
            <div className="chat-header">
                <h2>🤖 RAG Chat with Qwen VLM</h2>
                <p>Ask questions about your documents</p>
            </div>

            <div className="messages-container">
                {messages.length === 0 && (
                    <div className="empty-state">
                        <div className="empty-icon">💬</div>
                        <h3>Start a conversation</h3>
                        <p>Upload documents and ask questions to get answers with citations</p>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        <div className="message-avatar">
                            {msg.role === 'user' ? '👤' : '🤖'}
                        </div>
                        <div className="message-content">
                            <div className={`message-bubble ${msg.isError ? 'error' : ''}`}>
                                {msg.content}
                            </div>
                            {msg.sources && msg.sources.length > 0 && (
                                <div className="message-sources">
                                    <span className="sources-label">📚 Sources:</span>
                                    {msg.sources.map((s, i) => (
                                        <span key={i} className="source-tag">{s}</span>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {isLoading && (
                    <div className="message assistant">
                        <div className="message-avatar">🤖</div>
                        <div className="message-content">
                            <div className="message-bubble loading">
                                <div className="loading-dots">
                                    <span></span><span></span><span></span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            <form className="chat-input-form" onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a question about your documents..."
                    disabled={isLoading}
                />
                <button type="submit" disabled={isLoading || !input.trim()}>
                    {isLoading ? '⏳' : '➤'}
                </button>
            </form>
        </div>
    );
}

export default ChatWindow;
