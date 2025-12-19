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
            // If useAllDocs is false, we filter by selectedSources
            // If selectedSources is empty, no documents are selected - pass empty array
            // Backend will return no results for empty filter (correct behavior)
            const filterSources = useAllDocs ? null : (selectedSources.length > 0 ? selectedSources : []);

            // Warn if no sources selected when filtering
            if (!useAllDocs && selectedSources.length === 0) {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: '⚠️ No documents selected. Please select at least one document from the sidebar, or enable "Use all documents".',
                    isError: true
                }]);
                setIsLoading(false);
                return;
            }

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

    // Container styles for guaranteed layout
    const containerStyle = {
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        height: '100vh',
        position: 'relative',
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
        overflow: 'hidden'
    };

    const headerStyle = {
        flexShrink: 0,
        padding: '1.5rem 2rem',
        background: 'rgba(255, 255, 255, 0.03)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
        backdropFilter: 'blur(10px)'
    };

    const messagesStyle = {
        flex: 1,
        overflowY: 'auto',
        padding: '1.5rem',
        paddingBottom: '120px',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem'
    };

    const inputFormStyle = {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        display: 'flex',
        gap: '1rem',
        padding: '1.5rem 2rem',
        background: 'rgba(10, 10, 20, 0.98)',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        zIndex: 100
    };

    return (
        <div style={containerStyle}>
            <div style={headerStyle}>
                <h2 style={{
                    margin: 0,
                    color: '#fff',
                    fontSize: '1.5rem',
                    background: 'linear-gradient(135deg, #fff 0%, #a8b4ff 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent'
                }}>
                    🤖 RAG Chat with Qwen VLM
                </h2>
                <p style={{ margin: '0.25rem 0 0', color: 'rgba(255,255,255,0.5)', fontSize: '0.9rem' }}>
                    Ask questions about your documents
                </p>
            </div>

            <div style={messagesStyle}>
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

            <form style={inputFormStyle} onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a question about your documents..."
                    disabled={isLoading}
                    style={{
                        flex: 1,
                        padding: '1.1rem 1.5rem',
                        border: '2px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: '1rem',
                        background: 'rgba(255, 255, 255, 0.05)',
                        color: '#fff',
                        fontSize: '1rem',
                        outline: 'none'
                    }}
                />
                <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    style={{
                        padding: '1rem 1.75rem',
                        border: 'none',
                        borderRadius: '1rem',
                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        color: '#fff',
                        fontSize: '1.3rem',
                        cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
                        opacity: isLoading || !input.trim() ? 0.4 : 1
                    }}
                >
                    {isLoading ? '⏳' : '➤'}
                </button>
            </form>
        </div>
    );
}

export default ChatWindow;
