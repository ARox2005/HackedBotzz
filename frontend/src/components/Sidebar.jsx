import { useState, useEffect } from 'react';
import { getSources, getStats, uploadDocument, deleteDocument, clearCache, clearKnowledgeBase } from '../api';
import './Sidebar.css';

function Sidebar({ selectedSources, setSelectedSources, useAllDocs, setUseAllDocs, isOpen, onClose }) {
    const [sources, setSources] = useState([]);
    const [stats, setStats] = useState({ total_chunks: 0, total_sources: 0, cache_entries: 0, cache_hits: 0 });
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState('');
    const [successMsg, setSuccessMsg] = useState('');

    const fetchData = async () => {
        try {
            const [sourcesData, statsData] = await Promise.all([
                getSources(),
                getStats()
            ]);
            setSources(sourcesData);
            setStats(statsData);
            setError('');
        } catch (err) {
            setError('Failed to connect to backend');
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    const showSuccess = (msg) => {
        setSuccessMsg(msg);
        setTimeout(() => setSuccessMsg(''), 3000);
    };

    const handleUpload = async (e) => {
        const files = e.target.files;
        if (!files.length) return;

        setUploading(true);
        setError('');

        let successCount = 0;
        for (const file of files) {
            try {
                await uploadDocument(file);
                successCount++;
            } catch (err) {
                setError(`Failed to upload ${file.name}: ${err.message}`);
            }
        }

        setUploading(false);
        if (successCount > 0) {
            showSuccess(`✓ ${successCount} file(s) uploaded`);
        }
        fetchData();
        e.target.value = '';
    };

    const handleDelete = async (filename) => {
        if (!confirm(`Delete ${filename}?`)) return;

        try {
            await deleteDocument(filename);
            showSuccess(`✓ Deleted ${filename}`);
            fetchData();
        } catch (err) {
            setError(`Failed to delete: ${err.message}`);
        }
    };

    const handleClearCache = async () => {
        try {
            await clearCache();
            showSuccess('✓ Cache cleared');
            fetchData();
        } catch (err) {
            setError('Failed to clear cache');
        }
    };

    const handleClearKnowledgeBase = async () => {
        if (!confirm('Are you sure you want to delete ALL documents and embeddings? This cannot be undone.')) return;

        try {
            await clearKnowledgeBase();
            showSuccess('✓ Knowledge base cleared');
            setSelectedSources([]);
            fetchData();
        } catch (err) {
            setError('Failed to clear knowledge base');
        }
    };

    const toggleSource = (path) => {
        if (selectedSources.includes(path)) {
            setSelectedSources(selectedSources.filter(s => s !== path));
        } else {
            setSelectedSources([...selectedSources, path]);
        }
    };

    return (
        <div className={`sidebar ${isOpen ? 'open' : ''}`}>
            <div className="sidebar-header">
                <h1>Knowledge Base</h1>
            </div>

            {error && <div className="error-banner">⚠️ {error}</div>}
            {successMsg && <div className="success-banner">{successMsg}</div>}

            <div className="sidebar-section">
                <h3>Upload Documents</h3>
                <label className="upload-button">
                    <input
                        type="file"
                        multiple
                        accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"
                        onChange={handleUpload}
                        disabled={uploading}
                    />
                    {uploading ? (
                        <>
                            <span className="spinner"></span>
                            Uploading...
                        </>
                    ) : (
                        <>📤 Choose Files</>
                    )}
                </label>
                <p className="help-text">PDF, DOCX, TXT, PNG, JPG</p>
            </div>

            <div className="sidebar-section">
                <h3>Context Selection</h3>
                <label className="toggle-row">
                    <input
                        type="checkbox"
                        checked={useAllDocs}
                        onChange={(e) => setUseAllDocs(e.target.checked)}
                    />
                    <span>Use all documents</span>
                </label>
                {!useAllDocs && (
                    <p className="selection-hint">
                        👇 Select specific documents below
                        {selectedSources.length > 0 && (
                            <span className="selected-count"> ({selectedSources.length} selected)</span>
                        )}
                    </p>
                )}
            </div>

            <div className="sidebar-section documents-section">
                <h3>Documents ({sources.length})</h3>
                <div className="documents-list">
                    {sources.length === 0 ? (
                        <p className="empty-text">No documents yet</p>
                    ) : (
                        sources.map((source, index) => (
                            <div
                                key={source.path}
                                className="document-item"
                                style={{ animationDelay: `${index * 0.05}s` }}
                            >
                                {!useAllDocs && (
                                    <input
                                        type="checkbox"
                                        checked={selectedSources.includes(source.path)}
                                        onChange={() => toggleSource(source.path)}
                                    />
                                )}
                                <span className="doc-name" title={source.filename}>
                                    📄 {source.filename}
                                </span>
                                <button
                                    className="delete-btn"
                                    onClick={() => handleDelete(source.filename)}
                                    title="Delete"
                                >
                                    🗑️
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            <div className="sidebar-section stats-section">
                <h3>📊 Statistics</h3>
                <div className="stats-grid">
                    <div className="stat">
                        <span className="stat-value">{stats.total_sources}</span>
                        <span className="stat-label">Documents</span>
                    </div>
                    <div className="stat">
                        <span className="stat-value">{stats.total_chunks}</span>
                        <span className="stat-label">Chunks</span>
                    </div>
                    <div className="stat">
                        <span className="stat-value">{stats.cache_entries}</span>
                        <span className="stat-label">Cached</span>
                    </div>
                    <div className="stat">
                        <span className="stat-value">{stats.cache_hits}</span>
                        <span className="stat-label">Hits</span>
                    </div>
                </div>
            </div>

            <div className="sidebar-footer">
                <button className="clear-btn" onClick={handleClearCache}>
                    🧹 Clear Cache
                </button>
                <button className="clear-btn danger-btn" onClick={handleClearKnowledgeBase}>
                    🗑️ Clear All Data
                </button>
            </div>
        </div>
    );
}

export default Sidebar;
