/**
 * API client for RAG backend
 */

const API_BASE = 'http://localhost:8000/api';

/**
 * Fetch pipeline statistics
 */
export async function getStats() {
    const response = await fetch(`${API_BASE}/stats`);
    if (!response.ok) throw new Error('Failed to fetch stats');
    return response.json();
}

/**
 * Fetch all document sources
 */
export async function getSources() {
    const response = await fetch(`${API_BASE}/sources`);
    if (!response.ok) throw new Error('Failed to fetch sources');
    return response.json();
}

/**
 * Upload a document
 */
export async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
}

/**
 * Delete a document
 */
export async function deleteDocument(filename) {
    const response = await fetch(`${API_BASE}/sources/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Delete failed');
    }

    return response.json();
}

/**
 * Execute a RAG query
 */
export async function query(queryText, filterSources = null) {
    const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: queryText,
            filter_sources: filterSources,
            stream: false,
        }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Query failed');
    }

    return response.json();
}

/**
 * Execute a streaming RAG query
 */
export async function* queryStream(queryText, filterSources = null) {
    const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: queryText,
            filter_sources: filterSources,
            stream: true,
        }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Query failed');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                yield data;
            }
        }
    }
}

/**
 * Clear query cache
 */
export async function clearCache() {
    const response = await fetch(`${API_BASE}/cache/clear`, {
        method: 'POST',
    });

    if (!response.ok) throw new Error('Failed to clear cache');
    return response.json();
}
