"""
Query caching module for storing and retrieving previous query results.
Provides fast response for repeated queries without re-running the RAG pipeline.
"""

import hashlib
import json
import os
import pickle
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class QueryCache:
    """
    Cache for storing query-response pairs.
    
    Uses hash-based key generation from query text and selected sources
    to enable quick lookup of previously computed responses.
    
    Example:
        >>> cache = QueryCache()
        >>> cache.set("What is AI?", ["doc1.pdf"], {"answer": "AI is..."})
        >>> result = cache.get("What is AI?", ["doc1.pdf"])
        >>> print(result["answer"])
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_entries: int = 1000,
        ttl_hours: int = 24,
        persist_to_disk: bool = False
    ):
        """
        Initialize the query cache.
        
        Args:
            cache_dir: Directory for persistent cache (if persist_to_disk=True)
            max_entries: Maximum number of cached entries
            ttl_hours: Time-to-live for cache entries in hours
            persist_to_disk: Whether to persist cache to disk
        """
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        self.ttl_hours = ttl_hours
        self.persist_to_disk = persist_to_disk
        
        # In-memory cache: {hash: {"response": ..., "timestamp": ..., "hits": ...}}
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Track which sources are in the cache
        self._sources_in_cache: Dict[str, set] = {}  # source -> set of cache keys
        
        # Load from disk if available
        if persist_to_disk and cache_dir:
            self._load_from_disk()
    
    def _generate_key(self, query: str, sources: Optional[List[str]] = None) -> str:
        """
        Generate a unique cache key from query and sources.
        
        Args:
            query: The user query
            sources: List of selected source files (or None for all)
            
        Returns:
            Hash string as cache key
        """
        # Normalize query
        normalized_query = query.strip().lower()
        
        # Sort sources for consistent hashing
        if sources:
            sorted_sources = sorted(sources)
            sources_str = "|".join(sorted_sources)
        else:
            sources_str = "__ALL__"
        
        # Combine and hash
        combined = f"{normalized_query}:::{sources_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def get(
        self,
        query: str,
        sources: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for a query.
        
        Args:
            query: The user query
            sources: List of selected source files
            
        Returns:
            Cached response dict or None if not found/expired
        """
        key = self._generate_key(query, sources)
        
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check TTL
        if self._is_expired(entry):
            self._remove_entry(key)
            return None
        
        # Update hit count
        entry['hits'] = entry.get('hits', 0) + 1
        entry['last_accessed'] = datetime.now().isoformat()
        
        return entry['response']
    
    def set(
        self,
        query: str,
        sources: Optional[List[str]],
        response: Dict[str, Any]
    ) -> None:
        """
        Store a query-response pair in the cache.
        
        Args:
            query: The user query
            sources: List of selected source files
            response: The response dict to cache
        """
        key = self._generate_key(query, sources)
        
        # Evict if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()
        
        # Store entry
        self._cache[key] = {
            'response': response,
            'query': query,
            'sources': sources,
            'timestamp': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'hits': 0
        }
        
        # Track sources
        source_list = sources if sources else ['__ALL__']
        for source in source_list:
            if source not in self._sources_in_cache:
                self._sources_in_cache[source] = set()
            self._sources_in_cache[source].add(key)
        
        # Persist if enabled
        if self.persist_to_disk:
            self._save_to_disk()
    
    def invalidate(self, source: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            source: Specific source to invalidate, or None for all
            
        Returns:
            Number of entries invalidated
        """
        if source is None:
            # Clear all
            count = len(self._cache)
            self._cache.clear()
            self._sources_in_cache.clear()
            return count
        
        # Invalidate entries for specific source
        if source not in self._sources_in_cache:
            return 0
        
        keys_to_remove = list(self._sources_in_cache[source])
        
        for key in keys_to_remove:
            self._remove_entry(key)
        
        # Also invalidate queries that used "all sources"
        if '__ALL__' in self._sources_in_cache:
            all_keys = list(self._sources_in_cache['__ALL__'])
            for key in all_keys:
                self._remove_entry(key)
        
        return len(keys_to_remove)
    
    def _remove_entry(self, key: str) -> None:
        """Remove a cache entry and update source tracking."""
        if key not in self._cache:
            return
        
        entry = self._cache[key]
        sources = entry.get('sources') or ['__ALL__']
        
        # Remove from source tracking
        for source in sources:
            if source in self._sources_in_cache:
                self._sources_in_cache[source].discard(key)
                if not self._sources_in_cache[source]:
                    del self._sources_in_cache[source]
        
        del self._cache[key]
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if a cache entry has expired."""
        try:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            expiry = timestamp + timedelta(hours=self.ttl_hours)
            return datetime.now() > expiry
        except (KeyError, ValueError):
            return True
    
    def _evict_oldest(self) -> None:
        """Evict the oldest/least used cache entry."""
        if not self._cache:
            return
        
        # Find entry with oldest last_accessed time
        oldest_key = None
        oldest_time = None
        
        for key, entry in self._cache.items():
            try:
                accessed = datetime.fromisoformat(entry.get('last_accessed', entry['timestamp']))
                if oldest_time is None or accessed < oldest_time:
                    oldest_time = accessed
                    oldest_key = key
            except (KeyError, ValueError):
                oldest_key = key
                break
        
        if oldest_key:
            self._remove_entry(oldest_key)
    
    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.cache_dir:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, 'query_cache.pkl')
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'cache': self._cache,
                    'sources': {k: list(v) for k, v in self._sources_in_cache.items()}
                }, f)
        except Exception as e:
            print(f"Warning: Failed to save cache to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.cache_dir:
            return
        
        cache_file = os.path.join(self.cache_dir, 'query_cache.pkl')
        
        if not os.path.exists(cache_file):
            return
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self._cache = data.get('cache', {})
                sources = data.get('sources', {})
                self._sources_in_cache = {k: set(v) for k, v in sources.items()}
                
                # Clean up expired entries
                expired = [k for k, v in self._cache.items() if self._is_expired(v)]
                for key in expired:
                    self._remove_entry(key)
                    
        except Exception as e:
            print(f"Warning: Failed to load cache from disk: {e}")
            self._cache = {}
            self._sources_in_cache = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.get('hits', 0) for e in self._cache.values())
        
        return {
            'entries': len(self._cache),
            'max_entries': self.max_entries,
            'total_hits': total_hits,
            'sources_tracked': len(self._sources_in_cache),
            'ttl_hours': self.ttl_hours
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._sources_in_cache.clear()
        
        if self.persist_to_disk and self.cache_dir:
            cache_file = os.path.join(self.cache_dir, 'query_cache.pkl')
            if os.path.exists(cache_file):
                os.remove(cache_file)
