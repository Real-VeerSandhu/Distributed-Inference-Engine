import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Tuple


@dataclass
class CacheEntry:
    """Represents a single cache entry with value and metadata."""
    value: Any
    created_at: float
    ttl: Optional[float] = None  # in seconds


class KVStore:
    """
    In-memory key-value store with LRU eviction and optional SQLite persistence.
    
    Features:
    - LRU (Least Recently Used) eviction when max_size is reached
    - TTL (Time To Live) support for cache entries
    - Optional SQLite persistence for warm restarts
    - Thread-safe operations
    """
    
    def __init__(self, max_size: int = 1000, persist_path: Optional[str] = None):
        """
        Initialize the KVStore.
        
        Args:
            max_size: Maximum number of items to store before evicting the least recently used
            persist_path: If provided, enables SQLite persistence at the given path
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.persist_path = persist_path
        self.conn = None
        
        if self.persist_path:
            self._init_db()
            self._load_from_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database for persistence."""
        self.conn = sqlite3.connect(self.persist_path)
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS kv_store (
            key TEXT PRIMARY KEY,
            value BLOB,
            created_at REAL,
            ttl REAL
        )
        ''')
        self.conn.commit()
    
    def _load_from_db(self) -> None:
        """Load entries from SQLite database into memory."""
        if not self.conn:
            return
            
        cursor = self.conn.cursor()
        cursor.execute('SELECT key, value, created_at, ttl FROM kv_store')
        
        now = time.time()
        for key, value, created_at, ttl in cursor.fetchall():
            # Skip expired entries
            if ttl and (now - created_at) > ttl:
                self._delete_from_db(key)
                continue
                
            self.cache[key] = CacheEntry(
                value=self._deserialize(value),
                created_at=created_at,
                ttl=ttl
            )
    
    def _save_to_db(self, key: str, entry: CacheEntry) -> None:
        """Save an entry to the SQLite database."""
        if not self.conn:
            return
            
        cursor = self.conn.cursor()
        cursor.execute(
            'REPLACE INTO kv_store (key, value, created_at, ttl) VALUES (?, ?, ?, ?)',
            (key, self._serialize(entry.value), entry.created_at, entry.ttl)
        )
        self.conn.commit()
    
    def _delete_from_db(self, key: str) -> None:
        """Delete an entry from the SQLite database."""
        if not self.conn:
            return
            
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM kv_store WHERE key = ?', (key,))
        self.conn.commit()
    
    def _evict_if_needed(self) -> None:
        """Evict the least recently used item if we've reached max size."""
        if len(self.cache) >= self.max_size:
            # Remove the first item (least recently used)
            key, _ = self.cache.popitem(last=False)
            self._delete_from_db(key)
    
    def _get_current_time(self) -> float:
        """Get current time in seconds since epoch."""
        return time.time()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry has expired."""
        if entry.ttl is None:
            return False
        return (self._get_current_time() - entry.created_at) > entry.ttl
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize a value for storage."""
        import pickle
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize a value from storage."""
        import pickle
        return pickle.loads(data)
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a key-value pair in the store with an optional TTL.
        
        Args:
            key: The key to set
            value: The value to store
            ttl: Time to live in seconds (None for no expiration)
        """
        # Check if we need to evict an item
        if key not in self.cache:
            self._evict_if_needed()
        
        # Create and store the new entry
        entry = CacheEntry(
            value=value,
            created_at=self._get_current_time(),
            ttl=ttl
        )
        
        # Update the cache
        self.cache[key] = entry
        self.cache.move_to_end(key)  # Mark as recently used
        
        # Persist if needed
        self._save_to_db(key, entry)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the store by key.
        
        Args:
            key: The key to look up
            default: Default value to return if key is not found or expired
            
        Returns:
            The stored value or default if not found/expired
        """
        if key not in self.cache:
            return default
        
        entry = self.cache[key]
        
        # Check if the entry has expired
        if self._is_expired(entry):
            self.delete(key)
            return default
        
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        return entry.value
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the store.
        
        Args:
            key: The key to delete
            
        Returns:
            True if the key was deleted, False if it didn't exist
        """
        if key in self.cache:
            del self.cache[key]
            self._delete_from_db(key)
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists and is not expired.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key exists and is not expired, False otherwise
        """
        if key not in self.cache:
            return False
            
        entry = self.cache[key]
        if self._is_expired(entry):
            self.delete(key)
            return False
            
        return True
    
    def clear(self) -> None:
        """Clear all entries from the store."""
        self.cache.clear()
        if self.conn:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM kv_store')
            self.conn.commit()
    
    def close(self) -> None:
        """Close the store and release resources."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        return self.exists(key)
    
    def __getitem__(self, key: str) -> Any:
        """Get a value by key, raising KeyError if not found or expired."""
        value = self.get(key)
        if value is None and key not in self.cache:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a key-value pair with no TTL."""
        self.set(key, value)
    
    def __delitem__(self, key: str) -> None:
        """Delete a key, raising KeyError if not found."""
        if not self.delete(key):
            raise KeyError(key)
    
    def __len__(self) -> int:
        """Get the number of non-expired entries in the store."""
        # Remove expired entries
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            self.delete(key)
            
        return len(self.cache)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the store."""
        self.close()


def create_kv_store(max_size: int = 1000, persist: bool = False) -> KVStore:
    """
    Create a new KVStore instance with optional persistence.
    
    Args:
        max_size: Maximum number of items to store
        persist: If True, enables SQLite persistence with a default path
        
    Returns:
        A new KVStore instance
    """
    persist_path = 'kvstore.db' if persist else None
    return KVStore(max_size=max_size, persist_path=persist_path)