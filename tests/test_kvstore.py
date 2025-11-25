import os
import tempfile
import time
import unittest

from src.kvstore import KVStore, create_kv_store


class TestKVStore(unittest.TestCase):
    def setUp(self):
        self.kv = create_kv_store(max_size=3)
    
    def tearDown(self):
        self.kv.close()
        if hasattr(self, 'temp_db'):
            try:
                os.unlink(self.temp_db)
            except:
                pass
    
    def test_basic_operations(self):
        """Test basic set, get, and delete operations."""
        # Test set and get
        self.kv.set('key1', 'value1')
        self.assertEqual(self.kv.get('key1'), 'value1')
        
        # Test update
        self.kv.set('key1', 'new_value')
        self.assertEqual(self.kv.get('key1'), 'new_value')
        
        # Test non-existent key
        self.assertIsNone(self.kv.get('nonexistent'))
        self.assertEqual(self.kv.get('nonexistent', 'default'), 'default')
        
        # Test delete
        self.kv.delete('key1')
        self.assertIsNone(self.kv.get('key1'))
    
    def test_ttl(self):
        """Test time-to-live functionality."""
        self.kv.set('key1', 'value1', ttl=0.1)  # 100ms TTL
        self.assertEqual(self.kv.get('key1'), 'value1')
        
        # Wait for TTL to expire
        time.sleep(0.2)
        self.assertIsNone(self.kv.get('key1'))
    
    def test_lru_eviction(self):
        """Test least-recently-used eviction policy."""
        # Fill the cache
        self.kv.set('key1', 'value1')
        self.kv.set('key2', 'value2')
        self.kv.set('key3', 'value3')
        
        # Access key1 to make it most recently used
        self.kv.get('key1')
        
        # Add one more item - key2 should be evicted (least recently used)
        self.kv.set('key4', 'value4')
        
        self.assertIsNone(self.kv.get('key2'))  # Should be evicted
        self.assertEqual(self.kv.get('key1'), 'value1')  # Should still be there
        self.assertEqual(self.kv.get('key3'), 'value3')
        self.assertEqual(self.kv.get('key4'), 'value4')
    
    def test_persistence(self):
        """Test SQLite persistence."""
        # Create a temporary file for the database
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.temp_db = f.name
        
        try:
            # Create a persistent store and add some data
            with create_kv_store(max_size=10, persist_path=self.temp_db) as kv:
                kv.set('key1', 'value1', ttl=3600)
                kv.set('key2', 'value2')
            
            # Create a new store with the same database file
            with create_kv_store(max_size=10, persist_path=self.temp_db) as kv2:
                self.assertEqual(kv2.get('key1'), 'value1')
                self.assertEqual(kv2.get('key2'), 'value2')
                self.assertEqual(len(kv2), 2)
        finally:
            # Clean up
            if os.path.exists(self.temp_db):
                os.unlink(self.temp_db)
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with create_kv_store() as kv:
            kv['key1'] = 'value1'
            self.assertEqual(kv['key1'], 'value1')
        
        # Should still be able to access the store after exiting context
        self.assertEqual(kv['key1'], 'value1')
        kv.close()


if __name__ == '__main__':
    unittest.main()