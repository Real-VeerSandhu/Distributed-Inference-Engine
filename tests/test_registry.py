import unittest
from unittest.mock import patch, MagicMock
from src.model_registry import ModelRegistry, ModelShard, ModelStatus, ModelVersion

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        """Set up a fresh ModelRegistry instance for each test."""
        self.registry = ModelRegistry()
        
        # Sample model data for testing
        self.sample_model = {
            "model_name": "test_model",
            "version": "1.0",
            "model_path": "/path/to/model",
            "input_schema": {"input": "float32"},
            "output_schema": {"output": "float32"},
            "batch_size": 1,
            "max_batch_size": 32,
            "quantized": False,
            "metadata": {"description": "Test model"}
        }
        
        # Sample shard data
        self.sample_shard = {
            "shard_id": 0,
            "worker_id": "worker-1",
            "metadata": {"gpu": "A100"}
        }
    
    def test_register_model(self):
        """Test registering a new model version."""
        # Register a model
        model_data = self.sample_model
        self.registry.register_model(**model_data)
        
        # Verify the model was registered
        model = self.registry.get_model_version(model_data["model_name"], model_data["version"])
        self.assertIsNotNone(model)
        self.assertEqual(model.version, model_data["version"])
        self.assertEqual(model.model_path, model_data["model_path"])
        self.assertEqual(len(self.registry.list_models()), 1)
        
        # Test registering the same model version again (should update)
        updated_data = model_data.copy()
        updated_data["model_path"] = "/new/path/to/model"
        self.registry.register_model(**updated_data)
        model = self.registry.get_model_version(model_data["model_name"], model_data["version"])
        self.assertEqual(model.model_path, "/new/path/to/model")
    
    def test_add_shard(self):
        """Test adding a shard to a model version."""
        # First register a model
        self.registry.register_model(**self.sample_model)
        
        # Add a shard
        shard_data = self.sample_shard
        self.registry.add_shard(
            model_name=self.sample_model["model_name"],
            version=self.sample_model["version"],
            **shard_data
        )
        
        # Get the model and verify the shard was added
        model = self.registry.get_model_version(
            self.sample_model["model_name"],
            self.sample_model["version"]
        )
        self.assertEqual(len(model.shards), 1)
        shard = model.shards[0]
        self.assertEqual(shard.shard_id, shard_data["shard_id"])
        self.assertEqual(shard.worker_id, shard_data["worker_id"])
        
        # Verify worker_models was updated
        worker_models = self.registry.get_worker_models(shard_data["worker_id"])
        self.assertIn((self.sample_model["model_name"], self.sample_model["version"]), worker_models)
    
    def test_get_shard_for_key(self):
        """Test consistent hashing for shard selection."""
        # Register a model and add multiple shards
        self.registry.register_model(**self.sample_model)
        
        # Add 3 shards
        for i in range(3):
            self.registry.add_shard(
                model_name=self.sample_model["model_name"],
                version=self.sample_model["version"],
                shard_id=i,
                worker_id=f"worker-{i}"
            )
        
        # Test consistent hashing with different keys
        key1 = "test_key_1"
        key2 = "test_key_2"
        
        shard1 = self.registry.get_shard_for_key(
            self.sample_model["model_name"],
            self.sample_model["version"],
            key1
        )
        shard2 = self.registry.get_shard_for_key(
            self.sample_model["model_name"],
            self.sample_model["version"],
            key1
        )
        shard3 = self.registry.get_shard_for_key(
            self.sample_model["model_name"],
            self.sample_model["version"],
            key2
        )
        
        # Same key should always map to same shard
        self.assertEqual(shard1.shard_id, shard2.shard_id)
        
        # Different keys might map to same or different shards
        # Just verify they are valid shards
        self.assertIn(shard1.shard_id, [0, 1, 2])
        self.assertIn(shard3.shard_id, [0, 1, 2])
    
    def test_serialization(self):
        """Test serialization and deserialization of the registry."""
        # Register a model and add a shard
        model_name = self.sample_model["model_name"]
        version = self.sample_model["version"]
        
        self.registry.register_model(**self.sample_model)
        self.registry.add_shard(
            model_name=model_name,
            version=version,
            **self.sample_shard
        )
        
        # Convert to dict and back
        registry_dict = self.registry.to_dict()
        new_registry = ModelRegistry.from_dict(registry_dict)
        
        # Verify the new registry has the same content
        self.assertEqual(set(self.registry.list_models()), set(new_registry.list_models()))
        
        # Check model version exists
        orig_model = self.registry.get_model_version(model_name, version)
        new_model = new_registry.get_model_version(model_name, version)
        
        self.assertIsNotNone(orig_model)
        self.assertIsNotNone(new_model)
        self.assertEqual(orig_model.version, new_model.version)
        self.assertEqual(orig_model.model_path, new_model.model_path)
        self.assertEqual(len(orig_model.shards), len(new_model.shards))
        
        # Check shard if there are any
        if orig_model.shards and new_model.shards:
            orig_shard = orig_model.shards[0]
            new_shard = new_model.shards[0]
            self.assertEqual(orig_shard.shard_id, new_shard.shard_id)
            self.assertEqual(orig_shard.worker_id, new_shard.worker_id)
    
    def test_model_versions(self):
        """Test handling of multiple model versions."""
        # Register multiple versions of the same model
        for version in ["1.0", "1.1", "2.0"]:
            model_data = self.sample_model.copy()
            model_data["version"] = version
            model_data["model_path"] = f"/path/to/model/{version}"
            self.registry.register_model(**model_data)
        
        # Verify all versions are registered
        versions = self.registry.list_versions(self.sample_model["model_name"])
        self.assertEqual(len(versions), 3)
        self.assertIn("1.0", versions)
        self.assertIn("1.1", versions)
        self.assertIn("2.0", versions)
        
        # Verify we can get each version
        for version in ["1.0", "1.1", "2.0"]:
            model = self.registry.get_model_version(self.sample_model["model_name"], version)
            self.assertEqual(model.version, version)
            self.assertEqual(model.model_path, f"/path/to/model/{version}")
    
    def test_worker_models(self):
        """Test tracking of models per worker."""
        # Register models and add shards on different workers
        for i in range(3):
            model_data = self.sample_model.copy()
            model_data["model_name"] = f"model_{i}"
            model_data["version"] = "1.0"
            self.registry.register_model(**model_data)
            
            # Add shard for this model on worker-0
            self.registry.add_shard(
                model_name=model_data["model_name"],
                version=model_data["version"],
                shard_id=0,
                worker_id="worker-0"
            )
            
            # Also add a shard for model_0 on worker-1
            if i == 0:
                self.registry.add_shard(
                    model_name=model_data["model_name"],
                    version=model_data["version"],
                    shard_id=1,
                    worker_id="worker-1"
                )
        
        # Check worker-0 has all 3 models
        worker0_models = self.registry.get_worker_models("worker-0")
        self.assertEqual(len(worker0_models), 3)
        
        # Check worker-1 has 1 model
        worker1_models = self.registry.get_worker_models("worker-1")
        self.assertEqual(len(worker1_models), 1)
        self.assertEqual(worker1_models.pop(), ("model_0", "1.0"))
        
        # Check non-existent worker
        self.assertEqual(len(self.registry.get_worker_models("nonexistent")), 0)


if __name__ == "__main__":
    unittest.main()
