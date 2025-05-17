import pytest
import numpy as np
import sys
import os
from unittest import mock

# Adjust path to import from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import SonarEmbedder after path adjustment
from sonar_bertalign.embeddings import SonarEmbedder

# Mock the SONAR library's load_text_encoder
# This allows testing SonarEmbedder without actual SONAR installation or models.
@pytest.fixture
def mock_sonar_encoder(monkeypatch):
    class MockSonarModel:
        def __init__(self, model_name, device):
            self.model_name = model_name
            self.device = device
            print(f"MockSonarModel initialized with {model_name} on {device}")

        def encode_text_batch(self, sentences):
            # Simulate returning embeddings: (batch_size, embedding_dim)
            # Using a fixed embedding_dim for test consistency, e.g., 1024 (common for SONAR)
            print(f"MockSonarModel: Encoding batch of {len(sentences)} sentences.")
            if not sentences:
                return np.array([]) # Return empty numpy array for empty list of sentences
            # In a real scenario, this would be a torch tensor
            # For mock, directly return numpy array as SonarEmbedder converts
            return np.random.rand(len(sentences), 1024).astype(np.float32)
        
        def to(self, device):
            self.device = device
            print(f"MockSonarModel: Moved to {device}")
            return self

    def mock_load_text_encoder(model_name_or_path, device='cpu'):
        return MockSonarModel(model_name_or_path, device)

    monkeypatch.setattr("sonar_bertalign.embeddings.load_text_encoder", mock_load_text_encoder)
    # Also ensure SONAR_AVAILABLE is True for these tests if it's checked internally
    monkeypatch.setattr("sonar_bertalign.embeddings.SONAR_AVAILABLE", True)
    return mock_load_text_encoder

def test_sonar_embedder_init(mock_sonar_encoder, capsys):
    embedder = SonarEmbedder(model_name_or_path="test_model", device="cpu")
    assert embedder.device == "cpu"
    assert embedder.model is not None
    assert embedder.model.model_name == "test_model"
    captured = capsys.readouterr()
    assert "MockSonarModel initialized with test_model on cpu" in captured.out

def test_sonar_embedder_encode(mock_sonar_encoder):
    embedder = SonarEmbedder(model_name_or_path="test_model", device="cpu")
    sentences = ["Hello world", "This is a test."]
    embeddings = embedder.encode(sentences, batch_size=1)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 1024) # 2 sentences, 1024 dim from mock
    assert embeddings.dtype == np.float32

def test_sonar_embedder_encode_empty_list(mock_sonar_encoder):
    embedder = SonarEmbedder(model_name_or_path="test_model", device="cpu")
    sentences = []
    embeddings = embedder.encode(sentences)
    assert embeddings.shape == (0,) or embeddings.shape == (0, 1024) # Depending on np.vstack behavior with empty list

def test_sonar_embedder_batching(mock_sonar_encoder, capsys):
    embedder = SonarEmbedder(model_name_or_path="test_model", device="cpu")
    sentences = ["s1", "s2", "s3", "s4", "s5"]
    # Mock model's encode_text_batch to track calls
    embedder.model.encode_text_batch = mock.MagicMock(side_effect=lambda s: np.random.rand(len(s), 1024).astype(np.float32))
    
    embeddings = embedder.encode(sentences, batch_size=2)
    assert embeddings.shape == (5, 1024)
    # Expected calls: [s1,s2], [s3,s4], [s5]
    assert embedder.model.encode_text_batch.call_count == 3 
    embedder.model.encode_text_batch.assert_any_call(["s1", "s2"])
    embedder.model.encode_text_batch.assert_any_call(["s3", "s4"])
    embedder.model.encode_text_batch.assert_any_call(["s5"])

# Test case for when SONAR_AVAILABLE is False (simulating SONAR not installed)
def test_sonar_unavailable(monkeypatch, capsys):
    # Mock load_text_encoder to return a dummy that will cause `hasattr` check to fail or use the placeholder
    class DummyPlaceholderEncoder:
        def __call__(self, *args, **kwargs):
            raise NotImplementedError("SONAR not installed (placeholder)")
        def to(self, device):
            return self
        # Crucially, does NOT have 'encode_text_batch'

    def failing_load_text_encoder(model_name_or_path, device):
        print(f"Placeholder: Would load SONAR model {model_name_or_path} on {device} (simulating unavailable)")
        return DummyPlaceholderEncoder()

    monkeypatch.setattr("sonar_bertalign.embeddings.load_text_encoder", failing_load_text_encoder)
    monkeypatch.setattr("sonar_bertalign.embeddings.SONAR_AVAILABLE", False)
    
    embedder = SonarEmbedder(model_name_or_path="any_model")
    sentences = ["Hello world"]
    embeddings = embedder.encode(sentences)
    
    captured = capsys.readouterr()
    assert "SONAR model not available" in captured.out or \
           "Warning: SONAR library not found" in captured.out # Check init message too
    assert embeddings.size == 0 