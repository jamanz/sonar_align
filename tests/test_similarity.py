import pytest
import numpy as np
import sys
import os

# Adjust path to import from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sonar_bertalign.similarity import cosine_similarity_matrix

@pytest.fixture
def embeddings_sample():
    embeds1 = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.1, 0.2, 0.3]
    ], dtype=np.float32)
    embeds2 = np.array([
        [0.1, 0.2, 0.3, 0.4], # Identical to embeds1[0]
        [0.8, 0.7, 0.6, 0.5], # Different
        [0.4, 0.3, 0.2, 0.1]  # Different
    ], dtype=np.float32)
    return embeds1, embeds2

def test_cosine_similarity_basic(embeddings_sample):
    embeds1, embeds2 = embeddings_sample
    similarity = cosine_similarity_matrix(embeds1, embeds2)
    assert similarity.shape == (3, 3)
    assert np.isclose(similarity[0, 0], 1.0) # Similarity with self
    # Check a non-identical pair (value will depend on actual cosine similarity)
    # For [0.1, 0.2, 0.3, 0.4] and [0.8, 0.7, 0.6, 0.5], cos sim is approx 0.7586
    # Formula: (0.08 + 0.14 + 0.18 + 0.2) / (sqrt(0.01+0.04+0.09+0.16) * sqrt(0.64+0.49+0.36+0.25))
    # = 0.6 / (sqrt(0.3) * sqrt(1.74)) = 0.6 / (0.5477 * 1.319) = 0.6 / 0.7224 approx = 0.8305
    # Let's re-calculate: (0.1*0.8 + 0.2*0.7 + 0.3*0.6 + 0.4*0.5) = 0.08 + 0.14 + 0.18 + 0.20 = 0.6
    # ||embeds1[0]|| = sqrt(0.01 + 0.04 + 0.09 + 0.16) = sqrt(0.30) ~= 0.5477225575
    # ||embeds2[1]|| = sqrt(0.64 + 0.49 + 0.36 + 0.25) = sqrt(1.74) ~= 1.319090596
    # cos_sim = 0.6 / (0.5477225575 * 1.319090596) ~= 0.6 / 0.72249 ~= 0.83046
    assert np.isclose(similarity[0, 1], 0.83046, atol=1e-5)

def test_empty_embeddings():
    empty_embeds = np.array([]).reshape(0, 4)
    embeds_valid = np.array([[1,2,3,4]])
    assert cosine_similarity_matrix(empty_embeds, embeds_valid).size == 0
    assert cosine_similarity_matrix(embeds_valid, empty_embeds).size == 0
    assert cosine_similarity_matrix(empty_embeds, empty_embeds).size == 0

def test_mismatched_dimensions():
    embeds1 = np.array([[1,2,3]])
    embeds2 = np.array([[1,2,3,4]])
    result = cosine_similarity_matrix(embeds1, embeds2)
    assert result.size == 0

def test_non_2d_input():
    embeds1_1d = np.array([1,2,3,4])
    embeds2_2d = np.array([[1,2,3,4]])
    assert cosine_similarity_matrix(embeds1_1d, embeds2_2d).size == 0
    assert cosine_similarity_matrix(embeds2_2d, embeds1_1d).size == 0

def test_not_numpy_input(capsys):
    assert cosine_similarity_matrix([[0.1,0.2]], np.array([[0.1,0.2]])).size == 0
    captured = capsys.readouterr()
    assert "Error: Inputs must be NumPy arrays." in captured.out 