import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity between two sets of embeddings.

    Args:
        embeddings1 (np.ndarray): A 2D NumPy array of shape (n, dim),
                                  where n is the number of sentences and dim is the embedding dimension.
        embeddings2 (np.ndarray): A 2D NumPy array of shape (m, dim),
                                  where m is the number of sentences and dim is the embedding dimension.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n, m) containing the cosine similarity scores.
                    Returns an empty array if inputs are invalid (e.g., not 2D or incompatible dimensions).
    """
    if not isinstance(embeddings1, np.ndarray) or not isinstance(embeddings2, np.ndarray):
        print("Error: Inputs must be NumPy arrays.")
        return np.array([])
    
    if embeddings1.ndim != 2 or embeddings2.ndim != 2:
        print("Error: Input embeddings must be 2D arrays.")
        return np.array([])

    if embeddings1.shape[0] == 0 or embeddings2.shape[0] == 0:
        # print("Warning: One or both embedding sets are empty. Returning empty similarity matrix.")
        return np.array([]) # Or handle as per desired behavior for empty inputs

    if embeddings1.shape[1] != embeddings2.shape[1]:
        print(f"Error: Embedding dimensions do not match. Got {embeddings1.shape[1]} and {embeddings2.shape[1]}.")
        return np.array([])

    try:
        # sklearn.metrics.pairwise.cosine_similarity is efficient and handles normalization.
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        return similarity_matrix
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        return np.array([])

if __name__ == '__main__':
    print("--- Testing cosine_similarity_matrix ---")

    # Example Embeddings
    # Typically, embedding dimension (dim) would be larger (e.g., 768, 1024)
    dim = 4 
    embeds1 = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.1, 0.2, 0.3]
    ])

    embeds2 = np.array([
        [0.1, 0.2, 0.3, 0.4], # Identical to embeds1[0]
        [0.8, 0.7, 0.6, 0.5], # Different
        [0.4, 0.3, 0.2, 0.1]  # Different
    ])

    print("\nEmbeddings 1:")
    print(embeds1)
    print("\nEmbeddings 2:")
    print(embeds2)

    similarity = cosine_similarity_matrix(embeds1, embeds2)
    print("\nCosine Similarity Matrix:")
    if similarity.size > 0:
        print(similarity)
        # Expected: First row, first col should be close to 1.0
        assert np.isclose(similarity[0, 0], 1.0), "Similarity of identical vectors should be 1.0"
    else:
        print("Similarity matrix is empty (error occurred or empty input).")

    # Test with empty arrays
    print("\nTesting with one empty array:")
    empty_embeds = np.array([]).reshape(0,dim)
    similarity_empty = cosine_similarity_matrix(embeds1, empty_embeds)
    if similarity_empty.size == 0:
        print("Correctly returned empty matrix for empty input.")
    else:
        print(f"Unexpected output for empty input: {similarity_empty}")

    similarity_empty_2 = cosine_similarity_matrix(empty_embeds, embeds2)
    if similarity_empty_2.size == 0:
        print("Correctly returned empty matrix for empty input (case 2).")
    else:
        print(f"Unexpected output for empty input (case 2): {similarity_empty_2}")

    # Test with mismatched dimensions
    print("\nTesting with mismatched dimensions:")
    embeds_wrong_dim = np.array([[0.1, 0.2, 0.3]])
    similarity_wrong_dim = cosine_similarity_matrix(embeds1, embeds_wrong_dim)
    if similarity_wrong_dim.size == 0:
        print("Correctly returned empty matrix for mismatched dimensions.")
    else:
        print(f"Unexpected output for mismatched dimensions: {similarity_wrong_dim}")

    print("\nTesting with non-2D arrays:")
    embeds_1d = np.array([0.1, 0.2, 0.3, 0.4])
    similarity_1d = cosine_similarity_matrix(embeds_1d, embeds2)
    if similarity_1d.size == 0:
        print("Correctly returned empty matrix for non-2D input.")
    else:
        print(f"Unexpected output for non-2D input: {similarity_1d}") 