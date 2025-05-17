from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaseAligner(ABC):
    """
    Abstract base class for sentence alignment algorithms.
    """

    @abstractmethod
    def align(self, 
              source_embeddings: np.ndarray, 
              target_embeddings: np.ndarray, 
              similarity_matrix: np.ndarray
             ) -> List[Tuple[int, int]]:
        """
        Aligns source sentences to target sentences based on their embeddings and similarity matrix.

        Args:
            source_embeddings (np.ndarray): Embeddings for source sentences (n_src, dim).
            target_embeddings (np.ndarray): Embeddings for target sentences (n_tgt, dim).
            similarity_matrix (np.ndarray): Precomputed similarity matrix (n_src, n_tgt).

        Returns:
            List[Tuple[int, int]]: A list of tuples, where each tuple (src_idx, tgt_idx)
                                     represents an alignment between a source sentence and a target sentence.
        """
        pass

class SimpleGreedyAligner(BaseAligner):
    """
    A simple greedy alignment strategy.
    For each source sentence, it finds the most similar target sentence based on a threshold.
    This is a basic forward alignment (ArgMax style from source to target).
    """

    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initializes the SimpleGreedyAligner.

        Args:
            similarity_threshold (float): The minimum cosine similarity for an alignment to be considered.
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        self.similarity_threshold = similarity_threshold

    def align(self, 
              source_embeddings: np.ndarray, 
              target_embeddings: np.ndarray, 
              similarity_matrix: np.ndarray
             ) -> List[Tuple[int, int]]:
        """
        Aligns source to target sentences using a greedy approach.

        Args:
            source_embeddings (np.ndarray): Not directly used by this simple version if similarity_matrix is provided,
                                          but kept for API consistency.
            target_embeddings (np.ndarray): Not directly used by this simple version if similarity_matrix is provided.
            similarity_matrix (np.ndarray): A (n_src, n_tgt) matrix of similarity scores.

        Returns:
            List[Tuple[int, int]]: A list of (source_index, target_index) alignment pairs.
        """
        alignments: List[Tuple[int, int]] = []
        
        if similarity_matrix.size == 0:
            if source_embeddings.shape[0] > 0 and target_embeddings.shape[0] > 0:
                 print("Warning: Similarity matrix is empty, but embeddings are not. No alignments can be made.")
            return alignments
        
        num_source_sentences = similarity_matrix.shape[0]
        num_target_sentences = similarity_matrix.shape[1]

        if num_target_sentences == 0:
            # No target sentences to align to
            return alignments

        for i in range(num_source_sentences):
            if num_target_sentences == 0: # Should be caught above, but as a safeguard
                continue
            
            # Find the target sentence with the highest similarity for the current source sentence
            best_target_idx = np.argmax(similarity_matrix[i, :])
            best_similarity_score = similarity_matrix[i, best_target_idx]
            
            if best_similarity_score >= self.similarity_threshold:
                alignments.append((i, best_target_idx))
        
        return alignments

if __name__ == '__main__':
    print("--- Testing SimpleGreedyAligner ---")

    # Dummy embeddings (not used directly by aligner if matrix is given, but good for context)
    s_embeds = np.array([[1,0], [0,1], [0.7,0.7]])
    t_embeds = np.array([[0.9,0.1], [0.1,0.9], [0.6, 0.6], [0.5,0.5]])

    # Example similarity matrix (3 source sentences, 4 target sentences)
    sim_matrix = np.array([
        [0.9, 0.1, 0.6, 0.2],  # Source 0: best match is Target 0 (0.9)
        [0.2, 0.8, 0.5, 0.1],  # Source 1: best match is Target 1 (0.8)
        [0.1, 0.1, 0.2, 0.05]  # Source 2: best match is Target 2 (0.2) - below threshold 0.5
    ])
    print("\nSimilarity Matrix:")
    print(sim_matrix)

    aligner_thresh_0_5 = SimpleGreedyAligner(similarity_threshold=0.5)
    alignments_0_5 = aligner_thresh_0_5.align(s_embeds, t_embeds, sim_matrix)
    print(f"\nAlignments with threshold 0.5: {alignments_0_5}")
    # Expected: [(0,0), (1,1)]
    assert alignments_0_5 == [(0,0), (1,1)], f"Test failed for threshold 0.5. Got {alignments_0_5}"

    aligner_thresh_0_1 = SimpleGreedyAligner(similarity_threshold=0.1)
    alignments_0_1 = aligner_thresh_0_1.align(s_embeds, t_embeds, sim_matrix)
    print(f"Alignments with threshold 0.1: {alignments_0_1}")
    # Expected: [(0,0), (1,1), (2,2)]
    assert alignments_0_1 == [(0,0), (1,1), (2,2)], f"Test failed for threshold 0.1. Got {alignments_0_1}"

    aligner_thresh_0_95 = SimpleGreedyAligner(similarity_threshold=0.95)
    alignments_0_95 = aligner_thresh_0_95.align(s_embeds, t_embeds, sim_matrix)
    print(f"Alignments with threshold 0.95: {alignments_0_95}")
    # Expected: [] (as no score is >= 0.95, except sim_matrix[0,0] which is 0.9 - oh, wait, it is 0.9, not >= 0.95)
    # Correcting expectation for 0.95 threshold: []
    # Let's re-check the test matrix: S0-T0 is 0.9. If threshold is 0.95, it should not be included.
    # Corrected sim_matrix for clarity for 0.95 test
    sim_matrix_strict = np.array([
        [0.96, 0.1, 0.6, 0.2], 
        [0.2, 0.97, 0.5, 0.1],
        [0.94, 0.1, 0.2, 0.05] 
    ])
    print("\nStrict Similarity Matrix for 0.95 threshold test:")
    print(sim_matrix_strict)
    alignments_0_95_strict = aligner_thresh_0_95.align(s_embeds, t_embeds, sim_matrix_strict)
    print(f"Alignments with threshold 0.95 (strict matrix): {alignments_0_95_strict}")
    assert alignments_0_95_strict == [(0,0), (1,1)], f"Test failed for threshold 0.95 (strict). Got {alignments_0_95_strict}"
    
    # Test with no target sentences
    print("\nTesting with no target sentences:")
    empty_t_embeds = np.array([]).reshape(0,2)
    empty_sim_matrix = np.array([]).reshape(3,0)
    alignments_no_target = aligner_thresh_0_5.align(s_embeds, empty_t_embeds, empty_sim_matrix)
    print(f"Alignments with no target sentences: {alignments_no_target}")
    assert alignments_no_target == [], "Test failed for no target sentences."

    print("\nAll SimpleGreedyAligner tests passed.") 