#!/usr/bin/env python
import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# Add src to path to allow direct import of sonar_bertalign modules
# Assuming the script is in sonar_bertalign/scripts/
# Note: If the package is installed (e.g., using `pip install -e .`),
# this sys.path manipulation might not be strictly necessary as 
# `sonar_bertalign` should be in the Python path.
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

try:
    from sonar_bertalign.segmentation import SentenceSegmenter
    from sonar_bertalign.embeddings import SonarEmbedder # Placeholder, ensure SONAR is configured
    from sonar_bertalign.similarity import cosine_similarity_matrix
    from sonar_bertalign.alignment import SimpleGreedyAligner
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Ensure PYTHONPATH is correctly set or run from the project's root directory after installing the package.")
    print(f"Attempted to add to sys.path: {MODULE_PATH}")
    sys.exit(1)

def read_file_content(file_path: str) -> str:
    """Reads the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def write_alignments(aligned_pairs: list[tuple[str, str]], output_file: str):
    """Writes aligned sentence pairs to a TSV file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for src_sent, tgt_sent in tqdm(aligned_pairs, desc="Writing alignments"):
                f.write(f"{src_sent}\t{tgt_sent}\n")
        print(f"Alignments successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing alignments to {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Align parallel texts using SONAR embeddings.")
    parser.add_argument("--source_file", required=True, help="Path to the source language text file.")
    parser.add_argument("--target_file", required=True, help="Path to the target language text file.")
    parser.add_argument("--source_lang", required=True, help="Source language code (e.g., 'en', 'es', 'fr'). NLTK compatible.")
    parser.add_argument("--target_lang", required=True, help="Target language code (e.g., 'en', 'es', 'fr'). NLTK compatible.")
    parser.add_argument("--output_file", required=True, help="Path to save the aligned sentence pairs (TSV format).")
    parser.add_argument("--model_name", default="sonar_text_LID", help="Name or path of the SONAR model to use. (default: sonar_text_LID - a generic SONAR model, replace with a specific one like 'sonar_text_encoder_eng_rus' if available and configured).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use for embeddings ('cpu' or 'cuda').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for SONAR embedding generation.")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="Similarity threshold for alignment.")

    args = parser.parse_args()

    print("--- Starting Text Alignment Pipeline ---")
    start_time = time.time()

    # 1. Initialize components
    print("\n1. Initializing components...")
    segmenter = SentenceSegmenter()
    # Ensure the SonarEmbedder uses a valid model name for your setup
    embedder = SonarEmbedder(model_name_or_path=args.model_name, device=args.device)
    aligner = SimpleGreedyAligner(similarity_threshold=args.similarity_threshold)

    # 2. Read and segment texts
    print("\n2. Reading and segmenting texts...")
    source_text = read_file_content(args.source_file)
    target_text = read_file_content(args.target_file)

    source_sents = segmenter.segment(source_text, language=args.source_lang)
    target_sents = segmenter.segment(target_text, language=args.target_lang)

    if not source_sents or not target_sents:
        print("Error: No sentences found in one or both input files. Exiting.")
        sys.exit(1)
    
    print(f"Segmented source text into {len(source_sents)} sentences.")
    print(f"Segmented target text into {len(target_sents)} sentences.")

    # 3. Generate embeddings
    # Note: The SonarEmbedder.encode might print placeholder messages if SONAR isn't fully set up.
    print("\n3. Generating embeddings (this might take a while for large files)...")
    print(f"Using SONAR model: {args.model_name} on device: {args.device}")
    
    source_embeddings = np.array([])
    if source_sents:
        source_embeddings = embedder.encode(source_sents, batch_size=args.batch_size)
    
    target_embeddings = np.array([])
    if target_sents:
        target_embeddings = embedder.encode(target_sents, batch_size=args.batch_size)

    if source_embeddings.size == 0 or target_embeddings.size == 0:
        print("Error: Failed to generate embeddings for one or both texts. Exiting.")
        print("Please check SONAR model availability and configuration in embeddings.py")
        sys.exit(1)

    print(f"Generated source embeddings with shape: {source_embeddings.shape}")
    print(f"Generated target embeddings with shape: {target_embeddings.shape}")

    # 4. Compute similarity matrix
    print("\n4. Computing similarity matrix...")
    similarity_mat = cosine_similarity_matrix(source_embeddings, target_embeddings)

    if similarity_mat.size == 0 and (source_embeddings.size > 0 and target_embeddings.size > 0) :
        print("Error: Failed to compute similarity matrix, though embeddings were generated. Exiting.")
        sys.exit(1)
    elif similarity_mat.size == 0:
        print("Warning: Similarity matrix is empty (likely due to empty embeddings). No alignments possible.")
        # No need to exit, aligner will handle empty matrix
    else:
        print(f"Computed similarity matrix with shape: {similarity_mat.shape}")

    # 5. Align sentences
    print("\n5. Aligning sentences...")
    alignment_indices = aligner.align(source_embeddings, target_embeddings, similarity_mat)
    print(f"Found {len(alignment_indices)} alignments with threshold {args.similarity_threshold}.")

    # 6. Prepare and write output
    print("\n6. Preparing and writing output...")
    aligned_sentence_pairs = []
    for src_idx, tgt_idx in alignment_indices:
        if 0 <= src_idx < len(source_sents) and 0 <= tgt_idx < len(target_sents):
            aligned_sentence_pairs.append((source_sents[src_idx], target_sents[tgt_idx]))
        else:
            print(f"Warning: Invalid alignment index: src={src_idx}, tgt={tgt_idx}. Skipping.")

    write_alignments(aligned_sentence_pairs, args.output_file)

    end_time = time.time()
    print(f"\n--- Alignment pipeline completed in {end_time - start_time:.2f} seconds. ---")

if __name__ == "__main__":
    # Example usage (manual setup for testing if not run from command line):
    # Create dummy files first if you want to test this part directly.
    # with open("source.txt", "w") as f: f.write("Hello world.\nThis is a test.")
    # with open("target.txt", "w") as f: f.write("Bonjour le monde.\nCeci est un test.")
    # sys.argv = [
    #     __file__,
    #     "--source_file", "source.txt",
    #     "--target_file", "target.txt",
    #     "--source_lang", "english",
    #     "--target_lang", "french",
    #     "--output_file", "aligned_output.tsv",
    #     "--model_name", "sonar_text_LID", # Placeholder, use actual model if available
    #     "--device", "cpu",
    #     "--similarity_threshold", "0.2" # Lower threshold for dummy example
    # ]
    main() 