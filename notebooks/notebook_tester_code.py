# sonar_bertalign/notebooks/notebook_tester_code.py
"""
This script contains the core logic intended for the
`01_test_sonar_embedder.ipynb` notebook.
It can be imported into a notebook to execute these steps.
"""

import sys
import os
import numpy as np

# --- Path Setup ---
# Note: If the `sonar_bertalign` package is installed (e.g., using `pip install -e .`
# from the project root), this sys.path manipulation might not be strictly necessary
# when this script is imported by a notebook in the same environment.
# However, it's kept here for robustness if the script/notebook context is unusual.

# Assuming this script is in sonar_bertalign/notebooks/
# and we want to import from sonar_bertalign/src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..")
)  # Goes up one level to sonar_bertalign/
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    print(f"Adding {SRC_DIR} to sys.path for module import.")
    sys.path.insert(0, SRC_DIR)  # Insert at the beginning to prioritize project's src

try:
    from sonar_bertalign.embeddings import SonarEmbedder
except ImportError as e:
    print(f"Error importing SonarEmbedder: {e}")
    print(
        "Please ensure that the `sonar_bertalign` package is correctly installed or an editable install"
    )
    print(
        "(`pip install -e .` from the project root) has been performed, and that the Python path is correct."
    )
    print(f"Attempted to add to path: {SRC_DIR}")
    # To aid debugging if run directly:
    print(f"Current sys.path: {sys.path}")
    SonarEmbedder = (
        None  # Define as None so the script can be imported without immediate crash
    )


def run_embedding_test(model_name="sonar_text_LID", device="cpu"):
    """
    Runs the test for SonarEmbedder.
    """
    if SonarEmbedder is None:
        print("SonarEmbedder class not available due to import error. Cannot run test.")
        return

    print("\n--- Initializing SonarEmbedder for test ---       ")
    # Replace 'sonar_text_LID' with an actual SONAR model identifier if available
    # For initial testing, the placeholder model name in the class will be used if SONAR is not installed.
    embedder = SonarEmbedder(model_name_or_path=model_name, device=device)

    sample_sentences_en = [
        "Hello world!",
        "This is a test sentence for SONAR embeddings.",
    ]

    sample_sentences_es = [
        "¡Hola Mundo!",
        "Esta es una frase de prueba para las incrustaciones SONAR.",
    ]

    print("\n--- Encoding English sentences ---")
    embeddings_en = embedder.encode(sample_sentences_en)
    if embeddings_en.size > 0:
        print(f"English Embeddings Shape: {embeddings_en.shape}")
        # print(embeddings_en)
    else:
        print(
            "Failed to generate English embeddings. SONAR might not be installed, model path incorrect, or check SonarEmbedder logs."
        )

    print("\n--- Encoding Spanish sentences ---")
    embeddings_es = embedder.encode(sample_sentences_es)
    if embeddings_es.size > 0:
        print(f"Spanish Embeddings Shape: {embeddings_es.shape}")
        # print(embeddings_es)
    else:
        print(
            "Failed to generate Spanish embeddings. SONAR might not be installed, model path incorrect, or check SonarEmbedder logs."
        )

    print_next_steps()


def print_next_steps():
    print("\n" + "-" * 50)
    print("### Next Steps for SONAR Integration:")
    print(
        "1. Ensure `sonar-space` and `fairseq2` are correctly installed in your environment."
    )
    print(
        "   (e.g., by running `pip install -e .[dev]` from the project root - see README.md)"
    )
    print(
        "2. Obtain a valid SONAR text model name/path (e.g., from the SONAR model card or documentation). Some examples could be 'sonar_text_LID' or 'sonar_text_encoder_eng_rus' etc."
    )
    print(
        "3. When running the test, update the `model_name_or_path` argument for `SonarEmbedder` or `run_embedding_test` function."
    )
    print(
        "4. Verify and update the actual SONAR API calls within `src/sonar_bertalign/embeddings.py`:"
    )
    print("   - The `load_text_encoder` import and usage.")
    print(
        "   - The specific method for encoding text (e.g., `encode_text_batch` or similar)."
    )
    print(
        "   - Any necessary pre-processing (tokenization) if not handled by the SONAR model's encode method itself."
    )
    print("-" * 50 + "\n")


if __name__ == "__main__":
    print("Running SonarEmbedder test directly from script...")
    # Example: To test with a specific model from command line (if SONAR is installed)
    # You would typically call this from a notebook or another script.
    run_embedding_test()
    # To test with a specific model:
    # run_embedding_test(model_name="your_actual_sonar_model_name_here")
