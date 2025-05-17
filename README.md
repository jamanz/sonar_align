# SONAR-BERTAlign

This project adapts the BERTAlign methodology to use Meta's SONAR embeddings for multilingual and multimodal parallel text alignment.

## Project Structure

- `src/sonar_bertalign/`: Main source code
- `data/`: Sample and test data
- `notebooks/`: Jupyter notebooks for experimentation
- `scripts/`: Command-line interface scripts
- `tests/`: Unit tests

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd sonar_bertalign
    ```

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the package and its dependencies:**
    This project uses `pyproject.toml` for packaging. You can install it in editable mode, which is useful for development as changes to the source code will be immediately reflected without needing to reinstall.
    ```bash
    pip install -e .[dev]
    ```
    The `.[dev]` part also installs the development dependencies (like `black`, `isort`, `pytest`, `jupyterlab`). If you only need runtime dependencies, you can use `pip install -e .`.

4.  **Download NLTK data (if prompted or on first use of segmentation):
    ```python
    import nltk
    nltk.download('punkt')
    ```
    The `SentenceSegmenter` module will attempt to download this automatically if not found, but manual download can also be performed.

5.  **SONAR Model Setup:**
    - Ensure you have access to the SONAR models you intend to use.
    - The `SonarEmbedder` class currently uses placeholder model names and logic. You will need to update `src/sonar_bertalign/embeddings.py` with actual SONAR model identifiers and verify the model loading and encoding API calls against the `sonar-space` library documentation.

## Usage

Once installed, you can run the alignment script (example):

```bash
python scripts/align_texts.py --source_file path/to/your/source.txt --target_file path/to/your/target.txt --source_lang en --target_lang fr --output_file aligned_output.tsv --model_name <your_sonar_model_name>
```

Refer to the script's help for more options:
```bash
python scripts/align_texts.py --help
```

You can also explore the functionalities through the Jupyter notebooks in the `notebooks/` directory.

## Project Structure

- `pyproject.toml`: Project configuration and dependencies.
- `README.md`: This file.
- `src/sonar_bertalign/`: Main source code for the `sonar_bertalign` package.
  - `embeddings.py`: SONAR embedding generation.
  - `segmentation.py`: Sentence segmentation.
  - `similarity.py`: Similarity computation.
  - `alignment.py`: Alignment algorithms.
  - `idiom_handling.py`: Placeholder for figurative language.
- `scripts/`: Command-line interface scripts (e.g., `align_texts.py`).
- `notebooks/`: Jupyter notebooks for experimentation (e.g., testing `SonarEmbedder`).
- `data/`: Sample and test data (to be added).
- `tests/`: Unit tests (to be added). 