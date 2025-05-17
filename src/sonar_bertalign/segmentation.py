import nltk
from typing import List

# Ensure necessary NLTK data is available
# Users might need to run: nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer models not found. Attempting to download...")
    try:
        nltk.download('punkt', quiet=True)
        print("'punkt' downloaded successfully.")
    except Exception as e:
        print(f"Error downloading 'punkt': {e}")
        print("Please ensure 'punkt' is downloaded manually by running: import nltk; nltk.download('punkt')")

class SentenceSegmenter:
    """
    A class to segment text into sentences using NLTK.
    """

    def __init__(self):
        """
        Initializes the SentenceSegmenter.
        NLTK's default sentence tokenizer (Punkt) is language-adaptive but works best
        for languages it has been trained on. For specific languages, a language-specific
        Punkt model can be loaded if available, e.g., nltk.data.load('tokenizers/punkt/english.pickle').
        However, the generic sent_tokenize is often sufficient for many common languages.
        """
        pass # No specific initialization needed for default NLTK sent_tokenize

    def segment(self, text: str, language: str = 'english') -> List[str]:
        """
        Segments a given text into sentences.

        Args:
            text (str): The input text to segment.
            language (str): The language of the text. NLTK's sent_tokenize
                            supports multiple languages. The 'language' parameter in
                            nltk.sent_tokenize expects lowercase full language names
                            (e.g., 'english', 'french', 'spanish').

        Returns:
            List[str]: A list of sentences.
        """
        if not text:
            return []
        try:
            # NLTK's sent_tokenize can take a language parameter.
            # Ensure the language name matches NLTK's expectations if specified.
            # Common languages are supported by the default Punkt tokenizer.
            return nltk.sent_tokenize(text, language=language)
        except Exception as e:
            print(f"Error segmenting text for language '{language}': {e}")
            # Fallback to basic splitting if tokenization fails, though this is suboptimal.
            # A more robust solution would be to ensure appropriate language models are present.
            if '\n' in text: # If there are newlines, assume paragraph-like structure
                return [line for line in text.splitlines() if line.strip()]
            return [text] # As a last resort, return the whole text as one sentence

if __name__ == '__main__':
    print("--- Testing SentenceSegmenter ---")
    segmenter = SentenceSegmenter()

    text_en = "Hello Mr. Smith. How are you today? The weather is lovely. NLTK is great!"
    print(f"\nOriginal English text:\n{text_en}")
    sentences_en = segmenter.segment(text_en, language='english')
    print("Segmented English sentences:")
    for i, s in enumerate(sentences_en):
        print(f"{i+1}. {s}")

    text_es = "Hola Sra. Pérez. ¿Cómo está usted hoy? El tiempo es agradable. ¡NLTK es genial!"
    print(f"\nOriginal Spanish text:\n{text_es}")
    sentences_es = segmenter.segment(text_es, language='spanish')
    print("Segmented Spanish sentences:")
    for i, s in enumerate(sentences_es):
        print(f"{i+1}. {s}")

    text_multi_line = "This is the first sentence.\nThis is the second one. And this is part of the second too!\nThird sentence here."
    print(f"\nOriginal multi-line text:\n{text_multi_line}")
    sentences_multi_line = segmenter.segment(text_multi_line, language='english')
    print("Segmented multi-line sentences:")
    for i, s in enumerate(sentences_multi_line):
        print(f"{i+1}. {s}")

    # Test with a language for which NLTK might not have specific Punkt parameters by default
    # but the generic tokenizer might still work reasonably well.
    text_ja = "こんにちは。元気ですか。天気は良いですね。"
    print(f"\nOriginal Japanese text (example):\n{text_ja}")
    sentences_ja = segmenter.segment(text_ja, language='japanese') # NLTK's default might not be ideal here
    print("Segmented Japanese sentences (best effort by NLTK default punkt):")
    for i, s in enumerate(sentences_ja):
        print(f"{i+1}. {s}")
    print("(Note: Segmentation for languages like Japanese might require specialized tokenizers for best results.)") 