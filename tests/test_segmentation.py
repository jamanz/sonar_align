import pytest
import sys
import os

# Adjust path to import from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sonar_bertalign.segmentation import SentenceSegmenter

@pytest.fixture
def segmenter():
    return SentenceSegmenter()

def test_segment_english(segmenter):
    text = "Hello Mr. Smith. How are you today? The weather is lovely."
    expected = [
        "Hello Mr. Smith.",
        "How are you today?",
        "The weather is lovely."
    ]
    assert segmenter.segment(text, language='english') == expected

def test_segment_spanish(segmenter):
    text = "Hola Sra. Pérez. ¿Cómo está usted hoy? El tiempo es agradable."
    expected = [
        "Hola Sra. Pérez.",
        "¿Cómo está usted hoy?",
        "El tiempo es agradable."
    ]
    assert segmenter.segment(text, language='spanish') == expected

def test_segment_empty_string(segmenter):
    assert segmenter.segment("", language='english') == []

def test_segment_single_sentence(segmenter):
    text = "This is a single sentence without a period"
    expected = ["This is a single sentence without a period"]
    assert segmenter.segment(text, language='english') == expected

def test_segment_newline_fallback(segmenter, capsys, monkeypatch):
    # Temporarily make nltk.sent_tokenize raise an error to test fallback
    def mock_sent_tokenize(*args, **kwargs):
        raise Exception("Mock NLTK failure")
    
    monkeypatch.setattr("nltk.sent_tokenize", mock_sent_tokenize)
    
    text_with_newlines = "First line.\nSecond line."
    expected_fallback = ["First line.", "Second line."]
    result = segmenter.segment(text_with_newlines, language='english')
    captured = capsys.readouterr()
    assert "Error segmenting text" in captured.out # Check for error message
    assert result == expected_fallback

    text_no_newlines = "Single line no newline fallback"
    expected_single = ["Single line no newline fallback"]
    result_single = segmenter.segment(text_no_newlines, language='english')
    assert result_single == expected_single 