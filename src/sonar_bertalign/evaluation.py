from typing import List, Set, Tuple, Dict, Any

def load_gold_alignments(file_path: str, delimiter: str = '\t') -> Set[Tuple[str, str]]:
    """
    Loads gold standard alignments from a file.
    Assumes each line contains a source sentence and a target sentence separated by a delimiter.

    Args:
        file_path (str): Path to the gold standard alignment file.
        delimiter (str): The delimiter used to separate source and target sentences.
                         Defaults to tab (\t).

    Returns:
        Set[Tuple[str, str]]: A set of (source_sentence, target_sentence) tuples.
                                Returns an empty set if the file is not found or an error occurs.
    """
    gold_alignments: Set[Tuple[str, str]] = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split(delimiter)
                if len(parts) == 2:
                    gold_alignments.add((parts[0].strip(), parts[1].strip()))
                elif line.strip(): # Non-empty line that doesn't split into 2 parts
                    print(f"Warning: Gold alignment file '{file_path}', line {line_num+1}: Expected 2 parts, got {len(parts)}. Line: '{line.strip()}'")
    except FileNotFoundError:
        print(f"Error: Gold alignment file not found at {file_path}")
    except Exception as e:
        print(f"Error reading gold alignment file {file_path}: {e}")
    return gold_alignments

def calculate_metrics(predicted_alignments: List[Tuple[str, str]], 
                        gold_alignments: Set[Tuple[str, str]]) -> Dict[str, float]:
    """
    Calculates precision, recall, and F1-score for sentence alignments.

    Args:
        predicted_alignments (List[Tuple[str, str]]): A list of (source, target) sentence pairs predicted by the aligner.
        gold_alignments (Set[Tuple[str, str]]): A set of gold standard (source, target) sentence pairs.

    Returns:
        Dict[str, float]: A dictionary containing precision, recall, and F1-score.
                          Returns all metrics as 0.0 if either predicted or gold is empty.
    """
    # Convert predicted list to a set for efficient intersection
    predicted_set = set(predicted_alignments)

    if not predicted_set or not gold_alignments:
        # If either is empty, metrics are typically 0, or could be undefined/NaN depending on preference.
        # Returning 0 for simplicity here.
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "num_predicted": len(predicted_set), "num_gold": len(gold_alignments), "num_correct": 0}

    # True positives: alignments present in both predicted and gold sets
    true_positives = len(predicted_set.intersection(gold_alignments))

    # Precision = TP / (TP + FP) = TP / |Predicted|
    precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0.0

    # Recall = TP / (TP + FN) = TP / |Gold|
    recall = true_positives / len(gold_alignments) if len(gold_alignments) > 0 else 0.0

    # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "num_predicted": len(predicted_set),
        "num_gold": len(gold_alignments),
        "num_correct": true_positives
    }

if __name__ == '__main__':
    print("--- Testing Evaluation Module ---")

    # Create dummy gold and predicted data for testing
    dummy_gold_path = "dummy_gold.txt"
    dummy_predicted_path = "dummy_predicted.txt" # For load_gold_alignments test

    gold_data = [
        ("Hello world", "Bonjour le monde"),
        ("How are you?", "Comment allez-vous?"),
        ("Good morning", "Bonjour"),
        ("Thank you", "Merci")
    ]

    predicted_data = [
        ("Hello world", "Bonjour le monde"), # Correct
        ("How are you?", "Comment ça va?"),   # Incorrect translation for this exact match test
        ("Good morning", "Bonjour"),         # Correct
        ("See you later", "Au revoir")       # Not in gold
    ]

    # Test load_gold_alignments
    with open(dummy_gold_path, 'w', encoding='utf-8') as f:
        for s, t in gold_data:
            f.write(f"{s}\t{t}\n")
        f.write("This line has only one part\n") # Malformed line
        f.write("One\tTwo\tThree\n") # Malformed line

    loaded_gold = load_gold_alignments(dummy_gold_path)
    print(f"\nLoaded gold alignments (expected {len(gold_data)}): {len(loaded_gold)}")
    # print(f"Loaded: {loaded_gold}")
    assert loaded_gold == set(gold_data), "load_gold_alignments failed to load correctly"
    print("load_gold_alignments test passed.")

    # Test calculate_metrics
    metrics = calculate_metrics(predicted_data, set(gold_data))
    print("\nCalculated Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Expected values for the dummy data:
    # Predicted: 4, Gold: 4
    # Correct (TP): ("Hello world", "Bonjour le monde"), ("Good morning", "Bonjour") -> 2
    # Precision = 2 / 4 = 0.5
    # Recall = 2 / 4 = 0.5
    # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 2 * 0.25 / 1.0 = 0.5
    assert metrics["num_correct"] == 2, "Metric calculation error (num_correct)"
    assert abs(metrics["precision"] - 0.5) < 1e-9, "Metric calculation error (precision)"
    assert abs(metrics["recall"] - 0.5) < 1e-9, "Metric calculation error (recall)"
    assert abs(metrics["f1_score"] - 0.5) < 1e-9, "Metric calculation error (f1_score)"
    print("calculate_metrics test passed.")

    # Test with empty predicted
    metrics_empty_pred = calculate_metrics([], set(gold_data))
    print(f"\nMetrics with empty predicted: {metrics_empty_pred}")
    assert metrics_empty_pred["f1_score"] == 0.0, "Empty predicted test failed"

    # Test with empty gold
    metrics_empty_gold = calculate_metrics(predicted_data, set())
    print(f"Metrics with empty gold: {metrics_empty_gold}")
    assert metrics_empty_gold["f1_score"] == 0.0, "Empty gold test failed"

    # Clean up dummy files
    import os
    try:
        os.remove(dummy_gold_path)
    except OSError:
        pass

    print("\nAll evaluation module tests passed.") 