from typing import List, Dict

# Placeholder for a more sophisticated idiom dictionary or detection mechanism.
# This could be populated from external resources or learned models in the future.
KNOWN_IDIOMS: Dict[str, List[str]] = {
    "en": [
        "kick the bucket",
        "break a leg",
        "spill the beans",
        "a piece of cake"
    ],
    "es": [
        "tomar el pelo",
        "estar en las nubes",
        "no tener pelos en la lengua"
    ],
    "fr": [
        "poser un lapin",
        "avoir un chat dans la gorge",
        "raconter des salades"
    ]
    # Add more languages and idioms as needed
}

def is_potential_idiom(sentence: str, lang: str, custom_idiom_list: List[str] = None) -> bool:
    """
    Checks if a sentence potentially contains a known idiom for a given language.
    This is a very basic placeholder implementation based on substring matching.

    Args:
        sentence (str): The sentence to check.
        lang (str): The language code (e.g., 'en', 'es').
        custom_idiom_list (List[str], optional): A user-provided list of idioms to check against.
                                                  If provided, this list is used instead of the default KNOWN_IDIOMS.

    Returns:
        bool: True if a known idiom is found (case-insensitive) in the sentence, False otherwise.
    """
    sentence_lower = sentence.lower()
    idioms_to_check = []

    if custom_idiom_list:
        idioms_to_check = [idiom.lower() for idiom in custom_idiom_list]
    elif lang in KNOWN_IDIOMS:
        idioms_to_check = [idiom.lower() for idiom in KNOWN_IDIOMS[lang]]
    else:
        # No known idioms for this language in the default list
        return False

    for idiom in idioms_to_check:
        if idiom in sentence_lower:
            return True
    return False

# Further design considerations for idiom handling within the alignment process:
# 1. Specialized Similarity Metrics:
#    - If an idiom is detected, the standard cosine similarity might be misleading.
#    - Could we use a different metric? Or adjust the threshold?
#    - Potentially use a bilingual idiom dictionary to find equivalent idioms and then compare those,
#      or compare the non-idiomatic meaning if available.
#
# 2. Alignment Strategy Adjustment:
#    - If source_sentence_A is an idiom, and target_sentence_B is its known translation,
#      their direct SONAR embeddings might not be very close if the translation is non-literal.
#    - The alignment algorithm could have a special rule: if is_potential_idiom(src_sent) and
#      is_potential_idiom(tgt_sent) (and they are known equivalents), boost their similarity.
#    - Or, if an idiom is detected in the source, the aligner could search for a broader range of
#      target sentences or use contextual clues more heavily.
#
# 3. External Knowledge Bases:
#    - Integrate with resources like Wiktionary, OmegaWiki, or specialized idiom translation databases.
#
# 4. User-Provided Dictionaries:
#    - Allow users to provide their own bilingual idiom dictionaries to improve accuracy for specific domains.
#
# 5. Logging and Flagging:
#    - At a minimum, alignments involving potential idioms could be flagged in the output for manual review.

if __name__ == '__main__':
    print("--- Testing Figurative Language Handling (Placeholder) ---")

    test_sentence_en_idiom = "He decided to spill the beans about the project."
    test_sentence_en_no_idiom = "He told everyone about the project."
    test_sentence_es_idiom = "Me estás tomando el pelo, ¿verdad?"
    test_sentence_fr_idiom = "Il m'a posé un lapin hier soir."
    test_sentence_unknown_lang = "This is a test."

    print(f"\nSentence: '{test_sentence_en_idiom}' (en)")
    print(f"Contains idiom? {is_potential_idiom(test_sentence_en_idiom, 'en')}") # Expected: True

    print(f"\nSentence: '{test_sentence_en_no_idiom}' (en)")
    print(f"Contains idiom? {is_potential_idiom(test_sentence_en_no_idiom, 'en')}") # Expected: False

    print(f"\nSentence: '{test_sentence_es_idiom}' (es)")
    print(f"Contains idiom? {is_potential_idiom(test_sentence_es_idiom, 'es')}") # Expected: True

    print(f"\nSentence: '{test_sentence_fr_idiom}' (fr)")
    print(f"Contains idiom? {is_potential_idiom(test_sentence_fr_idiom, 'fr')}") # Expected: True

    print(f"\nSentence: '{test_sentence_unknown_lang}' (de) - no idioms defined")
    print(f"Contains idiom? {is_potential_idiom(test_sentence_unknown_lang, 'de')}") # Expected: False

    custom_idioms = ["fly off the handle", "miss the boat"]
    test_sentence_custom_idiom = "He tends to fly off the handle easily."
    print(f"\nSentence: '{test_sentence_custom_idiom}' (en) with custom list")
    print(f"Contains idiom (custom)? {is_potential_idiom(test_sentence_custom_idiom, 'en', custom_idiom_list=custom_idioms)}") # Expected: True 