"""Features for Wikipedia Vandalism Detection
This script extracts features from Wikipedia edits to help compute features
for detecting vandalism.
Based on the Wikipedia Article: https://en.wikipedia.org/wiki/User:ClueBot_NG
Based on the Wikipedia Article: https://en.wikipedia.org/wiki/Wikipedia:Vandalism
Based on the paper: "Wikipedia Vandalism Detection Through Machine Learning:
Feature Review and New Proposals Lab Report for PAN at CLEF 2010"
Link: https://arxiv.org/pdf/1210.5560
Author: Santiago M. Mola Velasco
Email: santiago.mola@bitsnbrains.net
License: Undetermined

"""

"""
This script extracts features from Wikipedia edits to help computes features
    for detect vandalism.

Features include:
- Number of characters in the edit
- Number of words in the edit
- Number of links in the edit
- Number of templates in the edit
- Number of categories in the edit
- Number of references in the edit
- Number of images in the edit
- Number of sections in the edit
- Number of external links in the edit
- Number of internal links in the edit
- Number of redirects in the edit
- Number of new pages created in the edit
- Number of pages deleted in the edit
- Number of pages moved in the edit
- Number of pages restored in the edit
- Number of pages reverted in the edit
- Number of pages merged in the edit
- Number of pages split in the edit
- Number of pages renamed in the edit
- Number of pages unlinked in the edit
- Number of pages linked in the edit
- Number of pages linked to in the edit
- Number of pages linked from in the edit

ClueBot_NG Features:
- Old text and new text. The old and new without any preprocess.
- Case-sensitive inserted words. The set of inserted words.
- Inserted words. The set of inserted words, and converted to lowercase.
- Concatenated inserted words. All inserted words concatenated and separated with
spaces. This is defined both case-sensitive and insensitive.
- Inserted text. Inserted lines as reported by the diff algorithm
- Anonymous: Wether the editor is anonymous or not. Vandals are likely to be
    anonymous. This feature is used in a way or another in most antivandalism
    working bots such as ClueBot and AVBOT. In the PAN-WVC10 training set
    (Potthast, 2010) anonymous edits represent 29% of the regular edits and 87%
    of vandalism edits.
- Comment length: Length in characters of the edit summary. Long comments might
    indicate regular editing and short or blank ones might suggest vandalism,
    however, this feature is quite weak, since leaving an empty comment in
    regular editing is a common practice.
- Upper to lower ratio† Uppercase to lowercase letters ratio,
    i.e., (1+|upper|) / (1+|lower|).
    Vandals often do not follow capitalization rules, writing everything in lowercase or in uppercase.
- Upper to all ratio† Uppercase letters to all letters to ratio,
    i.e., (1+|upper|) / (1+|lower|+|upper|).
- Digit ratio Digit to all characters ratio,
    i.e., (1+|digit|) / (1+|all|).
    This feature helps to spot minor edits that only change numbers, which
    might help to find some cases of subtle vandalism where the vandal changes
    arbitrarily a date or a number to introduce misinformation.
- Non-alphanumeric ratio Non-alphanumeric to all characters ratio,
i.e., (1+|nonalphanumeric|) / (1+|all|).
    An excess of non-alphanumeric characters in short texts might indicate
    excessive use of exclamation marks or emoticons.
- Character diversity Measure of different characters compared to the length of
    inserted text, given by the expression: length^( 1 / (different chars) ).
    This feature helps to spot random keyboard hits and other non-sense. It
    should take into account QWERTY keyboard layout in the future.
- Character distribution† Kullback-Leibler divergence of the character
    distribution of the inserted text with respect the expectation. Useful to
    detect non-sense.
- Compressibility† Compression rate of inserted text using the LZW algorithm. 2
    Useful to detect non-sense, repetitions of the same character or words, etc.
- Size increment Absolute increment of size, i.e., |new| - |old|.
    The value of this feature is already well-established. ClueBot uses various
    thresholds of size increment for its heuristics, e.g., a big size decrement
    is considered an indicator of blanking.
- Size ratio: Size of the new revision relative to the old revision,
    i.e., (1+|new|) / (1+|old|).
    Complements size increment.
- Average term frequency: Average relative frequency of inserted words in the
    new revision. In long and well-established articles too many words that do
    not appear in the rest of the article indicates that the edit might be
    including non-sense or non-related content.
- Longest word: Length of the longest word in inserted text.
    Useful to detect non-sense.
- Longest character sequence: Longest consecutive sequence of the same
    character in the inserted text. Long sequences of the same character are
    frequent in vandalism (e.g. aaggggghhhhhhh!!!!!, soooooo huge).

# This script is a work in progress and is not yet complete.
# It is intended to be used as a starting point for extracting features from
# Wikipedia edits for vandalism detection.
"""

import re, os
from collections import Counter
from typing import Dict, List, Tuple
from nltk import ngrams


# Predefined list of bad words
#   This is a very short sample list for demonstration; 
#   real list should be more comprehensive and multilingual.
#   It is recommended to use a more extensive list of bad words for better detection.
#   The list can be extended with more words or phrases that are commonly used in vandalism or offensive content.
#   This list can be used to check if the edit contains any bad words.
#   It can be used to check if the edit contains any bad words.
BAD_WORDS = {"bad", "idiot", "stupid", "vandal", "destroy", "spam"}

def extract_features(edit: Dict) -> Dict:
    """
    Extracts features from a Wikipedia edit.

    Parameters:
    - edit: Dict, a dictionary containing the edit data.

    Returns:
    - A dictionary with extracted features.
    """
    features = {}
    # Basic metadata features
    features['edit_id'] = edit.get('edit_id', '')
    features['user_id'] = edit.get('user_id', '')
    features['user_name'] = edit.get('user_name', '')
    features['timestamp'] = edit.get('timestamp', '')
    features['is_anonymous'] = edit.get('is_anonymous', False)
    features['is_bot'] = edit.get('is_bot', False)
    features['is_minor'] = edit.get('is_minor', False)
    features['is_reverted'] = edit.get('is_reverted', False)
    features['is_deleted'] = edit.get('is_deleted', False)
    features['is_rollback'] = edit.get('is_rollback', False)
    features['is_new_page'] = edit.get('is_new_page', False)
    features['is_edited_page'] = edit.get('is_edited_page', False)
    features['is_merged_page'] = edit.get('is_merged_page', False)
    features['is_split_page'] = edit.get('is_split_page', False)
    features['is_renamed_page'] = edit.get('is_renamed_page', False)
    features['is_unlinked_page'] = edit.get('is_unlinked_page', False)
    features['is_linked_page'] = edit.get('is_linked_page', False)
    features['is_linked_to_page'] = edit.get('is_linked_to_page', False)
    features['is_linked_from_page'] = edit.get('is_linked_from_page', False)
    features['is_redirect'] = edit.get('is_redirect', False)
    features['is_external_link'] = edit.get('is_external_link', False)
    features['is_internal_link'] = edit.get('is_internal_link', False)
    features['is_section_edit'] = edit.get('is_section_edit', False)
    features['is_template_edit'] = edit.get('is_template_edit', False)
    features['is_category_edit'] = edit.get('is_category_edit', False)
    features['is_reference_edit'] = edit.get('is_reference_edit', False)
    features['is_image_edit'] = edit.get('is_image_edit', False)
    features['is_link_edit'] = edit.get('is_link_edit', False)
    features['is_comment_edit'] = edit.get('is_comment_edit', False)
    features['is_anonymous_edit'] = edit.get('is_anonymous_edit', False)
    features['is_bot_edit'] = edit.get('is_bot_edit', False)
    features['is_minor_edit'] = edit.get('is_minor_edit', False)
    features['is_vandalism'] = edit.get('is_vandalism', False)
    features['is_valid'] = edit.get('is_valid', True)  # Whether the edit is valid or not
    features['is_invalid'] = not features['is_valid']  # Inverse of is_valid
    features['is_test_edit'] = edit.get('is_test_edit', False)  # Whether the edit is a test edit
    features['is_reverted_edit'] = edit.get('is_reverted_edit', False)  # Whether the edit has been reverted
    features['is_rollback_edit'] = edit.get('is_rollback_edit', False)  # Whether the edit is a rollback
    features['is_new_page_edit'] = edit.get('is_new_page_edit', False)  # Whether the edit is a new page
    features['is_edited_page_edit'] = edit.get('is_edited_page_edit', False)  # Whether the edit is an edited page
    features['is_merged_page_edit'] = edit.get('is_merged_page_edit', False)  # Whether the edit is a merged page
    features['is_split_page_edit'] = edit.get('is_split_page_edit', False)  # Whether the edit is a split page
    features['is_renamed_page_edit'] = edit.get('is_renamed_page_edit', False)  # Whether the edit is a renamed page
    features['is_unlinked_page_edit'] = edit.get('is_unlinked_page_edit', False)  # Whether the edit is an unlinked page
    features['is_linked_page_edit'] = edit.get('is_linked_page_edit', False)  # Whether the edit is a linked page
    features['is_linked_to_page_edit'] = edit.get('is_linked_to_page_edit', False)  # Whether the edit is a linked to page
    features['is_linked_from_page_edit'] = edit.get('is_linked_from_page_edit', False)  # Whether the edit is a linked from page
    features['is_redirect_edit'] = edit.get('is_redirect_edit', False)  # Whether the edit is a redirect
    features['is_external_link_edit'] = edit.get('is_external_link_edit', False)  # Whether the edit is an external link
    features['is_internal_link_edit'] = edit.get('is_internal_link_edit', False)  # Whether the edit is an internal link
    features['is_section_edit_edit'] = edit.get('is_section_edit_edit', False)  # Whether the edit is a section edit
    features['is_template_edit_edit'] = edit.get('is_template_edit_edit', False)  # Whether the edit is a template edit
    features['is_category_edit_edit'] = edit.get('is_category_edit_edit', False)  # Whether the edit is a category edit
    features['is_reference_edit_edit'] = edit.get('is_reference_edit_edit', False)  # Whether the edit is a reference edit
    features['is_image_edit_edit'] = edit.get('is_image_edit_edit', False)  # Whether the edit is an image edit
    features['is_link_edit_edit'] = edit.get('is_link_edit_edit', False)  # Whether the edit is a link edit
    features['is_comment_edit_edit'] = edit.get('is_comment_edit_edit', False)  # Whether the edit is a comment edit
    features['is_anonymous_edit_edit'] = edit.get('is_anonymous_edit_edit', False)  # Whether the edit is an anonymous edit
    features['is_bot_edit_edit'] = edit.get('is_bot_edit_edit', False)  # Whether the edit is a bot edit
    features['is_minor_edit_edit'] = edit.get('is_minor_edit_edit', False)  # Whether the edit is a minor edit
    features['is_vandalism_edit'] = edit.get('is_vandalism_edit', False)  # Whether the edit is a vandalism edit
    features['edit_type'] = edit.get('edit_type', 'unknown')  # Type of edit (e.g., 'edit', 'revert', 'rollback', etc.)
    features['edit_action'] = edit.get('edit_action', 'unknown')  # Action taken in the edit (e.g., 'add', 'remove', 'modify', etc.)
    features['edit_summary'] = edit.get('edit_summary', '')  # Summary of the edit
    features['edit_tags'] = edit.get('edit_tags', [])  # Tags associated with the edit
    features['edit_namespace'] = edit.get('edit_namespace', '')  # Namespace of the edit (e.g., 'main', 'talk', 'user', etc.)
    features['edit_title'] = edit.get('edit_title', '')  # Title of the page being edited
    features['edit_language'] = edit.get('edit_language', 'en')  # Language of the edit (default is 'en' for English)
    features['edit_revision_id'] = edit.get('edit_revision_id', '')  # Revision ID of the edit
    features['edit_parent_revision_id'] = edit.get('edit_parent_revision_id', '')  # Parent revision ID of the edit
    features['edit_diff'] = edit.get('edit_diff', '')  # Diff of the edit, showing changes made
    features['edit_diff_size'] = len(edit.get('edit_diff', ''))  # Size of the diff in characters
    features['edit_diff_lines'] = len(edit.get('edit_diff', '').splitlines())  # Number of lines in the diff
    features['edit_diff_insertions'] = edit.get('edit_diff_insertions', 0)  # Number of insertions in the diff
    features['edit_diff_deletions'] = edit.get('edit_diff_deletions', 0)  # Number of deletions in the diff
    features['edit_diff_changes'] = edit.get('edit_diff_changes', 0)  # Number of changes in the diff
    features['edit_diff_additions'] = edit.get('edit_diff_additions', 0)  # Number of additions in the diff
    features['edit_diff_removals'] = edit.get('edit_diff_removals',
        0)  # Number of removals in the diff
    features['edit_diff_replacements'] = edit.get('edit_diff_replacements',
        0)  # Number of replacements in the diff
    features['edit_diff_inserted_text'] = edit.get('edit_diff_inserted_text', '')  # Text that was inserted in the edit
    features['edit_diff_deleted_text'] = edit.get('edit_diff_deleted_text', '')
    # Text that was deleted in the edit
    features['edit_diff_modified_text'] = edit.get('edit_diff_modified_text', '')  # Text that was modified in the edit
    features['edit_diff_unmodified_text'] = edit.get('edit_diff_unmodified_text', '')  # Text that was unmodified in the edit
    features['edit_diff_inserted_lines'] = edit.get('edit_diff_inserted_lines', [])  # Lines that were inserted in the edit
    features['edit_diff_deleted_lines'] = edit.get('edit_diff_deleted_lines', [])  # Lines that were deleted in the edit
    features['edit_diff_modified_lines'] = edit.get('edit_diff_modified_lines', [])  # Lines that were modified in the edit
    features['edit_diff_unmodified_lines'] = edit.get('edit_diff_unmodified_lines', [])  # Lines that were unmodified in the edit
    features['edit_diff_inserted_words'] = edit.get('edit_diff_inserted_words', [])  # Words that were inserted in the edit
    features['edit_diff_deleted_words'] = edit.get('edit_diff_deleted_words', [])  # Words that were deleted in the edit
    features['edit_diff_modified_words'] = edit.get('edit_diff_modified_words', [])  # Words that were modified in the edit
    features['edit_diff_unmodified_words'] = edit.get('edit_diff_unmodified_words', [])  # Words that were unmodified in the edit
    features['edit_diff_inserted_characters'] = edit.get('edit_diff_inserted_characters', [])  # Characters that were inserted in the edit
    features['edit_diff_deleted_characters'] = edit.get('edit_diff_deleted_characters', [])  # Characters that were deleted in the edit
    features['edit_diff_modified_characters'] = edit.get('edit_diff_modified_characters', [])  # Characters that were modified in the edit
    features['edit_diff_unmodified_characters'] = edit.get('edit_diff_unmodified_characters', [])  # Characters that were unmodified in the edit
    features['edit_diff_inserted_lines_count'] = len(edit.get('edit_diff_inserted_lines', []))  # Count of inserted lines in the edit
    features['edit_diff_deleted_lines_count'] = len(edit.get('edit_diff_deleted_lines', []))  # Count of deleted lines in the edit
    features['edit_diff_modified_lines_count'] = len(edit.get('edit_diff_modified_lines', []))  # Count of modified lines in the edit
    features['edit_diff_unmodified_lines_count'] = len(edit.get('edit_diff_unmodified_lines', []))  # Count of unmodified lines in the edit
    features['edit_diff_inserted_words_count'] = len(edit.get('edit_diff_inserted_words', []))  # Count of inserted words in the edit
    features['edit_diff_deleted_words_count'] = len(edit.get('edit_diff_deleted_words', []))  # Count of deleted words in the edit
    features['edit_diff_modified_words_count'] = len(edit.get('edit_diff_modified_words', []))  # Count of modified words in the edit
    features['edit_diff_unmodified_words_count'] = len(edit.get('edit_diff_unmodified_words', []))  # Count of unmodified words in the edit
    features['edit_diff_inserted_characters_count'] = len(edit.get('edit_diff_inserted_characters', []))  # Count of inserted characters in the edit
    features['edit_diff_deleted_characters_count'] = len(edit.get('edit_diff_deleted_characters', []))  # Count of deleted characters in the edit
    features['edit_diff_modified_characters_count'] = len(edit.get('edit_diff_modified_characters', []))  # Count of modified characters in the edit
    features['edit_diff_unmodified_characters_count'] = len(edit.get('edit_diff_unmodified_characters', []))  # Count of unmodified characters in the edit
    features['edit_diff_inserted_text_length'] = len(edit.get('edit_diff_inserted_text', ''))  # Length of inserted text in the edit
    features['edit_diff_deleted_text_length'] = len(edit.get('edit_diff_deleted_text', ''))  # Length of deleted text in the edit
    features['edit_diff_modified_text_length'] = len(edit.get('edit_diff_modified_text', ''))  # Length of modified text in the edit
    features['edit_diff_unmodified_text_length'] = len(edit.get('edit_diff_unmodified_text', ''))  # Length of unmodified text in the edit
    features['edit_diff_inserted_lines_length'] = sum(len(line) for line in edit.get('edit_diff_inserted_lines', []))  # Total length of inserted lines in the edit
    features['edit_diff_deleted_lines_length'] = sum(len(line) for line in edit.get('edit_diff_deleted_lines', []))  # Total length of deleted lines in the edit
    features['edit_diff_modified_lines_length'] = sum(len(line) for line in edit.get('edit_diff_modified_lines', []))  # Total length of modified lines in the edit
    features['edit_diff_unmodified_lines_length = sum(len(line) for line in edit.get('edit_diff_unmodified_lines', []))  # Total length of unmodified lines in the edit
    features['edit_diff_inserted_words_length'] = sum(len(word) for word in edit.get('edit_diff_inserted_words', []))  # Total length of inserted words in the edit
    features['edit_diff_deleted_words_length'] = sum(len(word) for word in edit.get('edit_diff_deleted_words', []))  # Total length of deleted words in the edit
    features['edit_diff_modified_words_length'] = sum(len(word) for word in edit.get('edit_diff_modified_words', []))  # Total length of modified words in the edit
    features['edit_diff_unmodified_words_length'] = sum(len(word) for word in edit.get('edit_diff_unmodified_words', []))  # Total length of unmodified words in the edit
    features['edit_diff_inserted_characters_length'] = sum(len(char) for char in edit.get('edit_diff_inserted_characters', []))  # Total length of inserted characters in the edit
    features['edit_diff_deleted_characters_length'] = sum(len(char) for char in edit.get('edit_diff_deleted_characters', []))  # Total length of deleted characters in the edit
    features['edit_diff_modified_characters_length'] = sum(len(char) for char in edit.get('edit_diff_modified_characters', []))  # Total length of modified characters in the edit
    features['edit_diff_unmodified_characters_length'] = sum(len(char) for char in edit.get('edit_diff_unmodified_characters', []))  # Total length of unmodified characters in the edit
    features['edit_diff_inserted_text_words'] = edit.get('edit_diff_inserted_text_words', [])  # Words that were inserted in the edit
    features['edit_diff_deleted_text_words'] = edit.get('edit_diff_deleted_text_words', [])  # Words that were deleted in the edit
    features['edit_diff_modified_text_words'] = edit.get('edit_diff_modified_text_words', [])  # Words that were modified in the edit
    features['edit_diff_unmodified_text_words'] = edit.get('edit_diff_unmodified_text_words', [])  # Words that were unmodified in the edit
    features['edit_diff_inserted_text_characters'] = edit.get('edit_diff_inserted_text_characters', [])  # Characters that were inserted in the edit
    features['edit_diff_deleted_text_characters'] = edit.get('edit_diff_deleted_text_characters', [])  # Characters that were deleted in the edit
    features['edit_diff_modified_text_characters'] = edit.get('edit_diff_modified_text_characters', [])  # Characters that were modified in the edit
    features['edit_diff_unmodified_text_characters'] = edit.get('edit_diff_unmodified_text_characters', [])  # Characters that were unmodified in the edit
    features['edit_diff_inserted_text_words_count'] = len(edit.get('edit_diff_inserted_text_words', []))  # Count of inserted words in the edit
    features['edit_diff_deleted_text_words_count'] = len(edit.get('edit_diff_deleted_text_words', []))  # Count of deleted words in the edit
    features['edit_diff_modified_text_words_count'] = len(edit.get('edit_diff_modified_text_words', []))  # Count of modified words in the edit
    features['edit_diff_unmodified_text_words_count'] = len(edit.get('edit_diff_unmodified_text_words', []))  # Count of unmodified words in the edit
    features['edit_diff_inserted_text_characters_count'] = len(edit.get('edit_diff_inserted_text_characters', []))  # Count of inserted characters in the edit
    features['edit_diff_deleted_text_characters_count'] = len(edit.get('edit_diff_deleted_text_characters', []))  # Count of deleted characters in the edit
    features['edit_diff_modified_text_characters_count'] = len(edit.get('edit_diff_modified_text_characters', []))  # Count of modified characters in the edit
    features['edit_diff_unmodified_text_characters_count'] = len(edit.get('edit_diff_unmodified_text_characters', []))  # Count of unmodified characters in the edit
    features['edit_diff_inserted_text_words_length'] = sum(len(word) for word in edit.get('edit_diff_inserted_text_words', []))  # Total length of inserted words in the edit
    features['edit_diff_deleted_text_words_length'] = sum(len(word) for word in edit.get('edit_diff_deleted_text_words', []))  # Total length of deleted words in the edit
    features['edit_diff_modified_text_words_length'] = sum(len(word) for word in edit.get('edit_diff_modified_text_words', []))  # Total length of modified words in the edit
    features['edit_diff_unmodified_text_words_length'] = sum(len(word) for word in edit.get('edit_diff_unmodified_text_words', []))  # Total length of unmodified words in the edit
    features['edit_diff_inserted_text_characters_length'] = sum(len(char) for char in edit.get('edit_diff_inserted_text_characters', []))  # Total length of inserted characters in the edit
    features['edit_diff_deleted_text_characters_length'] = sum(len(char) for char in edit.get('edit_diff_deleted_text_characters', []))  # Total length of deleted characters in the edit
    features['edit_diff_modified_text_characters_length'] = sum(len(char) for char in edit.get('edit_diff_modified_text_characters', []))  # Total length of modified characters in the edit
    features['edit_diff_unmodified_text_characters_length'] = sum(len(char) for char in edit.get('edit_diff_unmodified_text_characters', []))  # Total length of unmodified characters in the edit
    features['edit_diff_inserted_text_ngrams'] = edit.get('edit_diff_inserted_text_ngrams', [])  # N-grams of inserted text in the edit
    features['edit_diff_deleted_text_ngrams'] = edit.get('edit_diff_deleted_text_ngrams', [])  # N-grams of deleted text in the edit
    features['edit_diff_modified_text_ngrams'] = edit.get('edit_diff_modified_text_ngrams', [])  # N-grams of modified text in the edit
    features['edit_diff_unmodified_text_ngrams'] = edit.get('edit_diff_unmodified_text_ngrams', [])
    # N-grams of unmodified text in the edit
    features['edit_diff_inserted_text_ngrams_count'] = len(edit.get('edit_diff_inserted_text_ngrams', []))  # Count of inserted text n-grams in the edit
    features['edit_diff_deleted_text_ngrams_count'] = len(edit.get('edit_diff_deleted_text_ngrams', []))  # Count of deleted text n-grams in the edit
    features['edit_diff_modified_text_ngrams_count'] = len(edit.get('edit_diff_modified_text_ngrams', []))  # Count of modified text n-grams in the edit
    features['edit_diff_unmodified_text_ngrams_count'] = len(edit.get('edit_diff_unmodified_text_ngrams', []))  # Count of unmodified text n-grams in the edit
    features['edit_diff_inserted_text_ngrams_length'] = sum(len(ngram) for ngram in edit.get('edit_diff_inserted_text_ngrams', []))  # Total length of inserted text n-grams in the edit
    features['edit_diff_deleted_text_ngrams_length'] = sum(len(ngram) for ngram in edit.get('edit_diff_deleted_text_ngrams', []))  # Total length of deleted text n-grams in the edit
    features['edit_diff_modified_text_ngrams_length'] = sum(len(ngram) for ngram in edit.get('edit_diff_modified_text_ngrams', []))  # Total length of modified text n-grams in the edit
    features['edit_diff_unmodified_text_ngrams_length'] = sum(len(ngram) for ngram in edit.get('edit_diff_unmodified_text_ngrams', []))  # Total length of unmodified text n-grams in the edit
    features['edit_diff_inserted_text_ngrams_words'] = edit.get('edit_diff_inserted_text_ngrams_words', [])  # Words in inserted text n-grams in the edit
    features['edit_diff_deleted_text_ngrams_words'] = edit.get('edit_diff_deleted_text_ngrams_words', [])  # Words in deleted text n-grams in the edit
    features['edit_diff_modified_text_ngrams_words'] = edit.get('edit_diff_modified_text_ngrams_words', [])  # Words in modified text n-grams in the edit
    features['edit_diff_unmodified_text_ngrams_words'] = edit.get('edit_diff_unmodified_text_ngrams_words', [])  # Words in unmodified text n-grams in the edit
    features['edit_diff_inserted_text_ngrams_words_count'] = len(edit.get('edit_diff_inserted_text_ngrams_words', []))  # Count of words in inserted text n-grams in the edit
    features['edit_diff_deleted_text_ngrams_words_count'] = len(edit.get('edit_diff_deleted_text_ngrams_words', []))  # Count of words in deleted text n-grams in the edit
    features['edit_diff_modified_text_ngrams_words_count'] = len(edit.get('edit_diff_modified_text_ngrams_words', []))  # Count of words in modified text n-grams in the edit
    features['edit_diff_unmodified_text_ngrams_words_count'] = len(edit.get('edit_diff_unmodified_text_ngrams_words', []))  # Count of words in unmodified text n-grams in the edit
    features['edit_diff_inserted_text_ngrams_characters'] = edit.get('edit_diff_inserted_text_ngrams_characters', [])  # Characters in inserted text n-grams in the edit
    features['edit_diff_deleted_text_ngrams_characters'] = edit.get('edit_diff_deleted_text_ngrams_characters', [])  # Characters in deleted text n-grams in the edit
    features['edit_diff_modified_text_ngrams_characters'] = edit.get('edit_diff_modified_text_ngrams_characters', [])  # Characters in modified text n-grams in the edit
    features['edit_diff_unmodified_text_ngrams_characters'] = edit.get('edit_diff_unmodified_text_ngrams_characters', [])  # Characters in unmodified text n-grams in the edit
    features['edit_diff_inserted_text_ngrams_characters_count'] = len(edit.get('edit_diff_inserted_text_ngrams_characters', []))  # Count of characters in inserted text n-grams in the edit
    features['edit_diff_deleted_text_ngrams_characters_count'] = len(edit.get('edit_diff_deleted_text_ngrams_characters', []))  # Count of characters in deleted text n-grams in the edit
    features['edit_diff_modified_text_ngrams_characters_count'] = len(edit.get('edit_diff_modified_text_ngrams_characters', []))  # Count of characters in modified text n-grams in the edit
    features['edit_diff_unmodified_text_ngrams_characters_count'] = len(edit.get('edit_diff_unmodified_text_ngrams_characters', []))  # Count of characters in unmodified text n-grams in the edit
    features['edit_diff_inserted_text_ngrams_characters_length'] = sum(len(char) for char in edit.get('edit_diff_inserted_text_ngrams_characters', []))  # Total length of inserted text n-grams characters
    features['edit_diff_deleted_text_ngrams_characters_length'] = sum(len(char) for char in edit.get('edit_diff_deleted_text_ngrams_characters', []))  # Total length of deleted text n-grams characters
    features['edit_diff_modified_text_ngrams_characters_length'] = sum(len(char) for char in edit.get('edit_diff_modified_text_ngrams_characters', []))  # Total length of modified text n-grams characters
    features['edit_diff_unmodified_text_ngrams_characters_length'] = sum(len(char) for char in edit.get('edit_diff_unmodified_text_ngrams_characters', []))  # Total length of unmodified text n-grams characters
    features['edit_diff_inserted_text_ngrams_characters_words'] = edit.get('edit_diff_inserted_text_ngrams_characters_words', [])  # Words in inserted text n-grams characters
    features['edit_diff_deleted_text_ngrams_characters_words'] = edit.get('editq_diff_deleted_text_ngrams_characters_words', [])  # Words in deleted text n-grams characters
    features['edit_diff_modified_text_ngrams_characters_words'] = edit.get('edit_diff_modified_text_ngrams_characters_words', [])  # Words in modified text n-grams characters
    features['edit_diff_unmodified_text_ngrams_characters_words'] = edit.get('edit_diff_unmodified_text_ngrams_characters_words', [])  # Words in unmodified text n-grams characters
    features['edit_diff_inserted_text_ngrams_characters_words_count'] = len(edit.get('edit_diff_inserted_text_ngrams_characters_words', []))  # Count of words in inserted text n-grams characters
    features['edit_diff_deleted_text_ngrams_characters_words_count'] = len(edit.get('edit_diff_deleted_text_ngrams_characters_words', []))  # Count of words in deleted text n-grams characters
    features['edit_diff_modified_text_ngrams_characters_words_count'] = len(edit.get('edit_diff_modified_text_ngrams_characters_words', []))  # Count of words in modified text n-grams characters
    features['edit_diff_unmodified_text_ngrams_characters_words_count'] = len(edit.get('edit_diff_unmodified_text_ngrams_characters_words', []))  # Count of words in unmodified text n-grams characters
    features['edit_diff_inserted_text_ngrams_characters_length'] = sum(len(char) for char in edit.get('edit_diff_inserted_text_ngrams_characters', []))  # Total length of inserted text n-grams characters
    features['edit_diff_deleted_text_ngrams_characters_length'] = sum(len(char) for char in edit.get('edit_diff_deleted_text_ngrams_characters', []))  # Total length of deleted text n-grams characters
    features['edit_diff_modified_text_ngrams_characters_length'] = sum(len(char) for char in edit.get('edit_diff_modified_text_ngrams_characters', []))  # Total length of modified text n-grams characters
    features['edit_diff_unmodified_text_ngrams_characters_length'] = sum(len(char) for char in edit.get('edit_diff_unmodified_text_ngrams_characters', []))  # Total length of unmodified text n-grams characters
    features['edit_diff_inserted_text_ngrams_characters_words_length'] = sum(len(word) for word in edit.get('edit_diff_inserted_text_ngrams_characters_words', []))  # Total length of inserted text n-grams characters words
    features['edit_diff_deleted_text_ngrams_characters_words_length'] = sum(len(word) for word in edit.get('edit_diff_deleted_text_ngrams_characters_words', []))  # Total length of deleted text n-grams characters words
    features['edit_diff_modified_text_ngrams_characters_words_length'] = sum(len(word) for word in edit.get('edit_diff_modified_text_ngrams_characters_words', []))  # Total length of modified text n-grams characters words
    features['edit_diff_unmodified_text_ngrams_characters_words_length'] = sum(len(word) for word in edit.get('edit_diff_unmodified_text_ngrams_characters_words', []))  # Total length of unmodified text n-grams characters words

    # Basic text features
    old_text = edit.get('old_text', '')
    new_text = edit.get('new_text', '')

    features['old_length'] = len(old_text)
    features['new_length'] = len(new_text)
    features['size_increment'] = features['new_length'] - features['old_length']
    features['size_ratio'] = (1 + features['new_length']) / (1 + features['old_length']) if features['old_length'] > 0 else 1.0
    features['average_term_frequency'] = (
        sum(Counter(new_text.split()).values()) / len(new_text.split())
        if new_text.split() else 0
    )
    features['longest_word'] = max((len(word) for word in new_text.split()), default=0)
    features['longest_character_sequence'] = max(
        (len(match.group(0)) for match in re.finditer(r'(.)\1+', new_text)), default=0
    )
    features['upper_to_lower_ratio'] = (
        (1 + sum(1 for c in new_text if c.isupper())) /
        (1 + sum(1 for c in new_text if c.islower()))
    ) if any(c.islower() for c in new_text) else 1.0
    features['upper_to_all_ratio'] = (
        (1 + sum(1 for c in new_text if c.isupper())) /
        (1 + sum(1 for c in new_text if c.isalnum()))
    ) if any(c.isalnum() for c in new_text) else 1.0
    features['digit_ratio'] = (
        (1 + sum(1 for c in new_text if c.isdigit())) /
        (1 + len(new_text))
    ) if len(new_text) > 0 else 0.0
    features['non_alphanumeric_ratio'] = (
        (1 + sum(1 for c in new_text if not c.isalnum())) /
        (1 + len(new_text))
    ) if len(new_text) > 0 else 0.0
    features['character_diversity'] = (
        len(set(new_text)) ** (1 / len(new_text)) if len(new_text) > 0 else 0.0
    )
    features['character_distribution'] = (
        sum((new_text.count(c) / len(new_text)) * (1 / 26) for c in set(new_text)) if len(new_text) > 0 else 0.0
    )
    features['compressibility'] = (
        len(new_text) / len(new_text.encode('utf-8')) if len(new_text) > 0 else 0.0
    )
    features['anonymous'] = edit.get('anonymous', False)
    features['comment_length'] = len(edit.get('comment', ''))
    features['inserted_text'] = edit.get('inserted_text', '')
    features['inserted_words'] = set(word.lower() for word in new_text.split())
    features['inserted_words_case_sensitive'] = set(new_text.split())
    features['concatenated_inserted_words'] = ' '.join(new_text.split()).lower()
    features['concatenated_inserted_words_case_sensitive'] = ' '.join(new_text.split())
    features['case_sensitive_inserted_words'] = set(new_text.split())
    features['case_insensitive_inserted_words'] = set(word.lower() for word in new_text.split())
    features['inserted_lines'] = edit.get('inserted_lines', [])
    features['inserted_lines_case_sensitive'] = edit.get('inserted_lines_case_sensitive', [])
    features['inserted_lines_lowercase'] = edit.get('inserted_lines_lowercase', [])
    features['inserted_lines_uppercase'] = edit.get('inserted_lines_uppercase', [])
    features['inserted_lines_non_alphanumeric'] = edit.get('inserted_lines_non_alphanumeric', [])
    features['inserted_lines_alphanumeric'] = edit.get('inserted_lines_alphanumeric', [])
    features['inserted_lines_digits'] = edit.get('inserted_lines_digits', [])
    features['inserted_lines_non_digits'] = edit.get('inserted_lines_non_digits', [])
    features['inserted_lines_special_characters'] = edit.get('inserted_lines_special_characters', [])
    features['inserted_lines_non_special_characters'] = edit.get('inserted_lines_non_special_characters', [])
    features['inserted_lines_punctuation'] = edit.get('inserted_lines_punctuation', [])
    features['inserted_lines_non_punctuation'] = edit.get('inserted_lines_non_punctuation', [])
    features['inserted_lines_uppercase_ratio'] = (
        (1 + sum(1 for c in new_text if c.isupper())) /
        (1 + len(new_text))
    ) if len(new_text) > 0 else 0.0
    features['inserted_lines_lowercase_ratio'] = (
        (1 + sum(1 for c in new_text if c.islower())) /
        (1 + len(new_text))
    ) if len(new_text) > 0 else 0.0
    features['inserted_lines_digit_ratio'] = (
        (1 + sum(1 for c in new_text if c.isdigit())) /
        (1 + len(new_text))
    ) if len(new_text) > 0 else 0.0
    # Character counts
    features['num_characters'] = len(new_text)
    features['num_words'] = len(new_text.split())
    features['num_links'] = new_text.count('http') + new_text.count('www')  # Simple link detection
    features['num_templates'] = new_text.count('{{') - new_text.count('}}')
    features['num_categories'] = new_text.count('[[') - new_text.count(']]')
    features['num_references'] = new_text.count('<ref') + new_text.count('</ref>')
    features['num_images'] = new_text.count('File:') + new_text.count('Image:')
    features['num_sections'] = new_text.count('==')  # Simple section detection
    features['num_external_links'] = new_text.count('http') + new_text.count('www')  # Simple external link detection
    features['num_internal_links'] = new_text.count('[[') - new_text.count(']]')  # Simple internal link detection
    features['num_redirects'] = new_text.count('#REDIRECT') + new_text.count('#redirect')
    features['num_new_pages'] = new_text.count('new page')  # Placeholder for new pages
    features['num_deleted_pages'] = new_text.count('deleted page')  # Placeholder for deleted pages
    features['num_moved_pages'] = new_text.count('moved page')  # Placeholder for moved pages
    features['num_restored_pages'] = new_text.count('restored page')  # Placeholder for restored pages
    features['num_reverted_pages'] = new_text.count('reverted page')  # Placeholder for reverted pages
    features['num_merged_pages'] = new_text.count('merged page')  # Placeholder for merged pages
    features['num_split_pages'] = new_text.count('split page')  # Placeholder for split pages
    features['num_renamed_pages'] = new_text.count('renamed page')  # Placeholder for renamed pages
    features['num_unlinked_pages'] = new_text.count('unlinked page')  # Placeholder for unlinked pages
    features['num_linked_pages'] = new_text.count('linked page')  # Placeholder for linked pages
    features['num_linked_to_pages'] = new_text.count('linked to page')  # Placeholder for linked to pages
    features['num_linked_from_pages'] = new_text.count('linked from page')  # Placeholder for linked from pages
    features['num_bad_words'] = sum(word in BAD_WORDS for word in new_text.lower().split())
    features['bad_words'] = set(word for word in new_text.lower().split() if word in BAD_WORDS)



    # N-grams
    n_grams = ngrams(new_text.split(), 2)  # Example for bigrams
    features['bigrams'] = Counter(n_grams)

    return features