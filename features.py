"""Features for Wikipedia Vandalism Detection
This script extracts features from Wikipedia edits to help compute features
for detecting vandalism.
Based on the Wikipedia Article: https://en.wikipedia.org/wiki/User:ClueBot_NG
Based on the paper: "Wikipedia Vandalism Detection Through Machine Learning:
Feature Review and New Proposals Lab Report for PAN at CLEF 2010"
Link: https://arxiv.org/pdf/1210.5560
Author: Santiago M. Mola Velasco
Email: santiago.mola@bitsnbrains.net
License: Undetermined

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


