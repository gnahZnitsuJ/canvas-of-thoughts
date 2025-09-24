# preprocessing functions

import re

# vocabulary preprocessing functions to overcome limitations with
#   nengo_spa python identifier limitations
CharReplacements = {
    '!': 'CV_EXCLAMATION_MARK',
    '"': 'CV_DOUBLE_QUOTE',
    '#': 'CV_HASH',
    '$': 'CV_DOLLAR_SIGN',
    '%': 'CV_PERCENT_SIGN',
    '&': 'CV_AMPERSAND',
    "'": 'CV_SINGLE_QUOTE',
    '(': 'CV_LEFT_PARENTHESIS',
    ')': 'CV_RIGHT_PARENTHESIS',
    '*': 'CV_ASTERISK',
    '+': 'CV_PLUS',
    ',': 'CV_COMMA',
    '-': 'CV_HYPHEN',
    '.': 'CV_PERIOD',
    '/': 'CV_FORWARD_SLASH',
    ':': 'CV_COLON',
    ';': 'CV_SEMICOLON',
    '<': 'CV_LESS_THAN',
    '=': 'CV_EQUALS',
    '>': 'CV_GREATER_THAN',
    '?': 'CV_QUESTION_MARK',
    '@': 'CV_AT_SYMBOL',
    '[': 'CV_LEFT_BRACKET',
    '\\': 'CV_BACKSLASH',
    ']': 'CV_RIGHT_BRACKET',
    '^': 'CV_CARET',
    '_': 'CV_UNDERSCORE',
    '`': 'CV_GRAVE_ACCENT',
    '{': 'CV_LEFT_BRACE',
    '|': 'CV_PIPE',
    '}': 'CV_RIGHT_BRACE',
    '~': 'CV_TILDE',
}

CharReplacementsTable = str.maketrans(CharReplacements) # translation table

InvCharReplacements = {v:k for k, v in CharReplacements.items()} # inverse dict for translation

# turning "words" (tokens) to usable identifiers
def WordsToSPAVocab(w: list):
    # translate special characters capital start for identifiers
    w = [x if x[0:3] == "CV_" else "WV_" + x.translate(CharReplacementsTable) for x in w]
    return w

# turning usable identifiers into "words" (tokens)
def SPAVocabToWords(w: list):
    
    # Create a function to replace all the placeholders with their corresponding characters
    def replace_placeholder_with_char(match):
        word = match.group(0)
        return InvCharReplacements.get(word, word)  # Return the original if no replacement exists

    # Regular expression to match all the placeholder words (e.g., "EXCLAMATION_MARK")
    pattern = r'\b(?:' + '|'.join(map(re.escape, InvCharReplacements.keys())) + r')\b'

    # inverse operations for WordsToSPAVocab
    w = [x[3:] for x in w]
    w = [re.sub(pattern, replace_placeholder_with_char, x) for x in w] # removing special characters
    return w 