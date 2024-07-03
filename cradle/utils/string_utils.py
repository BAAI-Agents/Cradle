import hashlib
import sys
import re


def hash_text_sha256(text: str) -> str:
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()


def contains_regex_characters(s: str) -> bool:
    # Pattern to match special regex characters
    regex_chars_pattern = r'[\.\^\$\*\+\?\{\}\[\]\\|()]'

    # Search for special regex characters in the string
    return re.search(regex_chars_pattern, s)


def strip_anchor_chars(s: str) -> str:
    # Strip the first character if it's '^'
    if s.startswith('^'):
        s = s[1:]

    # Strip the last character if it's '$'
    if s.endswith('$'):
        s = s[:-1]

    return s


def contains_punctuation(s: str) -> bool:
    # Pattern to match punctuation characters
    punctuation_pattern = r'[^\w\s_]'

    # Search for punctuation characters in the string
    return re.search(punctuation_pattern, s)


def is_numbered_bullet_list_item(s: str) -> int:
    # Regular expression to match the series of numbers followed by a dot
    pattern = r'^\d+\.'

    match = re.match(pattern, s)
    if match:
        # If there's a match, return the index right after the matched pattern
        return match.end()
    else:
        # If no match, return an invalid value
        return -1


def replace_unsupported_chars(text, replacement='?'):
    encoding = sys.getdefaultencoding()  # Get the default system encoding
    return text.encode(encoding, errors='replace').decode(encoding).replace('\ufffd', replacement)


