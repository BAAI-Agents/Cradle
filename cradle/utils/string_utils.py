import hashlib
import sys
import re


def hash_text_sha256(text: str) -> str:
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()


def contains_regex_characters(s):
    # Pattern to match special regex characters
    regex_chars_pattern = r'[\.\^\$\*\+\?\{\}\[\]\\|()]'

    # Search for special regex characters in the string
    return re.search(regex_chars_pattern, s)


def strip_anchor_chars(s):
    # Strip the first character if it's '^'
    if s.startswith('^'):
        s = s[1:]

    # Strip the last character if it's '$'
    if s.endswith('$'):
        s = s[:-1]

    return s


def replace_unsupported_chars(text, replacement='?'):
    encoding = sys.getdefaultencoding()  # Get the default system encoding
    return text.encode(encoding, errors='replace').decode(encoding).replace('\ufffd', replacement)
