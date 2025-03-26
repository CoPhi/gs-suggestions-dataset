import re

PUNCTUATION_REGEX = re.compile(r"[.·,;:!?']")
BRACKETS_REGEX = re.compile(r"\[(.*?)\]")
UNMATCHED_BRACKETS_REGEX = re.compile(r"[\[\]]")
SUPPLEMENTS_REGEX = re.compile (r"(\[[^\]]+\])")
MISSING_LINES_REGEX = re.compile(r"⟦-+\⟧")