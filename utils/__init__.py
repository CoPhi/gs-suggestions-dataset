import re

"""
    Queste espressioni regolari vogliono codificare pattern che rappresentano gli elementi editoriali presenti nella convenzione Leiden+.
    https://papyri.info/docs/leiden_plus
"""

PUNCTUATION_REGEX = re.compile(r"[.·,;:!?']")

BRACKETS_REGEX = re.compile(r"\[(.*?)\]")
UNMATCHED_BRACKETS_REGEX = re.compile(r"[\[\]]")

SUPPLEMENTS_REGEX = re.compile (r"(\[[^\]]+\])")

MISSING_LINES_REGEX = re.compile(r"⟦-+\⟧")

MARKER_REGEX = re.compile(r"&lt;(.*?)&gt;")
UNKNOWN_LEFT_MARKER_REGEX = re.compile(r"<gap/>lt;(.*?)&gt;")
EXTENDED_LINE_RIGHT_MARKER_REGEX = re.compile (r"break=\"no\"/&gt;") #

EXPUNCTION_REGEX = re.compile(r"\{\{(.*?)\}\}|\{(.*?)\}")

VACAT_REGEX = re.compile(r"vac\.|vacat")

NOTES_REGEX = re.compile(r"‡\d+")