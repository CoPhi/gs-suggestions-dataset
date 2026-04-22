import re

RE_WHITESPACE = re.compile(r'\s+')
RE_SENTENCE_SPLIT = re.compile(r'([;·°]|(?<!\.)\.(?!\.))')
RE_TEST_CASE = re.compile(r"\[([^\]]+)\]")
