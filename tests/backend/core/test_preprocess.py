from backend.core.preprocess import (
    contains_lacunae,
    strip_diacritics,
    normalize_greek,
    clean_lacunae,
    is_part_of_lacuna,
    remove_brackets,
    remove_lb,
    remove_punctuation,
    filter_dash,
    process_editorial_marks,
)
from backend.core import UNK_TOKEN, GAP_TOKEN

def test_contains_lacunae():
    assert contains_lacunae("text") is False
    assert contains_lacunae("test.") is False
    assert contains_lacunae("NONE_tag") is True
    assert contains_lacunae("a.b") is True
    assert contains_lacunae("...") is True
    assert contains_lacunae(f"{GAP_TOKEN}") is True

def test_strip_diacritics():
    assert strip_diacritics("ἄνθρωπος") == "ανθρωπος"
    assert strip_diacritics("ὅτι") == "οτι"

def test_normalize_greek():
    assert normalize_greek("ἄνθρωπος", case_folding=True) == "ΑΝΘΡΩΠΟΣ"
    assert normalize_greek("ἄνθρωπος", case_folding=False) == "ανθρωπος"

def test_clean_lacunae_docstring_examples():
    assert clean_lacunae("γέ.δουσιν") == UNK_TOKEN
    assert clean_lacunae("....") == UNK_TOKEN
    assert clean_lacunae(f"{GAP_TOKEN}.λέγειν") == f"{UNK_TOKEN} .λέγειν"

def test_is_part_of_lacuna():
    assert is_part_of_lacuna(".") is True
    assert is_part_of_lacuna(".a") is False
    assert is_part_of_lacuna(f"{GAP_TOKEN}") is True

def test_remove_brackets():
    assert remove_brackets("hello [world]") == "hello world"
    assert remove_brackets("[start] to end") == "start to end"
    assert remove_brackets("isolated [ and ]") == "isolated  and"

def test_remove_lb():
    assert remove_lb("line\n123") == "line " # \n\d* replaced by space
    assert remove_lb("line\n text") == "line  text"
    assert remove_lb("line\n") == "line "

def test_remove_punctuation():
    assert remove_punctuation("hello, world. How: are! you?") == "hello world How are you"

def test_filter_dash():
    assert filter_dash("hello - world") == "hello world"
    assert filter_dash("start -end") == "startend"
    assert filter_dash("start- end") == "startend"
    assert filter_dash("start-end") == "startend"

def test_process_editorial_marks():
    # Brackets tests
    assert process_editorial_marks("ἀλλὰ μὴν ἐν τῶι κατ[αϲκευάζειν") == "ἀλλὰ μὴν ἐν τῶι καταϲκευάζειν"
    
    # Integrations
    assert process_editorial_marks("a || b") == "a   b"
    assert process_editorial_marks("a ‖ b") == "a   b"
    # Dactyl patterns (assuming ⏑⏑‒ is replaced with GAP_TOKEN)
    assert process_editorial_marks("test ⏑⏑‒") == f"test {GAP_TOKEN}"
    # Leiden lb
    assert process_editorial_marks("a|b") == "a b"
    # Unclear signs
    assert process_editorial_marks("a+b*c") == "abc"
    # Vacat
    assert process_editorial_marks("abcvac.def") == "abcdef"
    assert process_editorial_marks("abcvacatdef") == "abcdef"
    # Doubts
    assert process_editorial_marks("abc?") == "abc"
    # Missing lines
    assert process_editorial_marks("abc⟦---⟧def") == "abcdef"
    # Parentheses
    assert process_editorial_marks("a(bc)") == "abc"
    # Markers
    #assert process_editorial_marks("a<bc>d") == "abcd"
    assert process_editorial_marks("a&lt;bc&gt;d") == "abcd"
    # Expunctions
    assert process_editorial_marks("a{bc}d") == "ad"
    assert process_editorial_marks("a{{bc}}d") == "ad"
    # Parallel text / dead words
    assert process_editorial_marks("a†bc†d") == "ad"
    assert process_editorial_marks("x_y") == "xy"
    # Notes
    assert process_editorial_marks("abc‡123") == "abc"
