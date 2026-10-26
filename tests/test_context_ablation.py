import os
import tempfile
import pytest

from models.bert.dataset.dev_set import DevCase
from models.bert.evaluation.ablation import (
    truncate_context_by_words,
    truncate_cases_by_words,
    window_label,
    plot_context_ablation_curves,
    generate_markdown_report,
    resolve_models_list,
)


def test_truncate_context_by_words_basic():
    text = "Οἱ δὲ φιλόσοφοι λέγ[...]σιν ὅτι τὸ ἀγαθόν ἐστιν"

    # Full context (w < 0)
    assert truncate_context_by_words(text, window_size=-1) == text
    assert truncate_context_by_words(text, window_size=None) == text

    # W = 2
    res_w2 = truncate_context_by_words(text, window_size=2)
    assert res_w2 == "δὲ φιλόσοφοι λέγ[...]σιν ὅτι τὸ"

    # W = 1
    res_w1 = truncate_context_by_words(text, window_size=1)
    assert res_w1 == "φιλόσοφοι λέγ[...]σιν ὅτι"

    # W = 0 (solo la parola con la lacuna)
    res_w0 = truncate_context_by_words(text, window_size=0)
    assert res_w0 == "λέγ[...]σιν"


def test_truncate_context_by_words_asymmetric():
    text = "Οἱ δὲ φιλόσοφοι λέγ[...]σιν ὅτι τὸ ἀγαθόν ἐστιν"
    res_asym = truncate_context_by_words(text, window_left=1, window_right=3)
    assert res_asym == "φιλόσοφοι λέγ[...]σιν ὅτι τὸ ἀγαθόν"


def test_truncate_context_by_words_edges():
    # Lacuna all'inizio
    text_start = "[....] ἐστιν ἡ σοφία τῶν ἀνθρώπων"
    res_start = truncate_context_by_words(text_start, window_size=2)
    assert res_start == "[....] ἐστιν ἡ"

    # Lacuna alla fine
    text_end = "ἡ σοφία τῶν ἀνθρώπων [....]"
    res_end = truncate_context_by_words(text_end, window_size=2)
    assert res_end == "τῶν ἀνθρώπων [....]"

    # Testo senza lacuna
    text_nogap = "οὐδεμία lacuna ἐνταῦθα"
    assert truncate_context_by_words(text_nogap, window_size=2) == text_nogap


def test_truncate_cases_by_words():
    cases = [
        DevCase(
            x="Οἱ δὲ φιλόσοφοι λέγ[...]σιν ὅτι τὸ ἀγαθόν",
            y=["ου"],
            gap_length=2,
            corpus_id="tlg",
            file_id="f1",
        )
    ]
    truncated = truncate_cases_by_words(cases, window_size=1)
    assert len(truncated) == 1
    assert truncated[0].x == "φιλόσοφοι λέγ[...]σιν ὅτι"
    assert truncated[0].y == ["ου"]
    assert truncated[0].gap_length == 2


def test_window_label():
    assert window_label(-1) == "Full"
    assert window_label(10) == "W=10"


def test_resolve_models_list():
    all_models = resolve_models_list("all")
    assert len(all_models) == 6
    assert "Jacobo/aristoBERTo" in all_models
    assert "CNR-ILC/gs-aristoBERTo" in all_models

    ft_models = resolve_models_list("ft")
    assert len(ft_models) == 3
    assert all("gs-" in m or "CNR-ILC" in m for m in ft_models)

    custom = resolve_models_list("m1, m2")
    assert custom == ["m1", "m2"]


def test_plot_and_markdown_generation():
    mock_results = {
        "Jacobo/aristoBERTo": {
            "W=2": {"top1": 10.0, "top5": 20.0, "bertscore_f1_top1": 80.0},
            "W=10": {"top1": 15.0, "top5": 25.0, "bertscore_f1_top1": 82.0},
            "Full": {"top1": 18.0, "top5": 30.0, "bertscore_f1_top1": 84.0},
        },
        "CNR-ILC/gs-aristoBERTo": {
            "W=2": {"top1": 25.0, "top5": 40.0, "bertscore_f1_top1": 86.0},
            "W=10": {"top1": 30.0, "top5": 45.0, "bertscore_f1_top1": 88.0},
            "Full": {"top1": 32.0, "top5": 50.0, "bertscore_f1_top1": 90.0},
        },
    }
    windows = [2, 10, -1]

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(tmpdir, "plot.png")
        md_path = os.path.join(tmpdir, "report.md")

        plot_context_ablation_curves(mock_results, output_path=plot_path, windows=windows)
        assert os.path.exists(plot_path)

        md = generate_markdown_report(
            mock_results,
            dataset_name="test_ds",
            split="test",
            max_cases=10,
            windows=windows,
            output_path=md_path,
        )
        assert os.path.exists(md_path)
        assert "Exact Match @1" in md
        assert "aristoBERTo" in md
        assert "Post-FT" in md
