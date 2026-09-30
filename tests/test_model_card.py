import os
import tempfile
import pytest
from models.bert.finetuning.model_card import (
    generate_model_card,
    save_model_card,
    _format_metric_row,
    _render_metrics_table,
    METRIC_DEFINITIONS,
)


def test_format_metric_row():
    # Percentage metric positive delta
    label, pre, post, delta, badge = _format_metric_row("Exact Match @1", 20.0, 30.5, is_pct=True)
    assert pre == "20.00%"
    assert post == "30.50%"
    assert delta == "+10.50%"
    assert badge == "🟢"

    # Percentage metric negative delta
    label, pre, post, delta, badge = _format_metric_row("Exact Match @1", 30.5, 25.0, is_pct=True)
    assert delta == "-5.50%"
    assert badge == "🔻"

    # Float metric
    label, pre, post, delta, badge = _format_metric_row("Mean Margin", 0.05, 0.12, is_pct=False)
    assert pre == "0.0500"
    assert post == "0.1200"
    assert delta == "+0.0700"
    assert badge == "🟢"


def test_render_metrics_table():
    pre = {"top1": 25.0, "top5": 50.0, "bertscore_f1_top1": 80.0}
    post = {"top1": 35.0, "top5": 60.0, "bertscore_f1_top1": 85.0}

    table_md = _render_metrics_table(pre, post, title="Test Policy")
    assert "#### Test Policy" in table_md
    assert "| **Exact Match @1** | 25.00% | **35.00%** | `+10.00%` | 🟢 |" in table_md
    assert "| **Exact Match @5** | 50.00% | **60.00%** | `+10.00%` | 🟢 |" in table_md
    assert "| **BERTScore F1 @1** | 80.00% | **85.00%** | `+5.00%` | 🟢 |" in table_md


def test_generate_model_card_full():
    checkpoint = "CNR-ILC/gs-GreBerta"
    base_model = "bowphs/GreBerta"
    dataset_name = "CNR-ILC/gs-dataset-tlg-uncased"

    pre = {
        "default": {"top1": 25.0, "top5": 50.0, "bertscore_f1_top1": 80.0, "cluster_inclusion_rate": 60.0},
        "word": {"top1": 20.0, "top5": 40.0, "bertscore_f1_top1": 75.0, "cluster_inclusion_rate": 55.0},
    }
    post = {
        "default": {"top1": 35.0, "top5": 62.0, "bertscore_f1_top1": 86.0, "cluster_inclusion_rate": 68.0},
        "word": {"top1": 28.0, "top5": 52.0, "bertscore_f1_top1": 81.0, "cluster_inclusion_rate": 63.0},
    }

    card = generate_model_card(
        checkpoint=checkpoint,
        base_model=base_model,
        dataset_name=dataset_name,
        pre_ft_metrics=pre,
        post_ft_metrics=post,
        eval_metrics={"eval_loss": 1.45, "eval_perplexity": 4.26},
        hyperparameters={"epochs": 3, "lr": 1.2e-6, "batch_size": 128},
        preprocessing_config={"case_folding": "lower", "strip_diacritics": True, "remove_punct": True},
    )

    # Verifica YAML Frontmatter
    assert card.startswith("---\n")
    assert "language:\n- grc\n- el" in card
    assert "license: apache-2.0" in card
    assert "pipeline_tag: fill-mask" in card
    assert f"base_model: {base_model}" in card
    assert "model-index:" in card
    assert "name: gs-GreBerta" in card
    assert "widget:" in card

    # Verifica Sezioni Markdown
    assert "# gs-GreBerta" in card
    assert "GreekSchools" in card
    assert "CNR-ILC" in card
    assert "Highlight dei Risultati" in card
    assert "Policy 'Default'" in card
    assert "Policy 'Word'" in card
    assert "Iperparametri di Addestramento" in card
    assert "Pipeline di Preprocessing e Normalizzazione Filologica" in card
    assert "transformers import pipeline" in card
    assert "AutoModelForMaskedLM" in card
    assert "Limiti e Avvertenze Filologiche" in card
    assert "@misc" in card


def test_save_model_card():
    with tempfile.TemporaryDirectory() as tmp_dir:
        content = "---\nlicense: apache-2.0\n---\n# Test Card"
        saved_file = save_model_card(content, tmp_dir)
        assert os.path.exists(saved_file)
        assert saved_file.endswith("README.md")
        with open(saved_file, "r", encoding="utf-8") as f:
            read_content = f.read()
        assert read_content == content
