"""
Generazione dei grafici di confronto Pre-FT vs Post-FT sui modelli BERT.

Le metriche Pre-FT vengono calcolate localmente da `scripts/baseline_eval.py`
e incollate in `PRE_FT_BASELINE`. Le metriche Post-FT vengono recuperate
direttamente dalle run completate su Weights & Biases.

Struttura delle entry (sia pre che post FT):
    {
        "Modello": str,   # es. "Logion", "GreBerta", "aristoBERTo"
        "Stato":   str,   # "Pre-FT" | "Post-FT"
        "K":       int,   # 1 | 5 | 10 | 20
        "EM":      float, # Exact Match @K (%)
        "BS_Max":  float, # BERTScore F1 @K – massimo tra i top-K (%)
        "BS_Mean": float, # BERTScore F1 @K – media tra i top-K (%)
    }
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Dict, Any
import torch

# Configurazioni globali
WANDB_ENTITY = "gabrielegiannessi-university-of-pisa"
WANDB_PROJECT = "gs-suggestions"

# K values valutati (allineati con baseline_eval.py e la pipeline di FT)
K_VALUES = [1, 5, 10, 20]

# Ordine canonico dei modelli nei grafici
MODEL_ORDER = ["Logion", "GreBerta", "aristoBERTo"]

# Metriche baseline Pre-FT (calcolate da scripts/baseline_eval.py)

PRE_FT_BASELINE: List[Dict[str, Any]] = [
    {
        "Modello": "aristoBERTo",
        "Stato": "Pre-FT",
        "K": 1,
        "EM": 0.0,
        "BS_Max": 83.0455,
        "BS_Mean": 83.0455,
    },
    {
        "Modello": "aristoBERTo",
        "Stato": "Pre-FT",
        "K": 5,
        "EM": 0.0,
        "BS_Max": 85.3121,
        "BS_Mean": 82.6134,
    },
    {
        "Modello": "aristoBERTo",
        "Stato": "Pre-FT",
        "K": 10,
        "EM": 0.0,
        "BS_Max": 85.8827,
        "BS_Mean": 82.4376,
    },
    {
        "Modello": "aristoBERTo",
        "Stato": "Pre-FT",
        "K": 20,
        "EM": 0.4082,
        "BS_Max": 86.694,
        "BS_Mean": 82.2863,
    },
    {
        "Modello": "GreBerta",
        "Stato": "Pre-FT",
        "K": 1,
        "EM": 4.0816,
        "BS_Max": 85.6644,
        "BS_Mean": 85.6644,
    },
    {
        "Modello": "GreBerta",
        "Stato": "Pre-FT",
        "K": 5,
        "EM": 6.1224,
        "BS_Max": 88.8883,
        "BS_Mean": 84.7805,
    },
    {
        "Modello": "GreBerta",
        "Stato": "Pre-FT",
        "K": 10,
        "EM": 8.9796,
        "BS_Max": 89.9732,
        "BS_Mean": 84.5359,
    },
    {
        "Modello": "GreBerta",
        "Stato": "Pre-FT",
        "K": 20,
        "EM": 24.898,
        "BS_Max": 92.9495,
        "BS_Mean": 84.7051,
    },
    {
        "Modello": "Logion",
        "Stato": "Pre-FT",
        "K": 1,
        "EM": 0.0,
        "BS_Max": 78.8011,
        "BS_Mean": 78.8011,
    },
    {
        "Modello": "Logion",
        "Stato": "Pre-FT",
        "K": 5,
        "EM": 0.0,
        "BS_Max": 82.4378,
        "BS_Mean": 78.4957,
    },
    {
        "Modello": "Logion",
        "Stato": "Pre-FT",
        "K": 10,
        "EM": 0.0,
        "BS_Max": 83.5386,
        "BS_Mean": 78.4146,
    },
    {
        "Modello": "Logion",
        "Stato": "Pre-FT",
        "K": 20,
        "EM": 0.0,
        "BS_Max": 84.5998,
        "BS_Mean": 78.3979,
    },
]


def fetch_wandb_metrics() -> List[Dict[str, Any]]:
    """
    Recupera le metriche finali sul test set dalle run completate su W&B.

    Le chiavi W&B attese (log da pipeline.py):
        test/hcb_top{K}                  → EM@K
        test/hcb_bertscore_f1_top{K}     → BS_Max@K
        test/hcb_bertscore_f1_top{K}_mean → BS_Mean@K
    """
    print(f"Sincronizzazione API W&B: {WANDB_ENTITY}/{WANDB_PROJECT}...")
    api = wandb.Api()

    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"state": "finished"},
    )

    data: List[Dict[str, Any]] = []

    for run in runs:
        # Considera solo le run che hanno almeno la metrica top-1 sul test set
        if "test/hcb_top1" not in run.summary:
            continue

        checkpoint = run.config.get("checkpoint", "Sconosciuto")
        model_name = checkpoint.split("/")[-1].replace("gs-", "")

        for k in K_VALUES:
            data.append(
                {
                    "Modello": model_name,
                    "Stato": "Post-FT",
                    "K": k,
                    "EM": run.summary.get(f"test/hcb_top{k}", 0.0),
                    "BS_Max": run.summary.get(f"test/hcb_bertscore_f1_top{k}", 0.0),
                    "BS_Mean": run.summary.get(
                        f"test/hcb_bertscore_f1_top{k}_mean", 0.0
                    ),
                }
            )

    return data


def _add_bar_annotations(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    """Aggiunge etichette numeriche sopra le barre."""
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                fmt.format(height),
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=7,
            )


def _build_combined_df(post_ft_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Unisce Pre-FT e Post-FT in un unico DataFrame normalizzato."""
    df = pd.DataFrame(PRE_FT_BASELINE + post_ft_data)
    df["Modello"] = pd.Categorical(df["Modello"], categories=MODEL_ORDER, ordered=True)
    df["K"] = pd.Categorical(df["K"], categories=K_VALUES, ordered=True)
    df = df.sort_values(["Modello", "K", "Stato"]).reset_index(drop=True)
    return df


METRIC_CONFIGS = [
    {
        "col": "EM",
        "title": "Exact Match @K su Test Set",
        "ylabel": "EM@K (%)",
        "palette": {"Pre-FT": "#a1c9f4", "Post-FT": "#1f77b4"},
    },
    {
        "col": "BS_Max",
        "title": "BERTScore F1 @K (max) su Test Set",
        "ylabel": "BERTScore F1 Max@K (%)",
        "palette": {"Pre-FT": "#ffb482", "Post-FT": "#d62728"},
    },
    {
        "col": "BS_Mean",
        "title": "BERTScore F1 @K (mean) su Test Set",
        "ylabel": "BERTScore F1 Mean@K (%)",
        "palette": {"Pre-FT": "#c3e2c2", "Post-FT": "#2ca02c"},
    },
]


def generate_comparison_plot(
    df: pd.DataFrame,
    output_path: str = "models/bert/evaluation/test_metrics_comparison.png",
) -> None:
    """
    Genera un pannello di grafici a barre raggruppati (Pre-FT vs Post-FT)
    per ciascuna metrica (EM, BS_Max, BS_Mean) e per ciascun K.

    Layout: 3 righe (metriche) × len(K_VALUES) colonne (K=1,5,10,20).
    """
    print("\n=== TABELLA METRICHE SUL TEST SET ===")
    print(df.to_markdown(index=False))
    print(f"\nGenerazione del grafico in '{output_path}'...")

    n_rows = len(METRIC_CONFIGS)
    n_cols = len(K_VALUES)

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        sharey="row",
    )
    fig.suptitle("Confronto Pre-FT vs Post-FT — Test Set", fontsize=16, y=1.02)

    for row_idx, mcfg in enumerate(METRIC_CONFIGS):
        metric_col = mcfg["col"]
        palette = mcfg["palette"]

        for col_idx, k in enumerate(K_VALUES):
            ax = axes[row_idx][col_idx]
            df_k = df[df["K"] == k]

            sns.barplot(
                data=df_k,
                x="Modello",
                y=metric_col,
                hue="Stato",
                hue_order=["Pre-FT", "Post-FT"],
                palette=palette,
                ax=ax,
            )

            ax.set_title(f"K = {k}", fontsize=11)
            ax.set_xlabel("Architettura", fontsize=9)
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            _add_bar_annotations(ax)

            # Label Y solo sulla colonna sinistra
            if col_idx == 0:
                ax.set_ylabel(mcfg["ylabel"], fontsize=9)
            else:
                ax.set_ylabel("")

            # Titolo riga (metrica) sulla colonna sinistra
            if col_idx == 0:
                ax.set_title(f"{mcfg['title']}\nK = {k}", fontsize=10)

            # Legenda solo sul primo pannello in alto a sinistra
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left", fontsize=8)
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Grafico salvato: {output_path}")


def plot_umap_suggestions_cluster(
    candidate_embeddings: list["torch.Tensor"],
    gold_embedding: "torch.Tensor",
    candidate_labels: list[str],
    gold_label: str,
    output_path: str = "models/bert/evaluation/umap_cluster.png"
) -> float:
    """
    Proietta gli embedding dei candidati e della gold label in 2D usando UMAP.
    Calcola e restituisce la densità (coesione) del cluster dei candidati 
    (misurata come similarità coseno media tra tutte le coppie di candidati).
    
    Requires: `pip install umap-learn`
    """
    try:
        import umap.umap_ as umap
    except ImportError:
        import umap
        
    import torch
    import numpy as np
    from scipy.spatial.distance import pdist
    
    # 1. Preparazione dei dati
    all_emb = candidate_embeddings + [gold_embedding]
    # Convertiamo in array NumPy [N, hidden_dim]
    X = torch.stack(all_emb).detach().cpu().numpy()
    
    # 2. UMAP Projection (2D)
    # Impostiamo n_neighbors proporzionale ai dati per evitare errori se i candidati sono pochi
    n_neighbors = min(15, max(2, len(X) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(X)
    
    cand_2d = embedding_2d[:-1]
    gold_2d = embedding_2d[-1]
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot dei candidati
    plt.scatter(cand_2d[:, 0], cand_2d[:, 1], c='#1f77b4', label='Suggerimenti', s=100, alpha=0.7)
    for i, label in enumerate(candidate_labels):
        plt.annotate(
            label, 
            (cand_2d[i, 0], cand_2d[i, 1]), 
            xytext=(5, 5), 
            textcoords='offset points', 
            fontsize=9
        )
        
    # Plot della gold label (stella rossa)
    plt.scatter(gold_2d[0], gold_2d[1], c='#d62728', label='Gold Label', s=250, marker='*')
    plt.annotate(
        gold_label, 
        (gold_2d[0], gold_2d[1]), 
        xytext=(5, 5), 
        textcoords='offset points', 
        fontsize=11, 
        fontweight='bold', 
        color='#d62728'
    )
    
    plt.title('UMAP Projection dei Suggerimenti Generati vs Gold Label', fontsize=14)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    # 4. Calcolo della Coesione del cluster (Densità) nello spazio ad alta dimensionalità
    cand_X = X[:-1]
    if len(cand_X) > 1:
        # pdist con metrica 'cosine' restituisce la distanza coseno (1 - cos_sim)
        distances = pdist(cand_X, metric='cosine')
        cohesion = 1.0 - np.mean(distances)  # 1.0 = cluster perfettamente denso, sovrapposto
    else:
        cohesion = 1.0
        
    return cohesion


if __name__ == "__main__":
    try:
        post_ft_data = fetch_wandb_metrics()

        if not post_ft_data:
            print(
                "Attenzione: nessuna metrica 'test/hcb_top1' trovata nelle run "
                "completate su W&B. Il grafico includerà solo le baseline Pre-FT."
            )

        combined_df = _build_combined_df(post_ft_data)
        generate_comparison_plot(combined_df)

    except wandb.errors.CommError as e:
        print(f"Errore di connessione a Weights & Biases: {e}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Si è verificato un errore inaspettato: {e}")
