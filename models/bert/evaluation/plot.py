import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# --- CONFIGURAZIONI GLOBALI ---
WANDB_ENTITY = "gabrielegiannessi-university-of-pisa"  
WANDB_PROJECT = "gs-suggestions"

# Metriche hardcodate dei modelli base (calcolate localmente)
PRE_FT_BASELINE = [
    {"Modello": "Logion", "Stato": "Pre-FT", "Top_1_EM": 0.0, "BS_F1_1": 0.0},
    {"Modello": "GreBerta", "Stato": "Pre-FT", "Top_1_EM": 0.0, "BS_F1_1": 0.0},
    {"Modello": "aristoBERTo", "Stato": "Pre-FT", "Top_1_EM": 0.0, "BS_F1_1": 0.0},
]

def fetch_wandb_metrics() -> List[Dict[str, Any]]:
    """Recupera le metriche finali dal server W&B filtrando solo le run completate."""
    print(f"Sincronizzazione API W&B: {WANDB_ENTITY}/{WANDB_PROJECT}...")
    api = wandb.Api()
    
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"state": "finished"} 
    )
    
    data = []
    for run in runs:
        if "test/hcb_top1" in run.summary:
            checkpoint = run.config.get("checkpoint", "Sconosciuto")
            model_name = checkpoint.split("/")[-1].replace("gs-", "")
            
            data.append({
                "Modello": model_name,
                "Stato": "Post-FT",
                "Top_1_EM": run.summary.get("test/hcb_top1", 0.0),
                "BS_F1_1": run.summary.get("test/hcb_bertscore_f1_top1", 0.0)
            })
    return data

def _add_bar_annotations(ax: plt.Axes) -> None:
    """Aggiunge le etichette testuali (valori numerici) sopra le barre del grafico."""
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

def generate_comparison_plot(df: pd.DataFrame, output_path: str = "models/bert/evaluation/test_metrics_comparison.png") -> None:
    """Genera e salva il grafico a barre raggruppate utilizzando la configurazione DRY."""
    print("\n=== TABELLA METRICHE SUL TEST SET ===")
    print(df.to_markdown(index=False))
    print(f"\nGenerazione del grafico vettoriale in {output_path}...")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_configs = [
        {
            "ax": axes[0], "y": "Top_1_EM", 
            "title": "Exact Match (Top-1) su Test Set", 
            "ylabel": "Top-1 EM (%)", 
            "palette": ['#a1c9f4', '#1f77b4']
        },
        {
            "ax": axes[1], "y": "BS_F1_1", 
            "title": "BERTScore (F1@1) su Test Set", 
            "ylabel": "BERTScore F1@1 (%)", 
            "palette": ['#ffb482', '#d62728']
        }
    ]

    for config in plot_configs:
        ax = config["ax"]
        sns.barplot(data=df, x='Modello', y=config["y"], hue='Stato', ax=ax, palette=config["palette"])
        ax.set_title(config["title"], fontsize=14, pad=15)
        ax.set_ylabel(config["ylabel"], fontsize=12)
        ax.set_xlabel('Architettura', fontsize=12)
        ax.set_ylim(0, 100)
        
        # Opzionale: sposta la legenda fuori dal grafico o disabilitala per pulizia
        if config["y"] == "Top_1_EM":
            ax.legend(loc='upper left')
        else:
            ax.get_legend().remove() # Rimuove legenda duplicata sul secondo plot
            
        _add_bar_annotations(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Salvataggio completato con successo.")

if __name__ == "__main__":
    try:
        post_ft_data = fetch_wandb_metrics()
        
        if not post_ft_data:
            print("Attenzione: Nessuna metrica 'test/hcb_top1' trovata nelle run completate su W&B.")
        else:
            # Unisce i dizionari e crea un DataFrame unico
            combined_df = pd.DataFrame(PRE_FT_BASELINE + post_ft_data)
            
            # Opzionale: Ordina i modelli per avere sempre Logion, GreBerta, aristoBERTo nello stesso ordine
            combined_df['Modello'] = pd.Categorical(combined_df['Modello'], ["Logion", "GreBerta", "aristoBERTo"])
            combined_df = combined_df.sort_values(['Modello', 'Stato'])
            
            generate_comparison_plot(combined_df)
            
    except wandb.errors.CommError as e:
        print(f"Errore di connessione a Weights & Biases: {e}")
    except Exception as e:
        print(f"Si è verificato un errore inaspettato: {e}")