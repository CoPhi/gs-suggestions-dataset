from matplotlib import pyplot as plt
import pandas as pd

from config.settings import LM_TYPE

def read_data(csv_filename: str) -> pd.DataFrame: 
    return pd.read_csv(csv_filename)

def create_table(csv_filename: str, output_filename: str = "tabella.png") -> None:
    df = read_data(csv_filename)
    fig, ax = plt.subplots(figsize=(len(df.columns), len(df.values)))  # Dimensione della figura
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
        if key[0] == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f1f1f2')
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")

if __name__ == '__main__': 
    create_table(f'{LM_TYPE}_results.csv')