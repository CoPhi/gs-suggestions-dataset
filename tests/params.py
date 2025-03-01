"""
    Questo modulo contiene funzioni per stampare i parametri e i risultati dei modelli MLE e LIDSTONE su file CSV.
    Oltre a questo, contiene funzioni per ottenere i migliori parametri per i modelli MLE e LIDSTONE dai file CSV.
"""

import pandas as pd
import os

def print_MLE_params_to_csv(params: dict) -> None:
    file_exists = os.path.isfile("MLE_results.csv")
    df = pd.DataFrame([params])
    df.to_csv("MLE_results.csv", mode="a", header=not file_exists, index=False)


def print_LIDSTONE_params_to_csv(params: dict) -> None:
    file_exists = os.path.isfile("LIDSTONE_results.csv")
    df = pd.DataFrame([params])
    df.to_csv("LIDSTONE_results.csv", mode="a", header=not file_exists, index=False)
    
def get_best_params_MLE():
    df = pd.read_csv("MLE_results.csv")
    best_params = df.loc[df["ACCURACY"].idxmax()]
    best_params_dict = best_params.to_dict()
    for col in df.columns:
        best_params_dict[col] = df[col].dtype.type(best_params_dict[col])
    return best_params_dict

def get_best_params_LIDSTONE():
    df = pd.read_csv("LIDSTONE_results.csv")
    best_params = df.loc[df["ACCURACY"].idxmax()]
    best_params_dict = best_params.to_dict()
    for col in df.columns:
        best_params_dict[col] = df[col].dtype.type(best_params_dict[col])
    return best_params_dict