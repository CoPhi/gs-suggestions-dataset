import csv
import pickle

def save_results_pickle(opt_log, filename):
    with open(filename, 'wb') as f:
        pickle.dump(opt_log, f)
        
def load_results_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)        

def save_results(logs, filename, is_lidstone=False):
    with open(filename, mode="w", newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Scrivi l'intestazione
        if is_lidstone:
            writer.writerow(["budget", "k_predictions", "dimension", "test_size", "min_frequency", "gamma", "topk-accuracy"])
        else:
            writer.writerow(["budget", "k_predictions", "dimension", "test_size", "min_frequency", "topk-accuracy"])

        # Scrivi i dati
        for log in logs:
            for budget, entry in log.items():
                params = entry["hyperparameter"].to_dict()
                acc = -entry["loss"]
                if is_lidstone:
                    writer.writerow([
                        budget,
                        params["k"],
                        params["n"],
                        params["test_size"],
                        params["min_freq"],
                        params["gamma"],
                        acc
                    ])
                else:
                    writer.writerow([
                        budget,
                        params["k"],
                        params["n"],
                        params["test_size"],
                        params["min_freq"],
                        acc
                    ])