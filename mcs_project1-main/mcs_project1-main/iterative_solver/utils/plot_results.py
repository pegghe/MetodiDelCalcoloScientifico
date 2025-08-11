import os
import matplotlib.pyplot as plt
import numpy as np

def plot_results(risultati_per_metodo, matrix_name):
    """
    Genera e salva i grafici comparativi tra metodi, in una cartella specifica per ogni matrice.

    Parametri:
    - risultati_per_metodo: dict con chiavi = nomi metodi, e valori = lista di tuple
    - matrix_name: nome della matrice (es: 'spa1')
    """

    # Cartella di destinazione: results/{matrix_name}_results/
    results_dir = os.path.join("results", f"{matrix_name}_results")
    os.makedirs(results_dir, exist_ok=True)

    metodi = list(risultati_per_metodo.keys())
    tolleranze = [t[0] for t in risultati_per_metodo[metodi[0]]]
    tolleranze_sorted = sorted(tolleranze, reverse=True)

    # === 1) Errore Relativo ===
    plt.figure(figsize=(8, 6))
    for metodo in metodi:
        data = {t[0]: t[2] for t in risultati_per_metodo[metodo]}
        errori_sorted = [data[tol] for tol in tolleranze_sorted]
        plt.plot(tolleranze_sorted, errori_sorted, marker='o', label=metodo, linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.minorticks_off()
    plt.xlabel("Tolleranza richiesta")
    plt.ylabel("Errore relativo ottenuto")
    plt.title("Errore Relativo vs Tolleranza")
    plt.legend()
    plt.grid(True, which="major", ls="--", lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "error_plot.png"))
    plt.close()

    # === 2) Numero Iterazioni ===
    plt.figure(figsize=(8, 6))
    for metodo in metodi:
        data = {t[0]: t[1] for t in risultati_per_metodo[metodo]}
        iterazioni_sorted = [data[tol] for tol in tolleranze_sorted]
        plt.plot(tolleranze_sorted, iterazioni_sorted, marker='o', label=metodo, linewidth=2)
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.minorticks_off()
    plt.xlabel("Tolleranza richiesta")
    plt.ylabel("Numero di iterazioni")
    plt.title("Iterazioni vs Tolleranza")
    plt.legend()
    plt.grid(True, which="major", ls="--", lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "iterations_plot.png"))
    plt.close()

    # === 3) Tempo di Calcolo ===
    plt.figure(figsize=(8, 6))
    for metodo in metodi:
        data = {t[0]: t[3] for t in risultati_per_metodo[metodo]}
        tempi_sorted = [data[tol] for tol in tolleranze_sorted]
        plt.plot(tolleranze_sorted, tempi_sorted, marker='o', label=metodo, linewidth=2)
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.minorticks_off()
    plt.xlabel("Tolleranza richiesta")
    plt.ylabel("Tempo di calcolo (s)")
    plt.title("Tempo vs Tolleranza")
    plt.legend()
    plt.grid(True, which="major", ls="--", lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "time_plot.png"))
    plt.close()

    print(f"Grafici generati correttamente in {results_dir}")
