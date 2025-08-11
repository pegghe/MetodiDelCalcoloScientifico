import os
import csv

def results_saver(risultati_per_metodo, matrix_name):
    """
    Salva i risultati in un file CSV ben formattato, in una cartella specifica per ogni matrice.

    Parametri:
    - risultati_per_metodo: dict, con chiavi = metodo, valori = lista tuple
      Ogni tupla: (tolleranza, iterazioni, errore_relativo, tempo_calcolo, convergenza)
    - matrix_name: nome della matrice (es: 'spa1')
    """

    # Cartella di destinazione: results/{matrix_name}_results/
    results_dir = os.path.join("results", f"{matrix_name}_results")
    os.makedirs(results_dir, exist_ok=True)

    # Percorso completo del file CSV
    filename = os.path.join(results_dir, "output.csv")

    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Intestazione
        writer.writerow(["Metodo", "Tolleranza", "Iterazioni", "Errore Relativo", "Tempo di Calcolo (s)", "Convergenza"])

        # Riga per riga
        for metodo, risultati in risultati_per_metodo.items():
            for risultato in risultati:
                tol, iterazioni, err_rel, tempo, conv = risultato

                # Formattazione più "umana"
                writer.writerow([
                    metodo,
                    f"{tol:.0e}",              # Es: 1e-06
                    iterazioni,
                    f"{err_rel:.2e}",          # Errore relativo scientifica con 2 cifre
                    f"{tempo:.4f}",            # Tempo con 4 decimali
                    "True" if conv else "False"      # Convergenza più leggibile
                ])

    print(f"Risultati salvati correttamente in {filename}")
