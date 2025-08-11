
from iterative_solver.test_matrices_folder import test_matrices_folder

class MainInterface:
    def __init__(self):
        self.banner()
        self.run()

    def banner(self):
        print("=" * 50)
        print("      Solver Iterativo per Sistemi Lineari")
        print("   Progetto 1 alternativo - Metodi del Calcolo Scientifico")
        print("=" * 50)

    def run(self):
        print("\nScegli una modalità :")
        print("1) Esegui le matrici fornite ( 'matrices' ) : ")
        print("2) Esegui le matrici custom ( 'custom_matrix' ) : ")

        scelta = input("Inserisci 1 o 2 : ").strip()

        if scelta == "1":
            print("\n[INFO] Avvio test su tutte le matrici in 'matrices/'...\n")
            test_matrices_folder("matrices")

        elif scelta == "2":
            print("\n[INFO] Avvio test su tutte le matrici in 'custom_matrix'...\n")
            test_matrices_folder("custom_matrix")

        else:
            print("\n[ERRORE] Scelta non valida. Riprova con 1 o 2.")

# Avvio automatico
if __name__ == "__main__":
    MainInterface()
