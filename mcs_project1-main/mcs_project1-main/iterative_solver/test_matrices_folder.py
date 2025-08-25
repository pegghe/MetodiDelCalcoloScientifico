import os
from typing import Callable, Dict, List, Tuple

from iterative_solver.iterative_methods.conjugate_gradient import conjugate_gradient
from iterative_solver.iterative_methods.gauss_seidel import gauss_seidel
from iterative_solver.iterative_methods.jacobi import jacobi
from iterative_solver.iterative_methods.gradient import gradient

from iterative_solver.utils.matrix_loader import load_matrix
from iterative_solver.utils.setup_variable import setup_variable
from iterative_solver.utils.print_results import print_results
from iterative_solver.utils.results_saver import results_saver
from iterative_solver.utils.plot_results import plot_results

# Tipo dei risultati per ogni run: (tol, iters, err_rel, t_calc, conv)
Result = Tuple[float, int, float, float, bool]


def _run_method_on_tolerances(
    name: str,
    solver_fn: Callable,
    A,
    b,
    x_esatto,
    tolleranze: List[float]
) -> List[Result]:
    """
    Esegue un singolo metodo iterativo su tutte le tolleranze richieste,
    stampa i risultati e ritorna la lista dei risultati.
    """
    print(f"\n\n\nTEST {name}\n")
    risultati: List[Result] = []

    for tol in tolleranze:
        print(f"Test con tolleranza: {tol}")
        x_approx, iters, err_rel, t_calc, conv = solver_fn(A, b, x_esatto, tol)
        print_results(x_approx, iters, err_rel, t_calc, conv)
        risultati.append((tol, iters, err_rel, t_calc, conv))

    return risultati


def test_matrices_folder(matrices_folder: str) -> None:
    # Trova tutti i file .mtx nella cartella (ordinati per stabilità dell'output)
    matrix_files = sorted(
        f for f in os.listdir(matrices_folder) if f.lower().endswith(".mtx")
    )

    # Diverse tolleranze per testare
    tolleranze = [1e-4, 1e-6, 1e-8, 1e-10]

    # Mappa nome->funzione per evitare ripetizioni
    metodi: Dict[str, Callable] = {
        "Jacobi": jacobi,
        "Gauss-Seidel": gauss_seidel,
        "Gradiente": gradient,
        "Gradiente coniugato": conjugate_gradient,
    }

    if not matrix_files:
        print(f"Nessun file .mtx trovato in: {matrices_folder}")
        return

    # Cicla su ogni matrice trovata
    for matrix_file in matrix_files:
        matrix_name = os.path.splitext(matrix_file)[0]  # es: 'spa1' da 'spa1.mtx'
        print(f"\n\n===== TEST MATRICE: {matrix_name} =====\n")

        # Carica la matrice
        A = load_matrix(os.path.join(matrices_folder, matrix_file))

        # Prepara variabili
        x_esatto, b = setup_variable(A)

        # Dizionario per salvare i risultati
        risultati: Dict[str, List[Result]] = {nome: [] for nome in metodi.keys()}

        # Esecuzione test per ogni metodo su tutte le tolleranze
        for nome, solver_fn in metodi.items():
            risultati[nome] = _run_method_on_tolerances(
                name=nome,
                solver_fn=solver_fn,
                A=A,
                b=b,
                x_esatto=x_esatto,
                tolleranze=tolleranze,
            )

        # === Salva i risultati specifici di questa matrice ===
        results_saver(risultati, matrix_name=matrix_name)

        # === Genera i grafici specifici di questa matrice ===
        plot_results(risultati, matrix_name=matrix_name)
