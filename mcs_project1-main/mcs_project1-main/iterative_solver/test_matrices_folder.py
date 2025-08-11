import os
from iterative_solver.iterative_methods.conjugate_gradient import conjugate_gradient
from iterative_solver.iterative_methods.gauss_seidel import gauss_seidel
from iterative_solver.iterative_methods.jacobi import jacobi
from iterative_solver.utils.matrix_loader import load_matrix
from iterative_solver.utils.setup_variable import setup_variable
from iterative_solver.iterative_methods.gradient import gradient
from iterative_solver.utils.print_results import print_results
from iterative_solver.utils.results_saver import results_saver
from iterative_solver.utils.plot_results import plot_results



def test_matrices_folder(matrices_folder):

    # Trova tutti i file .mtx nella cartella
    matrix_files = [f for f in os.listdir(matrices_folder) if f.endswith('.mtx')]

    # Diverse tolleranze per testare
    tolleranze = [1e-4, 1e-6, 1e-8, 1e-10]

    # Cicla su ogni matrice trovata
    for matrix_file in matrix_files:
        matrix_name = os.path.splitext(matrix_file)[0]  # es: 'spa1' da 'spa1.mtx'
        print(f"\n\n===== TEST MATRICE: {matrix_name} =====\n")

        # Carica la matrice
        A = load_matrix(os.path.join(matrices_folder, matrix_file))

        # Prepara variabili
        x_esatto, b = setup_variable(A)

        # Dizionario per salvare i risultati
        risultati = {
            "Jacobi": [],
            "Gauss-Seidel": [],
            "Gradiente": [],
            "Gradiente coniugato": []
        }


        # Test JACOBI
        print(f"\n\n\nTEST JACOBI\n")
        for tol in tolleranze:
            print(f"Test con tolleranza: {tol}")
            x_approx, iters, err_rel, t_calc, conv = jacobi(A, b, x_esatto, tol)
            print_results(x_approx, iters, err_rel, t_calc, conv)
            risultati["Jacobi"].append((tol, iters, err_rel, t_calc, conv))

        # Test GAUSS-SEIDEL
        print(f"\n\n\nTEST Gauss-Seidel\n")
        for tol in tolleranze:
            print(f"Test con tolleranza: {tol}")
            x_approx, iters, err_rel, t_calc, conv = gauss_seidel(A, b, x_esatto, tol)
            print_results(x_approx, iters, err_rel, t_calc, conv)
            risultati["Gauss-Seidel"].append((tol, iters, err_rel, t_calc, conv))

        # Test GRADIENTE
        print(f"\n\n\nTEST GRADIENTE\n")
        for tol in tolleranze:
            print(f"Test con tolleranza: {tol}")
            x_approx, iters, err_rel, t_calc, conv = gradient(A, b, x_esatto, tol)
            print_results(x_approx, iters, err_rel, t_calc, conv)
            risultati["Gradiente"].append((tol, iters, err_rel, t_calc, conv))

        # Test GRADIENTE CONIUGATO
        print(f"\n\n\nTEST GRADIENTE CONIUGATO\n")
        for tol in tolleranze:
            print(f"Test con tolleranza: {tol}")
            x_approx, iters, err_rel, t_calc, conv = conjugate_gradient(A, b, x_esatto, tol)
            print_results(x_approx, iters, err_rel, t_calc, conv)
            risultati["Gradiente coniugato"].append((tol, iters, err_rel, t_calc, conv))


        # === Salva i risultati specifici di questa matrice ===
        results_saver(risultati, matrix_name=matrix_name)

        # === Genera i grafici specifici di questa matrice ===
        plot_results(risultati, matrix_name=matrix_name)
