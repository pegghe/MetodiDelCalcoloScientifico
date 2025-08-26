import numpy as np
import scipy.sparse as sp
import time
import warnings

def _is_diagonally_dominant(A) -> bool:
    """
    Controlla (row-wise) se A è diagonalmente dominante (debole).
    """
    A = A.tocsr()
    n = A.shape[0]
    diag = A.diagonal()

    for i in range(n):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        idxs = A.indices[row_start:row_end]
        vals = A.data[row_start:row_end]

        sum_offdiag = 0.0
        aii = 0.0
        for j, v in zip(idxs, vals):
            if j == i:
                aii = v
            else:
                sum_offdiag += abs(v)

        if abs(aii) < sum_offdiag:
            return False
    return True


def gauss_seidel(A, b, x_true, tol, max_iter=20000):
    """
    Gauss-Seidel ottimizzato per matrici sparse (CSR).
    - Errore se la diagonale contiene zeri.
    - Warning se la matrice non è diagonalmente dominante.
    """
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()

    # --- Controlli ---
    diag = A.diagonal()
    if np.any(diag == 0):
        raise ValueError("La diagonale di A contiene almeno uno zero: Gauss-Seidel non è applicabile.")
    if not _is_diagonally_dominant(A):
        warnings.warn(
            "ATTENZIONE: la matrice non è diagonalmente dominante; Gauss-Seidel può non convergere.",
            RuntimeWarning,
            stacklevel=2
        )

    n = A.shape[0]
    x = np.zeros_like(b, dtype=np.float64)
    nb = np.linalg.norm(b)
    start_time = time.time()

    for k in range(max_iter):
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            Ai = A.indices[row_start:row_end]
            Av = A.data[row_start:row_end]

            sigma = 0.0
            aii = None

            for idx, j in enumerate(Ai):
                if j == i:
                    aii = Av[idx]
                else:
                    sigma += Av[idx] * x[j]

            if aii == 0:
                raise ZeroDivisionError(f"A[{i},{i}] nullo: divisione impossibile")

            x[i] = (b[i] - sigma) / aii

        # Criterio di arresto sul residuo relativo
        residuo = A @ x - b
        err_rel_res = np.linalg.norm(residuo) / nb
        if err_rel_res < tol:
            elapsed = time.time() - start_time
            err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
            return x, k + 1, err, elapsed, True

    elapsed = time.time() - start_time
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    return x, max_iter, err, elapsed, False
