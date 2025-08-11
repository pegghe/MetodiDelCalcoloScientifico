import numpy as np
import scipy.sparse as sp
import time

def gauss_seidel(A, b, x_true, tol, max_iter=20000):
    """
    Gauss-Seidel ottimizzato per matrici sparse (CSR).
    """

    n = A.shape[0]
    x = np.zeros_like(b, dtype=np.float64)
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

            if aii is None or aii == 0:
                raise ZeroDivisionError(f"A[{i},{i}] nullo: divisione impossibile")

            x[i] = (b[i] - sigma) / aii

        # Criterio di arresto
        residuo = A @ x - b
        err_rel = np.linalg.norm(residuo) / np.linalg.norm(b)
        if err_rel < tol:
            tempo = time.time() - start_time
            err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
            return x, k + 1, err, tempo, True

    tempo = time.time() - start_time
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    return x, max_iter, err, tempo, False
