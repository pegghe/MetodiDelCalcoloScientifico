import numpy as np
import scipy.sparse as sp
import time

def jacobi(A, b, x_true, tol, max_iter=20000):

    x = np.zeros_like(b)
    D = A.diagonal()
    D_inv = 1.0 / D

    # Prepara N come -A con diagonale azzerata, in modo efficiente
    A_nodiag = A.copy()
    A_nodiag = A_nodiag.tolil()
    A_nodiag.setdiag(0)
    N = -A_nodiag.tocsr()

    start_time = time.time()

    for k in range(max_iter):
        Nx = N.dot(x)
        x_new = D_inv * (Nx + b)

        residuo = A @ x_new - b
        err_rel_residuo = np.linalg.norm(residuo) / np.linalg.norm(b)

        if err_rel_residuo < tol:
            tempo = time.time() - start_time
            err_rel = np.linalg.norm(x_new - x_true) / np.linalg.norm(x_true)
            return x_new, k + 1, err_rel, tempo, True

        x = x_new

    tempo = time.time() - start_time
    err_rel = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    return x, max_iter, err_rel, tempo, False
