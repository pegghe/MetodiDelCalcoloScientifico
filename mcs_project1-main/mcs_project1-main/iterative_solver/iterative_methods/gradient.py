import numpy as np
import time
import scipy.sparse as sp

def gradient(A, b, x_exact, tol, max_iter=20000):
    """
    Metodo del gradiente ottimizzato per matrici sparse (funziona anche su dense).
    """

    x = np.zeros_like(b)
    r = b - A @ x  # Calcolo del residuo iniziale

    start_time = time.time()
    iterazioni = 0

    while np.linalg.norm(r) / np.linalg.norm(b) > tol and iterazioni < max_iter:
        iterazioni += 1
        Ar = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x = x + alpha * r
        r = b - A @ x

    tempo_calcolo = time.time() - start_time
    errore_relativo = np.linalg.norm(x_exact - x) / np.linalg.norm(x_exact)
    convergenza = iterazioni < max_iter

    return x, iterazioni, errore_relativo, tempo_calcolo, convergenza
