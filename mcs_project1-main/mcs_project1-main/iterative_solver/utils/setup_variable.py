import numpy as np
import scipy.sparse as sp

def setup_variable(A):
    """
    Genera il vettore x_esatto (tutti 1) e calcola b = A * x_esatto.
    Funziona sia con matrici dense che sparse.

    Parametri:
    ----------
    A : matrice (densa o sparsa)

    Ritorna:
    - x_esatto : vettore di 1
    - b : vettore calcolato come A @ x_esatto
    """
    size = A.shape[0]
    x_esatto = np.ones(size)

    # Usa @ che funziona sia per sparse che dense
    b = A @ x_esatto

    return x_esatto, b