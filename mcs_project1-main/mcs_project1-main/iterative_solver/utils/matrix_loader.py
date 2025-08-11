from scipy.io import mmread
from scipy.sparse import csr_matrix

def load_matrix(filepath):
    """
    Carica una matrice dal file .mtx e la restituisce in formato sparso CSR.

    Parametri:
    - filepath (str): percorso del file .mtx

    Ritorna:
    - A (csr_matrix): matrice sparsa in formato CSR
    """
    A = mmread(filepath)  # Legge la matrice (sparse)
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)  # Converte in CSR se non lo è già
    return A
