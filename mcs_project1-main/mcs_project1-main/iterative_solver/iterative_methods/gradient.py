import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _is_symmetric(A, atol: float = 1e-12) -> bool:
    """
    Ritorna True se A è (numericamente) simmetrica.
    Supporta dense e sparse.
    """
    if sp.issparse(A):
        # Confronto numerico: differenza sparsa e massimo valore assoluto
        diff = (A - A.T).tocoo()
        if diff.nnz == 0:
            return True
        return np.max(np.abs(diff.data)) <= atol
    else:
        return np.allclose(A, A.T, atol=atol, rtol=0.0)

def _is_positive_definite(A, atol: float = 0.0) -> bool:
    """
    Verifica PD:
    - dense: prova Cholesky; fallback a eigvalsh minimo
    - sparse simmetrica: stima l'autovalore minimo con eigsh(k=1, which='SA')
    """
    if sp.issparse(A):
        # Assumi simmetria già verificata a monte
        try:
            # Autovalore minimo (algebraic) per matrici simmetriche
            w_min, _ = spla.eigsh(A, k=1, which='SA')
            return w_min[0] > atol
        except Exception:
            return False
    else:
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            try:
                w = np.linalg.eigvalsh(A)
                return np.min(w) > atol
            except Exception:
                return False

def gradient(A, b, x_exact, tol, max_iter=20000):
    """
    Metodo del gradiente ottimizzato per matrici sparse (funziona anche su dense).

    Controlli eseguiti prima di partire:
      - A deve essere simmetrica
      - A deve essere definita positiva
    """
    # Se è sparse, conviene portarla almeno in CSR/CSC per prodotti efficienti
    if sp.issparse(A) and not (sp.isspmatrix_csr(A) or sp.isspmatrix_csc(A)):
        A = A.tocsr()

    # --- Controlli richiesti ---
    if not _is_symmetric(A):
        raise ValueError("Matrix A is not symmetric, Gradient method failed.")
    if not _is_positive_definite(A):
        raise ValueError("Matrix A is not positive-definite, Gradient method failed.")

    # --- Algoritmo ---
    x = np.zeros_like(b)
    r = b - A @ x  # residuo iniziale

    nb = np.linalg.norm(b)
    # Gestione caso ||b|| = 0 per evitare divisione per zero nella condizione di arresto
    if nb == 0.0:
        return x, 0, (np.linalg.norm(x_exact - x) / np.linalg.norm(x_exact) if np.linalg.norm(x_exact) > 0 else 0.0), 0.0, True

    start_time = time.time()
    iterazioni = 0

    while np.linalg.norm(r) / nb > tol and iterazioni < max_iter:
        iterazioni += 1
        Ar = A @ r
        rr = float(np.dot(r, r))
        rAr = float(np.dot(r, Ar))
        if rAr == 0.0:
            # Protezione numerica: passo nullo -> interrompo
            break
        alpha = rr / rAr
        x = x + alpha * r
        r = b - A @ x

    tempo_calcolo = time.time() - start_time
    # Errore relativo rispetto a x_exact (se ha norma > 0)
    nx_true = np.linalg.norm(x_exact)
    if nx_true > 0:
        errore_relativo = np.linalg.norm(x_exact - x) / nx_true
    else:
        errore_relativo = np.linalg.norm(x - x_exact)

    convergenza = (np.linalg.norm(r) / nb <= tol) and (iterazioni <= max_iter)

    return x, iterazioni, errore_relativo, tempo_calcolo, convergenza
