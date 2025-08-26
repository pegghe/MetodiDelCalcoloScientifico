import time
import numpy as np
import warnings
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _is_symmetric(A, atol: float = 1e-12) -> bool:
    if sp.issparse(A):
        D = (A - A.T).tocoo()
        return D.nnz == 0 or np.max(np.abs(D.data)) <= atol
    return np.allclose(A, A.T, atol=atol, rtol=0.0)

def _is_positive_definite(A, atol: float = 0.0) -> bool:
    if sp.issparse(A):
        try:
            w_min, _ = spla.eigsh(A, k=1, which='SA')  # min eigenvalue
            return float(w_min[0]) > atol
        except Exception:
            return False
    else:
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            try:
                w = np.linalg.eigvalsh(A)
                return float(np.min(w)) > atol
            except Exception:
                return False

def conjugate_gradient(A, b, x_exact, tol, maxIter=20000):
    """
    Conjugate Gradient per matrici SPD.
    - Se A NON è simmetrica o NON è PD -> raise (come nel progetto).
    - Se a runtime capita d^T A d <= 0 -> WARNING, stop pulito, convergenza=False.
    """
    # --- CONTROLLI SPD (come nel progetto: fanno raise) ---
    if not _is_symmetric(A):
        raise ValueError("Matrix A is not symmetric, Conjugate Gradient failed.")
    if not _is_positive_definite(A):
        raise ValueError("Matrix A is not positive-definite, Conjugate Gradient failed.")

    n = b.size
    x = np.zeros(n)
    r = b - A @ x
    d = r.copy()
    delta_new = float(r @ r)
    iterazioni = 0
    start_time = time.time()
    convergenza = True

    nb = np.linalg.norm(b)
    if nb == 0.0:
        tempo_calcolo = time.time() - start_time
        err_rel = (np.linalg.norm(x_exact - x) / np.linalg.norm(x_exact)
                   if np.linalg.norm(x_exact) > 0 else 0.0)
        return x, 0, err_rel, tempo_calcolo, True

    while np.sqrt(delta_new) / nb > tol:
        if iterazioni >= maxIter:
            convergenza = False
            break

        q = A @ d
        denom = float(d @ q)
        if denom <= 0 or np.isclose(denom, 0.0):
            # Prima facevamo raise. Ora: warning e uscita pulita.
            warnings.warn(
                "Breakdown in CG: detected d^T A d <= 0 (A may not be SPD). "
                "Stopping and returning current iterate.",
                RuntimeWarning
            )
            convergenza = False
            break

        alpha = delta_new / denom
        x = x + alpha * d
        r = r - alpha * q
        delta_old = delta_new
        delta_new = float(r @ r)
        beta = delta_new / delta_old
        d = r + beta * d
        iterazioni += 1

    tempo_calcolo = time.time() - start_time
    err_relativo = (np.linalg.norm(x_exact - x) / np.linalg.norm(x_exact)
                    if np.linalg.norm(x_exact) > 0 else np.linalg.norm(x - x_exact))

    return x, iterazioni, err_relativo, tempo_calcolo, convergenza
