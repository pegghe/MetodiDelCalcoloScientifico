import numpy as np
import scipy.sparse as sp
import time
import warnings

def jacobi(
    A, b, x_true, tol, max_iter=20000,
    check_matrix=True,
    require_diagonal_dominance=False,
    check_symmetry=False,
    symmetry_tol=1e-12
):
    """
    Jacobi per sistemi sparsi con controlli di matrice.

    Parametri extra:
    - check_matrix: se True esegue controlli strutturali su A e b.
    - require_diagonal_dominance: se True, solleva errore se A NON è (almeno) debolmente
      diagonalmente dominante per righe. Se False, emette solo un warning.
    - check_symmetry: se True, verifica (blandamente) A ≈ A^T (utile come diagnosi).
    - symmetry_tol: tolleranza per il check di simmetria.
    """
    if check_matrix:
        # Tipo e forma
        if not sp.isspmatrix(A):
            raise TypeError("A deve essere una matrice sparsa SciPy (scipy.sparse.spmatrix).")
        n, m = A.shape
        if n != m:
            raise ValueError(f"A deve essere quadrata; trovata {A.shape}.")
        b = np.asarray(b).reshape(-1)
        if b.size != n:
            raise ValueError(f"Dimensioni incoerenti: len(b)={b.size} ma A è {A.shape}.")

        # Diagonale non nulla
        D = A.diagonal()
        if np.any(D == 0):
            # Evita divisioni per zero in Jacobi
            idx0 = np.where(D == 0)[0][:5]  # mostra fino a 5 indici per diagnostica
            raise ValueError(f"Elementi nulli sulla diagonale di A ai/alle riga/e {idx0}. "
                             "Jacobi richiede D invertibile.")

        # check simmetria
        if check_symmetry:
            # Usa differenza sparsa senza densificare
            diff = (A - A.T).tocoo()
            max_abs = 0.0
            if diff.nnz > 0:
                max_abs = np.max(np.abs(diff.data))
            if max_abs > symmetry_tol:
                warnings.warn(f"A non risulta simmetrica: ||A - A.T||_max ≈ {max_abs:.2e} > {symmetry_tol:.2e}")

        # Dominanza diagonale
        absA = abs(A).tocsr()
        row_sums = np.array(absA.sum(axis=1)).ravel()
        offdiag = row_sums - np.abs(D)
        weak_dd = np.all(np.abs(D) >= offdiag - 1e-15)
        strict_some = np.any(np.abs(D) > offdiag + 1e-15)

        if not weak_dd:
            msg = ("La matrice A non è (nemmeno) diagonalmente dominante per righe. "
                   "Jacobi potrebbe non convergere.")
            if require_diagonal_dominance:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
        elif not strict_some:
            # Caso limite: solo uguaglianze; potrebbe rallentare la convergenza.
            warnings.warn("A è solo debolmente dominante (nessuna riga strettamente dominante).")

    # --- Implementazione Jacobi invariata
    x = np.zeros_like(b, dtype=float)
    D = A.diagonal().astype(float)
    D_inv = 1.0 / D

    # Prepara N = -A con diagonale azzerata (efficiente su sparsa)
    A_nodiag = A.tolil(copy=True)
    A_nodiag.setdiag(0)
    N = (-A_nodiag).tocsr()

    norm_b = np.linalg.norm(b)
    start_time = time.time()

    # Early exit se il residuo iniziale è già sotto soglia
    residuo0 = A @ x - b
    if norm_b == 0:
        # sistema omogeneo b=0: x=0 è soluzione
        tempo = time.time() - start_time
        err_rel = (np.linalg.norm(x - x_true) / np.linalg.norm(x_true)) if x_true is not None and np.linalg.norm(x_true) != 0 else None
        return x, 0, err_rel, tempo, True
    if np.linalg.norm(residuo0) / norm_b < tol:
        tempo = time.time() - start_time
        err_rel = (np.linalg.norm(x - x_true) / np.linalg.norm(x_true)) if x_true is not None and np.linalg.norm(x_true) != 0 else None
        return x, 0, err_rel, tempo, True

    for k in range(max_iter):
        Nx = N.dot(x)
        x_new = D_inv * (Nx + b)

        residuo = A @ x_new - b
        err_rel_residuo = np.linalg.norm(residuo) / norm_b

        if err_rel_residuo < tol:
            tempo = time.time() - start_time
            err_rel = (np.linalg.norm(x_new - x_true) / np.linalg.norm(x_true)) if x_true is not None and np.linalg.norm(x_true) != 0 else None
            return x_new, k + 1, err_rel, tempo, True

        # fallback numerico: interrompi se diverge o produce NaN/inf
        if not np.isfinite(err_rel_residuo):
            tempo = time.time() - start_time
            err_rel = (np.linalg.norm(x_new - x_true) / np.linalg.norm(x_true)) if x_true is not None and np.linalg.norm(x_true) != 0 else None
            return x_new, k + 1, err_rel, tempo, False

        x = x_new

    tempo = time.time() - start_time
    err_rel = (np.linalg.norm(x - x_true) / np.linalg.norm(x_true)) if x_true is not None and np.linalg.norm(x_true) != 0 else None
    return x, max_iter, err_rel, tempo, False
