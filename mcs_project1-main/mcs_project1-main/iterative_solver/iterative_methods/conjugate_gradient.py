import time
import numpy as np


def conjugate_gradient(A, b, x_exact, tol, maxIter=20000):
    n = b.size
    x = np.zeros(n)
    r = b - A @ x
    d = r.copy()
    delta_new = np.dot(r, r)
    delta_0 = delta_new
    iterazioni = 0
    start_time = time.time()

    convergenza = True

    while np.sqrt(delta_new) / np.linalg.norm(b) > tol:
        if iterazioni >= maxIter:
            convergenza = False
            break

        q = A @ d
        alpha = delta_new / np.dot(d, q)
        x += alpha * d
        r -= alpha * q
        delta_old = delta_new
        delta_new = np.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d
        iterazioni += 1

    errore_relativo = np.linalg.norm(x_exact - x) / np.linalg.norm(x_exact)
    tempo_calcolo = time.time() - start_time

    return x, iterazioni, errore_relativo, tempo_calcolo, convergenza