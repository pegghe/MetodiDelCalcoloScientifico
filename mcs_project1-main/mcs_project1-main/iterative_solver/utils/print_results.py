def print_results(
    x_approssimato,
    iterazioni: int,
    errore_relativo: float,
    tempo_calcolo: float,
    convergenza: bool,
    maxIter: int = 20000,
    preview: int = 5
):
    """
    Stampa in modo ordinato i risultati di un metodo iterativo.

    Parameters
    ----------
    x_approssimato : ndarray
        La soluzione calcolata.
    iterazioni : int
        Numero di iterazioni fatte.
    errore_relativo : float
        Errore relativo finale.
    tempo_calcolo : float
        Tempo impiegato in secondi.
    convergenza : bool
        True se metodo convergente, False altrimenti.
    maxIter : int, default=20000
        Numero massimo di iterazioni (usato per il messaggio in caso di non convergenza).
    preview : int, default=5
        Numero di elementi della soluzione da stampare come anteprima.
    """

    separatore = "=" * 60
    print(separatore)

    if convergenza:
        print("✔ Metodo convergente")
        print(f"- Iterazioni eseguite:   {iterazioni}")
        print(f"- Errore relativo:       {errore_relativo:.2e}")
        print(f"- Tempo di calcolo:      {tempo_calcolo:.4f} secondi")
        print(f"- Norma soluzione:       {float((x_approssimato**2).sum())**0.5:.2e}")
        print(f"- Soluzione (prime {preview}): {x_approssimato[:preview]} ...")
    else:
        print("✘ ATTENZIONE: Metodo NON convergente")
        print(f"- Superato il massimo numero di iterazioni (maxIter = {maxIter})")

    print(separatore, "\n")
