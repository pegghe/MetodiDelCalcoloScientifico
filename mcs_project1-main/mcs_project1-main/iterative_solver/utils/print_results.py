def print_results(x_approssimato, iterazioni, errore_relativo, tempo_calcolo, convergenza, maxIter=20000):
    """
    Stampa in modo ordinato i risultati di un metodo iterativo.

    Parameters:
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
    maxIter : int
        Numero massimo di iterazioni (usato per il messaggio in caso di non convergenza).
    """

    print("=" * 60)
    if convergenza:
        print(f"Metodo convergente")
        print(f"Numero di iterazioni..... {iterazioni}")
        print(f"Errore relativo.......... {errore_relativo:.2e}")
        print(f"Tempo di calcolo......... {tempo_calcolo:.4f} secondi")
        print(f"Soluzione approssimata... {x_approssimato[:5]} ...")
    else:
        print(f"ATTENZIONE: Metodo NON convergente.")
        print(f"Superato il massimo numero di iterazioni (maxIter).")

    print("=" * 60)
    print()  # Riga vuota per separare visivamente
