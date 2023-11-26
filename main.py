import numpy
import numpy as np
from typing import List


def mystery_algorithm(L: List[int]) -> int:
    """
    Implementation of the mystery algorithm

    Parameters
    ----------
    L: list
        A list of relative integers of size n. Each integer i between 1 and n appears only once either as a positive or
        a negative value.

    Returns
    -------
    A positive integer
    """
    #################################
    #       Creating variables      #
    #################################
    n: int = len(L)
    # Create arrays with fixed sized filled with 0
    T: np.ndarray = np.zeros(n + 2, dtype=int)
    P: np.ndarray = np.zeros(n + 2, dtype=int)  # Table des positions correctes des valeurs de T
    C: np.ndarray = np.full(2 * n + 2, np.nan)
    i: int  # Pointer to the end and beginning in the arêtes du désir
    j: int
    k: int

    #################################
    #   Initializing variables      #
    #################################
    # Adding the delimiters
    T[0] = 0
    T[n + 1] = n + 1
    # Adding the values of the list that will need to be sorted
    for i in range(1, n + 1):
        T[i] = L[i - 1]

    for i in range(0, n + 2):
        P[abs(T[i])] = i

    cpt: int = 0

    #################################
    #           Main loop           #
    #################################
    # loop while C contains a nan value
    '''
    cpt = indique une inversion si le numéro change
    '''
    while np.isnan(C).any():
        cpt = cpt + 1  # Index des arêtes du désir
        i = np.isnan(C).argmax()  # Get the nan indexes and returns the first
        C[i] = cpt
        b = False
        while b is False:  # TODO: Theory: while it's false we're in the same arc (same orientation)
            i = i + 1 if (i % 2) == 0 else i - 1  # add one to i if i is even, else removes one
            C[i] = cpt
            if (i % 2) == 0:  # en fonction de l'index pair ou impair il faut appliquer des formules différentes
                k = abs(T[i // 2])
                if T[i // 2] > 0:
                    j = 2 * P[k + 1] - 1 if T[P[k + 1]] >= 0 else 2 * P[k + 1]
                else:
                     j = 2 * P[k - 1] if T[P[k - 1]] >= 0 else 2 * P[k - 1] - 1  # Permet d'obtenir la fin d'un élément
            else:
                k = abs(T[(i + 1) // 2])  # Récupère le numéro où est censé se trouver la case et l'assigne à k, permet de faire début et fin d'une case
                if (T[(i + 1) // 2]) > 0:  # Permet de vérifier l'orientation du chiffre
                    j = 2 * P[k - 1] if T[P[k - 1]] >= 0 else 2 * P[k - 1] - 1  # La liste P permet de retrouver le vrai emplacement du chiffre et de
                else:  # Regarde si la valeur complémentaire est elle aussi négative
                    j = 2 * P[k + 1] - 1 if T[P[k + 1]] >= 0 else 2 * P[k + 1]

            if np.isnan(C[j]):
                i = j
                C[i] = cpt
            else:
                b = True

    print(f"T:\t{T}\nP:\t{P}\nC:\t{C.astype(int)}")

    return cpt


if __name__ == '__main__':
    # L: List[int] = [1, 2, 3]
    L: List[int] = [3, -4, -2, -1]
    cpt = mystery_algorithm(L)
    print(cpt)
