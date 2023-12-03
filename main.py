import numpy as np
import networkx as nx
from typing import List
import matplotlib.pyplot as plt

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
    T: np.ndarray = np.zeros(n + 2, dtype=int)  # Borned signed permutations
    P: np.ndarray = np.zeros(n + 2, dtype=int)  # Reference to the values position in the T table
    C: np.ndarray = np.full(2 * n + 2, np.nan)  # Table of cycles
    i: int  
    j: int
    k: int

    #################################
    #   Initializing variables      #
    #################################
    print("Filling the table of borned signed permutations T")
    # Adding the delimiters
    T[0] = 0
    T[n + 1] = n + 1
    # Adding the permutations
    for i in range(1, n + 1):
        T[i] = L[i - 1]

    # Referencing the position of the values in T
    print("Filling the C table which allows to know the position of the values in the T table")
    for i in range(0, n + 2):
        P[abs(T[i])] = i

    cpt: int = 0  # Number of cycles

    #################################
    #           Main loop           #
    #################################
    '''
    Takes the non assigned point in C, depending if the value is even or odd we follow either an edge of reality or an edge of desir
    Les divisions par deux pour correspondre les index de C avec les index de P
    Si on revient sur une case déjà initialisé on revient au point de départ
    '''
    while np.isnan(C).any():
        cpt = cpt + 1  # Increment the cycle counter by one
        i = np.isnan(C).argmax()  # First index of a non explored cycle
        C[i] = cpt  # Tag the cycle
        b = False  # While False we're in the same cycle
        while b is False:
            i = i + 1 if (i % 2) == 0 else i - 1  # Follows an edge of reality, either left (if the value is odds) or right (if the value is even)
            C[i] = cpt  # Tags the index with the current cycle
            # Follows the edge of desir (determines the value of j which is the next index of the edge of desir)
            if (i % 2) == 0: 
                k = abs(T[i // 2])  # The absolut value allows to search in the P table the position of T
                if T[i // 2] > 0:  # Allows to know which to next number we are, depending if it's even or odds we're either at the beginning or the end   
                    j = 2 * P[k + 1] - 1 if T[P[k + 1]] >= 0 else 2 * P[k + 1]
                else:
                     j = 2 * P[k - 1] if T[P[k - 1]] >= 0 else 2 * P[k - 1] - 1  # Permet d'obtenir la fin d'un élément
            else:
                k = abs(T[(i + 1) // 2])
                if (T[(i + 1) // 2]) > 0:
                    j = 2 * P[k - 1] if T[P[k - 1]] >= 0 else 2 * P[k - 1] - 1 
                else:
                    j = 2 * P[k + 1] - 1 if T[P[k + 1]] >= 0 else 2 * P[k + 1]

            # Either the index is empty and assign it to the current cycle or we're on back on an index of the current cycle  and completed the cycle
            if np.isnan(C[j]):
                i = j
                C[i] = cpt
            else:
                b = True

    print(f"T:\t{T}\nP:\t{P}\nC:\t{C.astype(int)}")
    print(f"The number of inversion needed to order correctly is: {len(L) + 1 - cpt}")
    draw_graph(T, P, C)

    return cpt

def draw_graph(T: List[int], P: List[int], C: List[int]) -> None:
    """
    Draw the breakpoint graph

    Parameters
    ----------
    T: borned signed permutation list
    P: Ordered values of T in a list
    C: Half points of T and the cycle it belongs to
    """
    g = nx.Graph()
    nodes: list = []
    for idx, t in enumerate(T):
        nodes.append((t, {"level": idx}))

    g.add_edges_from(nodes)

    # Adding edges of reality
    for i in range(0, len(T), 2):
        print(g.nodes[T[i]])
        #g.add_edge(g.nodes[T[i]], g.nodes[T[i + 1]])

    pos = nx.multipartite_layout(g, subset_key="level")
    nx.draw(g, pos, with_labels=True)
    plt.show()


if __name__ == '__main__':
    L: List[int] = [-2, -1, 3]
    #L: List[int] = [3, -4, -2, -1]
    cpt = mystery_algorithm(L)
    print(cpt)
