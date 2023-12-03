import numpy as np
import networkx as nx
from typing import List
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


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
            i = i + 1 if (
                                 i % 2) == 0 else i - 1  # Follows an edge of reality, either left (if the value is odds) or right (if the value is even)
            C[i] = cpt  # Tags the index with the current cycle
            # Follows the edge of desir (determines the value of j which is the next index of the edge of desir)
            if (i % 2) == 0:
                k = abs(T[i // 2])  # The absolut value allows to search in the P table the position of T
                if T[
                    i // 2] > 0:  # Allows to know which to next number we are, depending if it's even or odds we're either at the beginning or the end
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


def draw_graph(T: np.ndarray, P: np.ndarray, C: np.ndarray) -> None:
    """
    Draw the breakpoint graph
    The edges of reality are in blue and the edges of desire other colors (one color per cycle)
    The names of the nodes is formated as: b(eginning)/e(nd) + node number

    Parameters
    ----------
    T: borned signed permutation list
    P: Ordered values of T in a list
    C: Half points of T and the cycle it belongs to
    """
    colors = list(mcolors.BASE_COLORS)
    ax = plt.gca()
    g = nx.Graph()
    #############################
    #        Adding nodes       #
    #############################
    nodes: list = []
    level = 0  # Used to align the nodes
    for idx, t in enumerate(T):
        # checking if it's a borning number
        if t == 0:
            nodes.append((f"e{t}", {"level": level, "beginning": False}))
            level += 1
        elif t == len(T) - 1:
            nodes.append((f"b{t}", {"level": level, "beginning": True}))
        else:
            if t > 0:
                nodes.append((f'b{t}', {"level": level, "beginning": True}))
                level += 1
                nodes.append((f'e{t}', {"level": level, "beginning": False}))
                level += 1
            else:
                nodes.append((f'e{t}', {"level": level, "beginning": False}))
                level += 1
                nodes.append((f'b{t}', {"level": level, "beginning": True}))
                level += 1
        # nodes.append((t, {"level": idx}))

    # g.add_edges_from(nodes)
    g.add_nodes_from(nodes)
    pos = nx.multipartite_layout(g, subset_key="level")
    nodes = list(g)

    #############################
    #       edges of reality    #
    #############################
    for i in range(0, len(nodes) - 1, 2):
        g.add_edge(nodes[i], nodes[i + 1], reality=True)
    nx.draw_networkx_edges(g, pos, edgelist=g.edges(), width=1, edge_color=colors[0])
    g.add_edge(nodes[0], nodes[4], reality=False)

    #############################
    #       edges of desir      #
    #############################
    beginning_nodes: list = [b for (b, i) in g.nodes(data=True) if i["beginning"] is True]
    ending_nodes: list = [b for (b, i) in g.nodes(data=True) if i["beginning"] is False]
    desired_edges: list = []
    # Adding to the graph the edges of desir
    for i in range(len(P) - 1):  # Using P as it contains the ordered positions of T, the nodes used to create the nodes list
        ending_node = ending_nodes[P[i]]
        beginning_node = beginning_nodes[P[i + 1] - 1]
        cycle = int(C[nodes.index(ending_node)])
        desired_edges.append((ending_node, beginning_node, cycle))
    # Adding the curved edges
    for edge in desired_edges:
        end, beginning, cycle = edge
        rad = 0.5
        arrowprops = dict(arrowstyle='-',
                          color=colors[cycle + 1],
                          connectionstyle=f"arc3,rad={rad}",
                          linestyle='-',
                          alpha=0.6, )
        ax.annotate("",
                    xy=pos[end],
                    xytext=pos[beginning],
                    arrowprops=arrowprops)

    nx.draw_networkx_nodes(g, pos, node_size=200)
    nx.draw_networkx_labels(g, pos, font_size=10, font_family="sans-serif")

    plt.box(False)
    plt.show()


if __name__ == '__main__':
    L: List[int] = [-2, -1, 3]
    # L: List[int] = [3, -4, -2, -1]
    cpt = mystery_algorithm(L)
    print(cpt)
