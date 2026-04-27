"""Opérateurs de voisinage utilisés par le recuit simulé.

Quatre opérateurs travaillent par blocs contigus :
    - large_swap     : échange de deux blocs disjoints de même taille
    - block_reverse  : inversion d'un bloc
    - scramble       : permutation aléatoire d'une sous-séquence
    - insert_range   : déplacement d'un bloc à une autre position

Un opérateur de réparation cible aléatoirement une ville en retard et la
réinsère plus tôt dans le tour. La ville 0 (le dépôt, premier élément) est
toujours fixe, comme imposé par l'énoncé.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


METHODS = ("large_swap", "block_reverse", "scramble", "insert_range")
DEFAULT_PROBA = (0.25, 0.25, 0.25, 0.25)


def random_neighbour(solution: List[str], proba: Sequence[float] = DEFAULT_PROBA) -> List[str]:
    """Tire un voisin selon la distribution ``proba`` sur les 4 méthodes."""
    sol = solution.copy()
    n = len(sol)
    method = np.random.choice(METHODS, p=proba)

    if method == "large_swap":
        size = np.random.randint(2, max(3, n // 3))
        i = np.random.randint(1, n - size)
        j = np.random.randint(1, n - size)
        # éviter le chevauchement
        guard = 0
        while abs(i - j) < size and guard < 50:
            j = np.random.randint(1, n - size)
            guard += 1
        block_i = sol[i : i + size]
        block_j = sol[j : j + size]
        sol[i : i + size] = block_j
        sol[j : j + size] = block_i

    elif method == "block_reverse":
        i = np.random.randint(1, n - 3)
        j = np.random.randint(i + 2, n - 1)
        sol[i : j + 1] = sol[i : j + 1][::-1]

    elif method == "scramble":
        i = np.random.randint(1, n - 4)
        j = np.random.randint(i + 3, n - 1)
        block = sol[i : j + 1]
        np.random.shuffle(block)
        sol[i : j + 1] = block

    elif method == "insert_range":
        size = np.random.randint(2, max(3, n // 4))
        i = np.random.randint(1, n - size)
        block = sol[i : i + size]
        sol = sol[:i] + sol[i + size :]
        new_pos = np.random.randint(1, len(sol))
        sol = sol[:new_pos] + block + sol[new_pos:]

    return sol


def repair_neighbour(
    solution: List[str], violations: np.ndarray, proba: Sequence[float] = DEFAULT_PROBA
) -> List[str]:
    """Tente de déplacer une ville en retard plus tôt dans le tour.

    Si aucune violation n'est connue, repli sur ``random_neighbour``.
    """
    if violations is None or len(violations) == 0:
        return random_neighbour(solution, proba)

    sol = solution.copy()
    violated_idx = int(violations[np.random.randint(len(violations))][0])
    if violated_idx > 1:
        city = sol.pop(violated_idx)
        new_pos = np.random.randint(1, violated_idx)
        sol.insert(new_pos, city)
    return sol
