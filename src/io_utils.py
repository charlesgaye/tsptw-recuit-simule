"""Lecture des instances TSPTW, matrice de distances, fichiers solution.

Le format d'instance suit celui distribué avec le sujet : l'en-tête est ignoré
jusqu'à la première ligne dont le premier champ est un entier ; chaque ligne
décrit ensuite une ville (id, x, y, ?, wstart, wend).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


City = Dict[str, float]
Instance = Dict[str, City]


def load_instance(path: str) -> Instance:
    """Charge une instance TSPTW depuis un fichier texte.

    Retourne un dict id -> {x, y, wstart, wend}. Les ids sont des chaînes
    pour rester compatibles avec les solutions (qui sont des listes de
    chaînes).
    """
    instance: Instance = {}
    with open(path, "r") as f:
        # On saute les lignes d'en-tête jusqu'à trouver une ligne dont le
        # premier champ est un entier.
        tokens: List[str] = []
        for line in f:
            tokens = [t for t in line.split() if t]
            if tokens and tokens[0].isdigit():
                break

        # On consomme la ligne courante puis le reste du fichier.
        while tokens and tokens[0].isdigit() and int(tokens[0]) < 999:
            instance[tokens[0]] = {
                "x": float(tokens[1]),
                "y": float(tokens[2]),
                "wstart": float(tokens[4]),
                "wend": float(tokens[5]),
            }
            line = f.readline()
            if not line:
                break
            tokens = [t for t in line.split() if t]
    return instance


def euclidean_floor(a: City, b: City) -> int:
    """Distance euclidienne tronquée (convention de l'énoncé)."""
    return math.floor(math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2))


def compute_distance_matrix(instance: Instance) -> np.ndarray:
    """Calcule la matrice n+1 par n+1 puis applique la fermeture triangulaire.

    Les indices 1..n correspondent aux villes ; l'indice 0 est laissé nul
    pour conserver une indexation 1-basée cohérente avec les solutions.
    """
    n = len(instance)
    mat = np.zeros((n + 1, n + 1))
    keys = list(instance.keys())
    for i in keys:
        for j in keys:
            mat[int(i), int(j)] = euclidean_floor(instance[i], instance[j])
    # Fermeture pour respecter l'inégalité triangulaire (Floyd-Warshall).
    for k in keys:
        ki = int(k)
        for i in keys:
            ii = int(i)
            for j in keys:
                jj = int(j)
                if mat[ii, jj] > mat[ii, ki] + mat[ki, jj]:
                    mat[ii, jj] = mat[ii, ki] + mat[ki, jj]
    return mat


def load_or_build_distance_matrix(instance_path: str) -> np.ndarray:
    """Cache la matrice de distances dans <instance>_dist.npy."""
    cache = instance_path + "_dist.npy"
    if os.path.exists(cache):
        return np.load(cache)
    instance = load_instance(instance_path)
    mat = compute_distance_matrix(instance)
    np.save(cache, mat)
    return mat


def write_solution(path: str, solution: List[str], score: Optional[int] = None) -> None:
    """Écrit une solution au format <ordre> / <score> sur deux lignes."""
    with open(path, "w") as f:
        f.write(" ".join(str(x) for x in solution) + "\n")
        if score is not None:
            f.write(str(score))


def read_solution(path: str) -> Tuple[List[str], Optional[int]]:
    """Lit une solution. Retourne (ordre, score) où score peut être None."""
    with open(path, "r") as f:
        order = f.readline().split()
        raw = f.readline().strip()
        score = int(raw) if raw else None
    return order, score
