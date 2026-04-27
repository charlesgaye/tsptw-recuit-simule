"""Fonction de score pénalisée pour le TSPTW.

Le score combine la distance totale du tour, une pénalité forfaitaire par
ville visitée hors fenêtre, et deux pénalités proportionnelles au temps
total de retard et d'avance :

    score = distance + p_v * n_violations + p_r * t_retard + p_a * t_avance

Cette version reproduit exactement la formulation utilisée pendant le
projet, **bug d'avance compris**. Voir la note détaillée dans
``compute_score`` ; un correctif est proposé dans ``compute_score_fixed``
mais n'est jamais appelé par les expériences afin que les chiffres
restent traçables au rapport rendu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ScoreResult:
    """Résultat détaillé de l'évaluation d'une solution."""

    score: float
    distance: float
    n_violations: int
    t_retard: float
    t_avance: float
    visite_en_retard: np.ndarray  # (k, 2) : [position dans le tour, retard]
    visite_en_avance: np.ndarray  # (k, 2) : [position dans le tour, avance]


def compute_score(
    instance,
    sol: List[str],
    dist_mat: np.ndarray,
    pen_violation: float = 500.0,
    pen_retard: float = 25.0,
    pen_avance: float = 3.0,
) -> ScoreResult:
    """Évalue une solution. **Reproduit le bug de la version rendue.**

    Bug : dans le calcul d'avance, ``duree`` est écrasée par ``next_start``
    avant la soustraction ``next_start - duree``, qui vaut donc toujours 0.
    En conséquence ``t_avance`` est systématiquement nul et le terme
    ``pen_avance * t_avance`` ne contribue jamais au score.
    """
    n = len(sol)
    distance = 0.0
    duree = 0.0
    n_violations = 0
    t_retard = 0.0
    t_avance = 0.0
    visite_retard: List[List[int]] = []
    visite_avance: List[List[int]] = []

    for i in range(n - 1):
        d_step = dist_mat[int(sol[i]), int(sol[i + 1])]
        distance += d_step
        duree += d_step
        next_start = instance[sol[i + 1]]["wstart"]
        end_window = instance[sol[i + 1]]["wend"]

        if duree < next_start:
            duree = next_start                # bug : duree écrasée avant
            t_avance += next_start - duree    # → 0
            visite_avance.append([i, int(next_start - duree)])
        if duree > end_window:
            n_violations += 1
            t_retard += duree - end_window
            visite_retard.append([i, int(duree - end_window)])

    # retour au dépôt
    last = n - 1
    d_step = dist_mat[int(sol[-1]), int(sol[0])]
    distance += d_step
    duree += d_step
    next_start = instance[sol[0]]["wstart"]
    end_window = instance[sol[0]]["wend"]
    if duree < next_start:
        duree = next_start
        t_avance += next_start - duree        # bug : idem
        visite_avance.append([last, int(next_start - duree)])
    if duree > end_window:
        n_violations += 1
        t_retard += duree - end_window
        visite_retard.append([last, int(duree - end_window)])

    score = distance + pen_violation * n_violations + pen_retard * t_retard + pen_avance * t_avance

    return ScoreResult(
        score=score,
        distance=distance,
        n_violations=n_violations,
        t_retard=t_retard,
        t_avance=t_avance,
        visite_en_retard=np.array(visite_retard, dtype=int).reshape(-1, 2),
        visite_en_avance=np.array(visite_avance, dtype=int).reshape(-1, 2),
    )


def compute_score_fixed(
    instance,
    sol: List[str],
    dist_mat: np.ndarray,
    pen_violation: float = 500.0,
    pen_retard: float = 25.0,
    pen_avance: float = 3.0,
) -> ScoreResult:
    """Version corrigée : l'avance est calculée avant l'écrasement de ``duree``.

    Fournie à titre documentaire pour un éventuel post-mortem du projet.
    Aucune des expériences du rapport n'utilise cette fonction.
    """
    n = len(sol)
    distance = 0.0
    duree = 0.0
    n_violations = 0
    t_retard = 0.0
    t_avance = 0.0
    visite_retard: List[List[int]] = []
    visite_avance: List[List[int]] = []

    def step(i_pos: int, prev_id: str, next_id: str, duree_in: float) -> float:
        nonlocal distance, n_violations, t_retard, t_avance
        d_step = dist_mat[int(prev_id), int(next_id)]
        distance += d_step
        d = duree_in + d_step
        ws = instance[next_id]["wstart"]
        we = instance[next_id]["wend"]
        if d < ws:
            t_avance += ws - d
            visite_avance.append([i_pos, int(ws - d)])
            d = ws
        if d > we:
            n_violations += 1
            t_retard += d - we
            visite_retard.append([i_pos, int(d - we)])
        return d

    for i in range(n - 1):
        duree = step(i, sol[i], sol[i + 1], duree)
    duree = step(n - 1, sol[-1], sol[0], duree)

    score = distance + pen_violation * n_violations + pen_retard * t_retard + pen_avance * t_avance
    return ScoreResult(
        score=score,
        distance=distance,
        n_violations=n_violations,
        t_retard=t_retard,
        t_avance=t_avance,
        visite_en_retard=np.array(visite_retard, dtype=int).reshape(-1, 2),
        visite_en_avance=np.array(visite_avance, dtype=int).reshape(-1, 2),
    )
