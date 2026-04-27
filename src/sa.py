"""Recuit simulé pour le TSPTW.

Schéma de refroidissement géométrique : à chaque itération
``T <- coef * T`` avec ``coef = (T_end / T_start) ** (1 / nb_iter)``,
de sorte que ``T`` atteint exactement ``T_end`` après ``nb_iter`` itérations.

Voisinage hybride : avec probabilité 0.3 (et seulement si la solution
courante a au moins une violation), un opérateur de réparation cible une
ville en retard ; sinon un voisinage par blocs aléatoire est tiré selon
``proba_voisinage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import io_utils, neighborhoods, scoring


REPAIR_RATE = 0.3
TRACK_EVERY = 1000


@dataclass
class SAConfig:
    T_start: float = 5_000.0
    T_end: float = 0.1
    nb_iterations: int = 100_000
    proba_voisinage: Sequence[float] = (0.0, 0.1, 0.75, 0.15)
    pen_violation: float = 350.0
    pen_retard: float = 30.0
    pen_avance: float = 10.0


@dataclass
class SATrace:
    """Données collectées pour les graphiques de progression."""

    best_scores: List[float] = field(default_factory=list)
    best_violations: List[int] = field(default_factory=list)
    current_scores: List[float] = field(default_factory=list)
    current_violations: List[int] = field(default_factory=list)
    sample_scores: List[float] = field(default_factory=list)
    deltas: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)


@dataclass
class SAResult:
    solution: List[str]
    score: float
    n_violations: int
    best_curve: List[float]          # série meilleur score (toutes les TRACK_EVERY itérations)
    ref_curve: List[float]           # même série, mais sous une métrique de référence fixe
    trace: Optional[SATrace] = None  # rempli si track_progress=True


def random_solution(instance: dict) -> List[str]:
    """Permutation uniforme commençant par '1' (dépôt)."""
    cities = [str(i) for i in range(2, len(instance) + 1)]
    np.random.shuffle(cities)
    return ["1"] + cities


def simulated_annealing(
    instance_path: str,
    config: SAConfig = SAConfig(),
    track_progress: bool = False,
    rng_seed: Optional[int] = None,
    ref_penalties: Tuple[float, float, float] = (500.0, 25.0, 3.0),
) -> SAResult:
    """Lance un recuit simulé sur l'instance désignée.

    Args:
        instance_path: chemin vers le fichier d'instance.
        config: paramètres du recuit.
        track_progress: si True, retourne une trace détaillée pour les
            graphiques (coût mémoire et temps non négligeable).
        rng_seed: graine de ``np.random`` pour reproductibilité (optionnel).
        ref_penalties: pénalités d'une métrique de référence indépendante du
            paramétrage testé, utile pour comparer des configurations entre
            elles. Par défaut, les valeurs (500, 25, 3) du sujet.

    Returns:
        SAResult.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    instance = io_utils.load_instance(instance_path)
    dist_mat = io_utils.load_or_build_distance_matrix(instance_path)

    coef = (config.T_end / config.T_start) ** (1.0 / config.nb_iterations)
    T = config.T_start

    current = random_solution(instance)
    res = scoring.compute_score(
        instance, current, dist_mat,
        pen_violation=config.pen_violation,
        pen_retard=config.pen_retard,
        pen_avance=config.pen_avance,
    )
    current_score = res.score
    current_retard = res.visite_en_retard

    ref_res = scoring.compute_score(
        instance, current, dist_mat,
        pen_violation=ref_penalties[0],
        pen_retard=ref_penalties[1],
        pen_avance=ref_penalties[2],
    )

    best_solution = current.copy()
    best_score = current_score
    best_n_violations = res.n_violations
    best_ref_score = ref_res.score

    best_curve = [best_score]
    ref_curve = [best_ref_score]
    trace = SATrace() if track_progress else None
    if trace is not None:
        trace.best_scores.append(best_score)
        trace.best_violations.append(res.n_violations)
        trace.current_scores.append(current_score)
        trace.current_violations.append(res.n_violations)
        trace.sample_scores.append(current_score)
        trace.temperatures.append(T)
        trace.iterations.append(0)

    iteration = 1
    while T > config.T_end:
        # voisinage : réparation ciblée 30% du temps si violations connues
        if len(current_retard) > 0 and np.random.rand() < REPAIR_RATE:
            new_sol = neighborhoods.repair_neighbour(current, current_retard, config.proba_voisinage)
        else:
            new_sol = neighborhoods.random_neighbour(current, config.proba_voisinage)

        new_res = scoring.compute_score(
            instance, new_sol, dist_mat,
            pen_violation=config.pen_violation,
            pen_retard=config.pen_retard,
            pen_avance=config.pen_avance,
        )
        delta = new_res.score - current_score
        accept = (delta < 0) or (np.random.rand() < np.exp(-delta / T))

        if accept:
            current = new_sol
            current_score = new_res.score
            current_retard = new_res.visite_en_retard
            if new_res.score < best_score:
                best_solution = new_sol.copy()
                best_score = new_res.score
                best_n_violations = new_res.n_violations
                # mettre à jour la métrique de référence
                ref_res = scoring.compute_score(
                    instance, new_sol, dist_mat,
                    pen_violation=ref_penalties[0],
                    pen_retard=ref_penalties[1],
                    pen_avance=ref_penalties[2],
                )
                best_ref_score = ref_res.score

        T *= coef
        iteration += 1

        if iteration % TRACK_EVERY == 0:
            best_curve.append(best_score)
            ref_curve.append(best_ref_score)
            if trace is not None:
                trace.best_violations.append(best_n_violations)
                trace.current_scores.append(current_score)
                trace.current_violations.append(new_res.n_violations)
                trace.sample_scores.append(new_res.score)
                trace.deltas.append(delta)
                trace.temperatures.append(T)
                trace.iterations.append(iteration)
                trace.best_scores.append(best_score)

    return SAResult(
        solution=best_solution,
        score=best_score,
        n_violations=best_n_violations,
        best_curve=best_curve,
        ref_curve=ref_curve,
        trace=trace,
    )
