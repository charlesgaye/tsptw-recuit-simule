"""Moyennage d'essais et intervalles de confiance.

``multiple_runs`` exécute ``nb_runs`` recuits indépendants pour une même
configuration et empile leurs courbes de score. Le résultat est une matrice
``(nb_runs, n_points)`` directement consommable par ``plotting.plot_ci``.

Note méthodologique importante : si on ne fixe pas une graine par
configuration, les configurations testées dans une même session partagent
l'état global de ``np.random``. Dans ce cas, deux configurations avec la
même paramétrisation visible peuvent voir des séquences aléatoires
différentes, et l'intervalle de confiance affiché ne borne que la moyenne
d'UNE configuration, pas la variabilité entre configurations. Pour des
comparaisons rigoureuses, passer ``rng_seed=...`` à chaque appel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from . import sa


def multiple_runs(
    instance_path: str,
    config: sa.SAConfig,
    nb_runs: int = 50,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """Empile les ``ref_curve`` de ``nb_runs`` essais.

    Si ``rng_seed`` est fourni, la graine du run ``k`` est ``rng_seed + k``,
    ce qui rend l'expérience reproductible et garantit que des
    configurations différentes voient les mêmes graines d'initialisation.
    """
    runs = []
    for k in range(nb_runs):
        seed = None if rng_seed is None else rng_seed + k
        result = sa.simulated_annealing(instance_path, config, track_progress=False, rng_seed=seed)
        runs.append(result.ref_curve)
    return np.array(runs)


def sweep_proba(
    instance_path: str,
    base: sa.SAConfig,
    proba_list,
    nb_runs: int = 50,
    rng_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Balaye une liste de vecteurs ``proba_voisinage``.

    Retourne ``{label: matrice (nb_runs, n_points)}``.
    """
    out: Dict[str, np.ndarray] = {}
    for p in proba_list:
        cfg = replace(base, proba_voisinage=tuple(p))
        out[f"p={list(p)}"] = multiple_runs(instance_path, cfg, nb_runs, rng_seed)
    return out


def sweep_penalties(
    instance_path: str,
    base: sa.SAConfig,
    grid,
    nb_runs: int = 50,
    rng_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Balaye une grille de pénalités.

    ``grid`` est une liste de triplets ``(pen_violation, pen_retard, pen_avance)``.
    """
    out: Dict[str, np.ndarray] = {}
    for pv, pr, pa in grid:
        cfg = replace(base, pen_violation=pv, pen_retard=pr, pen_avance=pa)
        label = f"pen_v={pv}, pen_r={pr}, pen_a={pa}"
        out[label] = multiple_runs(instance_path, cfg, nb_runs, rng_seed)
    return out

def sweep_temperatures(
    instance_path: str,
    base: sa.SAConfig,
    temp_list,
    nb_runs: int = 50,
    rng_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Balaye une liste de températures de départ et d'arrivé.

    ``temp_list`` est une liste de couples ``(T_start, T_end)``.
    """
    out: Dict[str, np.ndarray] = {}
    for ts, te in temp_list:
        cfg = replace(base, T_start = ts, T_end = te)
        label = f"start_T={ts}, end_T={te}"
        out[label] = multiple_runs(instance_path, cfg, nb_runs, rng_seed)
    return out
