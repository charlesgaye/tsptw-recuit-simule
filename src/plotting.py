"""Tracés de progression et de comparaison entre configurations."""

from __future__ import annotations

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from . import sa


def plot_progress(trace: sa.SATrace) -> None:
    """Quatre sous-graphiques : score, violations, température, score vs violations."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(trace.iterations, trace.best_scores, "g-", linewidth=2, label="Meilleur score")
    ax1.plot(trace.iterations, trace.current_scores, "b-", alpha=0.5, label="Score courant")
    ax1.plot(trace.iterations, trace.sample_scores, "r-", alpha=0.5, label="Score voisin échantillonné")
    ax1.set_xlabel("Itérations")
    ax1.set_ylabel("Score")
    ax1.set_title("Évolution du score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(trace.iterations, trace.best_violations, "r-", linewidth=2, label="Violations (meilleure)")
    ax2.plot(trace.iterations, trace.current_violations, "orange", alpha=0.5, label="Violations (courante)")
    ax2.set_xlabel("Itérations")
    ax2.set_ylabel("Nombre de villes hors fenêtre")
    ax2.set_title("Évolution des violations")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.semilogy(trace.iterations, trace.temperatures, "purple", linewidth=2)
    ax3.set_xlabel("Itérations")
    ax3.set_ylabel("Température (log)")
    ax3.set_title("Refroidissement géométrique")
    ax3.grid(True, alpha=0.3)

    sc = ax4.scatter(trace.best_violations, trace.best_scores, c=trace.iterations, cmap="viridis", alpha=0.6)
    ax4.set_xlabel("Violations")
    ax4.set_ylabel("Score")
    ax4.set_title("Score vs violations (couleur = itération)")
    plt.colorbar(sc, ax=ax4, label="Itération")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ci(results: Dict[str, np.ndarray], skip_start: bool = False, n_markers: int = 11) -> None:
    """Trace des moyennes et IC à 95% sur la moyenne pour plusieurs configurations.

    ``results[label]`` est une matrice ``(nb_runs, n_points)``.

    Note : l'IC affiché est ``1.96 * sigma / sqrt(n)``, intervalle pour la
    moyenne d'UNE configuration à un instant donné. Il ne reflète pas la
    variance inter-configurations si l'état aléatoire est partagé entre
    plusieurs appels successifs.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    n_configs = len(results)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_configs, 1)))
    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "v", "D", "p", "*", "h")

    for idx, (label, runs) in enumerate(results.items()):
        runs = np.asarray(runs)
        n_runs, n_points = runs.shape

        offset = 20 if skip_start else 0
        mean = runs[:, offset:].mean(axis=0)
        n_eff = n_points - offset
        if n_eff <= 1:
            continue
        x = np.arange(n_eff)
        idxs = np.linspace(0, n_eff - 1, n_markers, dtype=int)

        std = np.array([
            np.sqrt(((runs[:, i + offset] - mean[i]) ** 2).sum() / max(n_runs - 1, 1))
            for i in idxs
        ])
        yerr = 1.96 / np.sqrt(n_runs) * std

        style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        color = colors[idx]

        ax.plot(x, mean, style, color=color, label=label, linewidth=1.5)
        ax.errorbar(idxs, mean[idxs], yerr=yerr, color=color, fmt=marker,
                    capsize=5, alpha=0.7, markersize=4)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlabel("Étape (×100 itérations)")
    ax.set_ylabel("Score (métrique de référence)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
