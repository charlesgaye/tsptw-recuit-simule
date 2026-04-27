"""Point d'entrée — TSPTW par recuit simulé.

Quatre modes :
    --mode run       : un seul recuit avec affichage des graphes de progression.
    --mode sweep_p   : balayage de vecteurs de probabilités de voisinage.
    --mode sweep_pen : balayage de pénalités.
    --mode sweep_temp: balayage du couple (T_start, T_end).

Exemples :
    python main.py --mode run --instance data/inst_concours
    python main.py --mode sweep_p --instance data/inst1 --runs 30
"""

from __future__ import annotations

import argparse

from src import experiments, io_utils, plotting, sa


PROBA_DEFAULTS = [
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.25, 0.25, 0.25, 0.25),
    (0.0, 0.1, 0.75, 0.15),
]

PENALTY_GRID = [
    (350, 30, 5),
    (350, 30, 10),
    (450, 30, 10),
]

TEMP_GRID = [
    (10**logts,10**logte) for logts in range(1,6,4) for logte in range(-3,1) 
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TSPTW — recuit simulé")
    p.add_argument("--mode", choices=("run", "sweep_p", "sweep_pen", "sweep_temp"), default="run")
    p.add_argument("--instance", default="data/inst_concours")
    p.add_argument("--T-start", type=float, default=5_000.0)
    p.add_argument("--T-end", type=float, default=0.1)
    p.add_argument("--iters", type=int, default=5_000_000)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-sol", default=None, help="Chemin de sortie pour le fichier .sol")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = sa.SAConfig(
        T_start=args.T_start,
        T_end=args.T_end,
        nb_iterations=args.iters,
    )

    if args.mode == "run":
        result = sa.simulated_annealing(args.instance, cfg, track_progress=True, rng_seed=args.seed)
        print(f"Score final : {result.score:.1f}  (violations = {result.n_violations})")
        if args.out_sol:
            io_utils.write_solution(args.out_sol, result.solution, score=int(result.score))
            print(f"Solution écrite : {args.out_sol}")
        plotting.plot_progress(result.trace)

    elif args.mode == "sweep_p":
        results = experiments.sweep_proba(args.instance, cfg, PROBA_DEFAULTS, args.runs, rng_seed=args.seed)
        plotting.plot_ci(results, skip_start=True)

    elif args.mode == "sweep_pen":
        results = experiments.sweep_penalties(args.instance, cfg, PENALTY_GRID, args.runs, rng_seed=args.seed)
        plotting.plot_ci(results, skip_start=True)

    elif args.mode == "sweep_temp":
        results = experiments.sweep_temperatures(args.instance, cfg, TEMP_GRID, args.runs, rng_seed=args.seed)
        plotting.plot_ci(results, skip_start=True)

if __name__ == "__main__":
    main()
