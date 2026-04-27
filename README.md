# TSPTW — recuit simulé

Implémentation d'un recuit simulé pour le problème du voyageur de commerce
avec fenêtres temporelles (TSPTW), réalisée dans le cadre d'un projet
académique.

## Structure du dépôt

```
TSPTW/
├── main.py               # Point d'entrée CLI (run / sweep_p / sweep_pen / sweep_temp)
├── src/                  # Code source (paquet Python)
│   ├── io_utils.py       # Lecture d'instance, matrice de distances, fichiers .sol
│   ├── scoring.py        # Fonction de score pénalisée
│   ├── neighborhoods.py  # Opérateurs de voisinage + heuristique de réparation
│   ├── sa.py             # Boucle du recuit simulé (SAConfig, simulated_annealing)
│   ├── experiments.py    # Moyennage d'essais, balayages de paramètres
│   └── plotting.py       # Courbes de progression et intervalles de confiance
├── data/                 # Instances inst1, inst2, inst3, inst_concours (+ .sol, _dist.npy)
├── figures/              # Figures utilisées dans le rapport
├── rapport.tex           # Rapport (source LaTeX)
├── rapport.pdf           # Rapport (compilé)
└── RapportTSPTW.pdf      # Version originale du rapport
```

## Dépendances

Python 3.9+ avec `numpy` et `matplotlib`.

```
pip install numpy matplotlib
```

## Format d'instance

Une instance est un fichier texte décrivant `n` villes. L'en-tête est ignoré
jusqu'à la première ligne dont le premier champ est un entier, après quoi
chaque ligne contient `id x y ? wstart wend`. Le dépôt est la ville d'id `0`.

Une solution est un fichier texte contenant la séquence d'ids visités, dépôt
exclu (le dépôt est implicitement ajouté en début et fin).

## Utilisation en ligne de commande

`main.py` expose quatre modes :

```
python main.py --mode run        --instance data/inst_concours --iters 1000000 --seed 0 --out-sol data/inst_concours.sol
python main.py --mode sweep_p    --instance data/inst1 --runs 30 --seed 0
python main.py --mode sweep_pen  --instance data/inst1 --runs 30 --seed 0
python main.py --mode sweep_temp --instance data/inst1 --runs 30 --seed 0
```

Options principales (`python main.py --help` pour la liste complète) :

- `--mode` : `run`, `sweep_p`, `sweep_pen`, `sweep_temp` (défaut : `run`).
- `--instance` : chemin vers l'instance (défaut : `data/inst_concours`).
- `--T-start`, `--T-end`, `--iters` : paramètres de température et nombre
  d'itérations.
- `--runs` : nombre d'essais indépendants pour les modes balayage.
- `--seed` : graine globale ; les balayages utilisent `seed + k` pour
  garantir la même séquence aléatoire entre configurations comparées.
- `--out-sol` : chemin de sortie du fichier `.sol` (mode `run`).

Les grilles de balayage (`PROBA_DEFAULTS`, `PENALTY_GRID`, `TEMP_GRID`) sont
définies en tête de `main.py` ; modifier directement le fichier pour
changer les valeurs explorées.

## Utilisation comme bibliothèque

Depuis la racine du dépôt :

```python
from src import sa

config = sa.SAConfig(
    T_start=5_000.0,
    T_end=0.1,
    nb_iterations=100_000,
    proba_voisinage=(0.0, 0.1, 0.75, 0.15),
    pen_violation=350.0,
    pen_retard=30.0,
    pen_avance=10.0,
)

result = sa.simulated_annealing("data/inst1", config, rng_seed=0)
print("Score final :", result.best_score)
print("Distance    :", result.best_breakdown.distance)
print("Violations  :", result.best_breakdown.n_violations)
```

Pour tracer la progression :

```python
from src import plotting
plotting.plot_progress(result.trace, save_path="progress.png")
```

Pour un balayage avec intervalles de confiance :

```python
from src import experiments, plotting

curves = experiments.sweep_proba(
    "data/inst1",
    base=config,
    proba_list=[(0.0, 0.1, 0.75, 0.15), (0.0, 0.5, 0.5, 0.0)],
    nb_runs=30,
    rng_seed=0,           # IMPORTANT : graine fixée pour comparer à séquence aléatoire identique
)
plotting.plot_ci(curves, save_path="ci.png")
```

## Notes

- Les matrices de distances sont mises en cache dans `data/<instance>_dist.npy`
  par `io_utils.load_or_build_distance_matrix`.
- `experiments.multiple_runs` accepte un `rng_seed` ; **toujours le fixer**
  lorsqu'on compare plusieurs configurations, pour éviter que l'état global
  de `np.random` ne diverge entre configurations (cf. note dans le rapport,
  section sur la pénalité d'avance).
- La fonction `scoring.compute_score` reproduit la fonction utilisée pour
  les expériences du rapport, qui contient un bug rendant la pénalité
  d'avance numériquement inactive. Une version corrigée
  `scoring.compute_score_fixed` est fournie pour référence mais n'est pas
  utilisée par défaut.
- Le module `io_utils` reprend la lecture d'instance fournie avec le sujet
  (fichier `scoreEtudiant.py`), réécrite de façon minimale.
