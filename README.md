# TSPTW — recuit simulé

Implémentation d'un recuit simulé pour le problème du voyageur de commerce
avec fenêtres temporelles (TSPTW), réalisée dans le cadre d'un projet
académique (compétition interne de métaheuristiques à CentraleSupélec).

## Résultats clés

- **1ère place** de promotion en compétition interne de métaheuristiques.
- Sur les instances de validation (`inst1` ~20 villes, `inst2` ~50 villes),
  la solution optimale fournie avec le sujet est retrouvée systématiquement.
- Sur l'instance de concours (`inst_concours`, ~100 villes), une solution
  admissible (sans retard ni avance) est obtenue de façon répétable.
- Sur `inst3` (~200 villes), aucune solution admissible n'a été atteinte —
  les meilleurs essais conservent 2 à 3 villes en retard. Documenté dans
  le rapport.

Configuration finale retenue :

| Paramètre | Valeur |
|---|---|
| `T_start` / `T_end` | 5000 / 0.1 |
| `nb_iterations` | 5 000 000 |
| `proba_voisinage` (large_swap, block_reverse, scramble, insert_range) | (0, 0.1, 0.75, 0.15) |
| Pénalités (violation, retard, avance) | (350, 30, 10) |

Voir [`RapportTSPTW.pdf`](RapportTSPTW.pdf) pour l'analyse complète :
sensibilité aux paramètres, intervalles de confiance sur ~30 runs,
comparaison avec une approche ACO concurrente, et discussion de l'écart à
l'optimum publiquement connu sur l'instance de concours (découvert après
rendu).

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

## Reproduire les expériences du rapport

Toutes les commandes ci-dessous fixent une graine pour la reproductibilité.
Un balayage prend de quelques minutes à plusieurs dizaines de minutes selon
`--runs` et `--iters` ; pour un essai rapide, réduire `--iters` à 100000.

```
# Recuit unique sur l'instance de concours, configuration finale
python main.py --mode run --instance data/inst_concours --iters 5000000 --seed 0

# Balayage des distributions de voisinage (figures de la section sur le voisinage)
python main.py --mode sweep_p --instance data/inst1 --runs 30 --seed 0

# Balayage des pénalités
python main.py --mode sweep_pen --instance data/inst1 --runs 30 --seed 0

# Balayage des températures
python main.py --mode sweep_temp --instance data/inst1 --runs 30 --seed 0
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

## Note technique — bug détecté après rendu

En relisant la fonction de score après le rendu, j'ai identifié un bug dans
le calcul de la pénalité d'avance : `duree` est écrasée par `next_start`
avant le calcul `next_start - duree`, qui vaut donc systématiquement 0.
Conséquence : `t_avance` est numériquement nul et le terme
`pen_avance * t_avance` ne contribue jamais au score. Les balayages de
`pen_avance` rapportés dans `--mode sweep_pen` sont donc en pratique neutres
sur ce terme.

J'ai choisi de **préserver la version originale** dans
`scoring.compute_score` pour garder les chiffres traçables au rapport rendu,
et de fournir une **version corrigée** (`scoring.compute_score_fixed`) à
titre documentaire. Aucune des expériences du rapport n'utilise la version
corrigée. Voir aussi la section dédiée du rapport (`ssec:bug-avance`).

## Notes

- Les matrices de distances sont mises en cache dans `data/<instance>_dist.npy`
  par `io_utils.load_or_build_distance_matrix`.
- `experiments.multiple_runs` accepte un `rng_seed` ; **toujours le fixer**
  lorsqu'on compare plusieurs configurations, pour éviter que l'état global
  de `np.random` ne diverge entre configurations (cf. note dans le rapport,
  section sur la pénalité d'avance).
- Le module `io_utils` reprend la lecture d'instance fournie avec le sujet
  (fichier `scoreEtudiant.py`), réécrite de façon minimale.
