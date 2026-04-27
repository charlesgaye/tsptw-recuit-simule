"""TSPTW — recuit simulé.

Modules:
    io_utils      : lecture des instances, matrice de distances, fichiers .sol.
    scoring       : fonction de score pénalisée (distance + violations + retard + avance).
    neighborhoods : opérateurs de voisinage et heuristique de réparation.
    sa            : boucle principale du recuit simulé.
    experiments   : moyennage d'essais et intervalles de confiance.
    plotting      : tracés de progression et de comparaison.
"""
