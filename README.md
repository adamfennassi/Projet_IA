# Intelligence Artificielle Hybride pour l'Ordonnancement

Projet tutore L3 MIASHS - Prof: Romain Guillaume (IRIT / ANITI)

## Objectif

Comparer 3 approches pour resoudre des problemes d'ordonnancement sur machines paralleles :

| Approche | Technologie | Avantage | Inconvenient |
|----------|-------------|----------|--------------|
| **PPC pure** | CP-SAT (OR-Tools) | Solution optimale | Lent sur les gros problemes |
| **ML pur** | PyTorch | Tres rapide (~0.002s) | Solution approximative |
| **Hybride PPC + ML** | CP-SAT + PyTorch | Qualite optimale + convergence acceleree | Necessite un entrainement prealable |

## Structure du projet

```
Projet_tutore_AI/
├── model_machine_parallele.ipynb   # Notebook original (modele CP-SAT du client)
├── generator.py                    # Generateur de datasets (3 niveaux de difficulte)
├── benchmark.py                    # Compare les 3 methodes et collecte les KPI
├── dashboard.py                    # Genere les graphiques de comparaison
├── benchmark_results.json          # Resultats bruts du benchmark
├── dashboard_kpi.png               # Dashboard visuel des KPI
└── algorithms/
    ├── cp_sat_solver.py            # Solveur PPC pur (CP-SAT)
    ├── ml_solver.py                # Solveur ML pur (PyTorch)
    └── hybrid_solver.py            # Solveur hybride PPC + ML
```

## Comment ca marche

### Le probleme d'ordonnancement

On a des **taches** a executer sur des **machines**. Chaque tache a :
- Une duree
- Une date de disponibilite (release date)
- Une echeance (due date)
- Un predecesseur eventuel (une tache qui doit finir avant)

Le but : affecter les taches aux machines et determiner quand les lancer, en **minimisant la somme des dates de debut**.

### Les 3 approches

**1. CP-SAT (PPC pure)** - `algorithms/cp_sat_solver.py`
- Modelise le probleme avec des variables, contraintes et un objectif
- Le solveur explore toutes les possibilites et trouve la solution optimale
- Garanti optimal mais le temps explose quand le probleme grandit

**2. ML pur (PyTorch)** - `algorithms/ml_solver.py`
- Un reseau de neurones a 2 tetes entraine sur des solutions CP-SAT :
  - Tete 1 : predit sur quelle machine affecter chaque tache
  - Tete 2 : predit la priorite de chaque tache (ordre de dispatching)
- En inference, un algorithme glouton construit la solution a partir des predictions
- Tres rapide mais pas toujours optimal

**3. Hybride PPC + ML** - `algorithms/hybrid_solver.py`
- Le ML predit une solution initiale (affectation + priorites)
- Ces predictions sont injectees comme **hints** dans CP-SAT
- CP-SAT demarre pres de l'optimum au lieu de partir de zero
- Combine la qualite de la PPC avec la rapidite du ML

### Le pipeline

```
generator.py          Genere des problemes (facile / moyen / difficile)
      |
      v
benchmark.py          Entraine le ML, teste les 3 methodes, collecte les KPI
      |
      v
benchmark_results.json    Donnees brutes (makespan, temps, objectif, utilisation)
      |
      v
dashboard.py          Genere les graphiques de comparaison
      |
      v
dashboard_kpi.png     Dashboard visuel final
```

## Installation

```bash
pip install ortools torch matplotlib numpy pandas
```

## Utilisation

### Etape 1 : Lancer le benchmark (~10 min, une seule fois)
```bash
cd Projet_tutore_AI
python benchmark.py
```

Ce script :
1. Genere 200 instances et entraine le reseau de neurones dessus (~10 min)
2. Teste les 3 methodes sur 10 instances par niveau de difficulte
3. Affiche un tableau comparatif dans le terminal
4. Sauvegarde les resultats dans `benchmark_results.json`

### Etape 2 : Generer le dashboard (instantane)
```bash
python dashboard.py
```

Genere `dashboard_kpi.png` avec 6 graphiques + un tableau recapitulatif.

### (Optionnel) Regenerer les datasets
```bash
python generator.py
```

Cree `dataset_facile.csv`, `dataset_moyen.csv`, `dataset_difficile.csv`.

## KPI mesures

| KPI | Description | Meilleur si... |
|-----|-------------|----------------|
| **Makespan** | Date de fin de la derniere tache | Plus petit |
| **Objectif** | Somme des dates de debut de toutes les taches | Plus petit |
| **Temps de resolution** | Duree du calcul en secondes | Plus petit |
| **Utilisation machines** | % du temps ou les machines travaillent | Plus grand |
| **Taux de resolution** | % d'instances ou une solution est trouvee | Plus grand |

## Niveaux de difficulte

| Niveau | Taches | Machines | Marge (slack) | Description |
|--------|--------|----------|---------------|-------------|
| Facile | 6 | 4 | 60% | Beaucoup de ressources et de flexibilite |
| Moyen | 10 | 3 | 30% | Ressources moderees |
| Difficile | 16 | 2 | 10% | Peu de machines, contraintes serrees |

## Technologies

- **Python 3.8+**
- **Google OR-Tools (CP-SAT)** - solveur de programmation par contraintes
- **PyTorch** - reseau de neurones pour le ML
- **Matplotlib** - visualisation des KPI
- **NumPy / Pandas** - manipulation de donnees

## Auteurs

Projet tutore L3 MIASHS - 2025
Client : Romain Guillaume, Maitre de conference HDR, IRIT / ANITI
