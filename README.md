# Intelligence Artificielle Hybride pour l'Ordonnancement

Projet tutore L3 MIASHS - Prof: Romain Guillaume

Réalisé par : Adam Fennassi, Oussama Touili

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
├── benchmark.py                    # Compare les 3 methodes en ligne de commande
├── dashboard.py                    # Genere les graphiques de comparaison (PNG)
├── manage.py                       # Point d'entree Django
│
├── algorithms/                     # Les 3 algorithmes d'ordonnancement
│   ├── cp_sat_solver.py            # Solveur PPC pur (CP-SAT)
│   ├── ml_solver.py                # Solveur ML pur (PyTorch)
│   └── hybrid_solver.py            # Solveur hybride PPC + ML
│
├── webapp/                         # Interface web Django
│   ├── views.py                    # Logique des pages
│   ├── models.py                   # Modele BenchmarkResult (stockage resultats)
│   ├── urls.py                     # Routes
│   ├── forms.py                    # Formulaire upload CSV
│   └── templates/webapp/           # Templates HTML (Bootstrap 5)
│       ├── base.html               # Layout commun (navbar, footer)
│       ├── index.html              # Page d'accueil + resultats recents
│       ├── upload.html             # Upload CSV + resolution CP-SAT
│       ├── benchmark_form.html     # Lancer un benchmark (3 methodes)
│       ├── dashboard.html          # Dashboard KPI avec graphiques
│       └── gantt.html              # Diagramme de Gantt + detail taches
│
└── config/                         # Configuration Django
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

## Interface web

L'application Django propose 4 pages :

### 1. Accueil (`/`)
- Vue d'ensemble du projet
- Liste des resultats recents avec boutons de suppression (individuel ou tout supprimer)

### 2. Upload CSV (`/upload/`)
- Charger un fichier CSV avec des taches et des machines
- Resolution avec **CP-SAT** (~1 seconde)
- Affiche le diagramme de Gantt et les KPI (makespan, objectif, temps)

### 3. Benchmark (`/benchmark/`)
- Lance la comparaison des **3 methodes** (CP-SAT, ML, Hybride) sur 3 niveaux de difficulte
- Parametrable : nombre d'instances d'entrainement et de test
- Genere le dashboard KPI automatiquement

### 4. Dashboard (`/dashboard/`)
- Graphiques de comparaison : makespan, temps, objectif, utilisation machines
- Courbes d'entrainement du modele ML
- Tableau recapitulatif avec tous les KPI

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

## Installation

```bash
pip install django ortools torch matplotlib numpy pandas
```

## Utilisation

### Option 1 : Interface web (recommande)

```bash
cd Projet_tutore_AI
python manage.py migrate
python manage.py runserver
```

Ouvrir **http://127.0.0.1:8000/** dans le navigateur.

- **Upload CSV** : resoudre un probleme avec CP-SAT, voir le Gantt
- **Benchmark** : comparer les 3 methodes, voir le dashboard KPI

### Option 2 : Ligne de commande

```bash
# Lancer le benchmark (~2 min)
python benchmark.py

# Generer le dashboard (PNG)
python dashboard.py

# Generer le Gantt comparatif
python dashboard.py --gantt
```

## Format CSV

```csv
task_name,duration,predecessors,relase_date,due_date
task_a_1,60,none,0,300
task_a_2,40,task_a_1,60,300
task_b_1,50,none,0,250
task_b_2,30,task_b_1,50,250

MACHINES,"m_1,m_2",,,
```

## KPI mesures

| KPI | Description | Meilleur si... |
|-----|-------------|----------------|
| **Makespan** | Date de fin de la derniere tache | Plus petit |
| **Objectif** | Somme des dates de debut de toutes les taches | Plus petit |
| **Temps de resolution** | Duree du calcul en secondes | Plus petit |
| **Utilisation machines** | % du temps ou les machines travaillent | Plus grand |
| **Taux de resolution** | % d'instances ou une solution est trouvee | Plus grand |

## Technologies

- **Python 3.8+**
- **Django** - interface web
- **Google OR-Tools (CP-SAT)** - solveur de programmation par contraintes
- **PyTorch** - reseau de neurones pour le ML
- **Matplotlib** - visualisation (Gantt, dashboard KPI)
- **Bootstrap 5** - interface responsive

## Auteurs

Projet tutore L3 MIASHS - 2025
Prof : Romain Guillaume, Maitre de conference HDR, IRIT / ANITI
