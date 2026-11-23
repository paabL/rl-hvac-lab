# Lab RL : un thermostat, mille maisons (prototype v3)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/prototype_lab_rl.ipynb) version formateur, avec les corrections

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/lab_rl_etudiant.ipynb) version étudiante, à compléter au tableau

Prototype à tester et recadrer, pas la version finale. Tout tourne sur CPU.

## Lancer

**Colab** : cliquer sur un badge, puis exécuter les cellules dans l'ordre. La première installe Stable-Baselines3 si besoin et télécharge seule les fichiers du lab. Si elle n'atteint pas GitHub, elle affiche un bouton pour téléverser `lab_rl_prototype.zip`.

**En local** : dans ce dossier, avec Python 3.10 ou plus et `numpy pandas scikit-learn matplotlib torch ipywidgets gymnasium stable-baselines3`, ouvrir le notebook dans Jupyter.

Réglages dans la première cellule :

- `RUN_TRAINING = True` réentraîne l'agent « classique » en direct (environ 1 min 30 sur CPU).
- `METEO_REELLE = True` remplace la météo synthétique par les janviers 2015-2024 de Montréal (Open-Meteo, Internet requis). Les agents fournis ont été entraînés sur la météo synthétique.
- `SHOW_WIDGETS = False` remplace les curseurs par des figures statiques.

## Contenu

| Fichier | Rôle |
| --- | --- |
| `lab_rl_etudiant.ipynb` | **la version étudiante à compléter au tableau** : chaque exercice contient des `___` ; tant qu'il est vide, la suite tourne avec la version du labo |
| `prototype_lab_rl.ipynb` | le notebook avec les corrections, sans sorties |
| `prototype_lab_rl_execute.ipynb` | le même, exécuté, pour lire les résultats sans rien lancer |
| `lab_utils.py` | simulateur de maisons, thermostats, REINFORCE, jouet du col |
| `rl_sb3.py` | la maison à la convention Gymnasium, les agents PPO sans mémoire de Stable-Baselines3, l'évaluation et les diagnostics de mémoire |
| `gru_ppo.py` | notre PPO à mémoire : GRU, actor, critic et tête de prédiction (à appeler, pas à lire) |
| `blanks.py` | les `___` et la vérification des exercices de la version étudiante |
| `schemas.py`, `images/` | les schémas du notebook, en français et en anglais (dessins du robot et des bâtiments repris du deck PluRL) |
| `entrainer_agents.py` | recalcule un agent : `python entrainer_agents.py contexte 400` (ajouter `--reprendre` pour continuer) |
| `agents/` | les agents précalculés (`.zip` de SB3 sans mémoire, `.pt` pour nos GRU) et leurs courbes d'apprentissage (`.json`) |
| `meteo_montreal.csv` | la météo utilisée (synthétique pour l'instant) |
| `lab_rl_prototype.zip` | tout ce qu'il faut à Colab, sans le notebook exécuté |

L'antisèche d'une page (code à taper, avec ce qu'il faut dire pour chaque ligne) est dans `../antiseche/`.
