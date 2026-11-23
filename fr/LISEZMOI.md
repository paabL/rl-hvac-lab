# Prototype du lab RL : un thermostat, mille maisons

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/prototype_lab_rl.ipynb)


Prototype à tester et recadrer, pas la version finale. Tout tourne sur CPU.

## Lancer

**Colab** : cliquer sur le badge ci-dessus, puis exécuter les cellules dans l'ordre. La première télécharge seule les fichiers du lab. Si elle n'atteint pas GitHub, elle affiche un bouton pour téléverser `lab_rl_prototype.zip`.

**En local** : dans ce dossier, avec Python 3.11 et `numpy pandas scikit-learn matplotlib torch ipywidgets`, ouvrir le notebook dans Jupyter.

Réglages dans la première cellule :

- `RUN_TRAINING = True` réentraîne l'agent « classique » en direct (environ 2 min sur CPU).
- `METEO_REELLE = True` remplace la météo synthétique par les janviers 2015-2024 de Montréal (Open-Meteo, Internet requis). Les agents fournis ont été entraînés sur la météo synthétique.
- `SHOW_WIDGETS = False` remplace les curseurs par des figures statiques.

## Contenu

| Fichier | Rôle |
| --- | --- |
| `lab_rl_etudiant.ipynb` | **la version étudiante à compléter au tableau**, sans réponses [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/lab_rl_etudiant.ipynb) |
| `prototype_lab_rl.ipynb` | le notebook avec les corrections, sans sorties |
| `prototype_lab_rl_execute.ipynb` | le même, exécuté, pour lire les résultats sans rien lancer |
| `lab_utils.py` | simulateur RC2, thermostats, agents, PPO, REINFORCE, diagnostics, jouet du col |
| `blanks.py` | les `___` des exercices de la version étudiante |
| `schemas.py` | les trois schémas (maison, boucle du RL, agent à mémoire), en français et en anglais |
| `entrainer_agents.py` | recalcule un agent : `python entrainer_agents.py contexte 400` (ajouter `--reprendre` pour continuer) |
| `agents/` | les agents précalculés (`.pt`) et leurs courbes d'apprentissage (`.json`) |
| `meteo_montreal.csv` | la météo utilisée (synthétique pour l'instant) |
