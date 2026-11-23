# Prototype du lab RL : un thermostat, mille maisons

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/prototype_lab_rl.ipynb)

> **Pour les TA.** Le dépôt étant privé, autorisez d'abord Colab une fois : **Fichier > Ouvrir un notebook > GitHub**, puis cochez **Include private repos**. Il n'ouvre que le notebook : téléchargez aussi [`lab_rl_prototype.zip`](https://github.com/paabL/rl-hvac-lab/blob/main/fr/lab_rl_prototype.zip), puis choisissez-le dans la première cellule de code.


Prototype à tester et recadrer, pas la version finale. Tout tourne sur CPU.

## Lancer

**Colab** : ouvrir `prototype_lab_rl.ipynb` (Fichier > Importer un notebook), puis lancer la cellule d'installation : elle affiche un bouton pour téléverser `lab_rl_prototype.zip` et le décompresse.

**En local** : dans ce dossier, avec Python 3.11 et `numpy pandas scikit-learn matplotlib torch ipywidgets`, ouvrir le notebook dans Jupyter.

Réglages dans la première cellule :

- `RUN_TRAINING = True` réentraîne l'agent « classique » en direct (environ 2 min sur CPU).
- `METEO_REELLE = True` remplace la météo synthétique par les janviers 2015-2024 de Montréal (Open-Meteo, Internet requis). Les agents fournis ont été entraînés sur la météo synthétique.
- `SHOW_WIDGETS = False` remplace les curseurs par des figures statiques.

## Contenu

| Fichier | Rôle |
| --- | --- |
| `prototype_lab_rl.ipynb` | le notebook, sans sorties |
| `prototype_lab_rl_execute.ipynb` | le même, exécuté, pour lire les résultats sans rien lancer |
| `lab_utils.py` | simulateur RC2, thermostats, agents, PPO, REINFORCE, diagnostics, jouet du col |
| `schemas.py` | les trois schémas (maison, boucle du RL, agent à mémoire), en français et en anglais |
| `entrainer_agents.py` | recalcule un agent : `python entrainer_agents.py contexte 400` (ajouter `--reprendre` pour continuer) |
| `agents/` | les agents précalculés (`.pt`) et leurs courbes d'apprentissage (`.json`) |
| `meteo_montreal.csv` | la météo utilisée (synthétique pour l'instant) |
