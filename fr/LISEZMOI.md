# Prototype du lab RL : un thermostat, mille maisons

Prototype à tester et recadrer, pas la version finale. Tout tourne sur CPU.

## Lancer

**Colab** : ouvrir `prototype_lab_rl.ipynb` (Fichier > Importer un notebook), exécuter la première cellule de code, choisir `lab_rl_prototype.zip` avec le bouton **Choose files**, puis exécuter les autres cellules dans l'ordre. Colab efface les fichiers téléversés quand l'environnement redémarre : il faut alors téléverser le zip de nouveau.

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
| `lab_rl_prototype.zip` | tout ce qu'il faut pour Colab : les fichiers ci-dessous et le notebook |
| `lab_utils.py` | simulateur RC2, thermostats, agents, PPO, REINFORCE, diagnostics, jouet du col |
| `entrainer_agents.py` | recalcule un agent : `python entrainer_agents.py contexte 400` (ajouter `--reprendre` pour continuer) |
| `agents/` | les agents précalculés (`.pt`) et leurs courbes d'apprentissage (`.json`) |
| `meteo_montreal.csv` | la météo utilisée (synthétique pour l'instant) |
