# Lab 10 · RL for home heating (prototype)

*Version française plus bas.*

**Status: prototype, not validated yet.** The numbers come from one seed, the weather is synthetic, and the content still needs to be reviewed by the teaching team.

**Do not share these notebooks with students as they are:** they contain the solutions to the five exercises.

One thermostat, a thousand houses: a 2-hour lab that climbs from classic RL on one house to a robust memoryless policy, then memory, in-context learning, and adaptation when the house changes (meta-RL). Everything runs on CPU.

| Folder | Language | Start with |
| --- | --- | --- |
| [`en/`](en/) | English, with instructor notes and answers | [`en/README.md`](en/README.md) |
| [`fr/`](fr/) | French, shorter, no instructor notes yet | [`fr/LISEZMOI.md`](fr/LISEZMOI.md) |

**Colab.** English: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/en/rl_lab_prototype_en.ipynb) French: [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/prototype_lab_rl.ipynb)

Click a badge and run the cells in order. The first cell downloads the lab's files by itself. If it cannot reach GitHub, it asks for the language's zip (`en/rl_lab_en.zip` or `fr/lab_rl_prototype.zip`).

**After changing a file,** rebuild the zips with `python tools/sync_to_team.py --zips`, since the notebooks download them.

---

# Labo 10 · RL pour le chauffage (prototype)

**Statut : prototype, pas encore validé.** Les chiffres viennent d'une seule graine, la météo est synthétique et le contenu doit encore être relu par l'équipe.

**Ne pas distribuer ces notebooks tels quels aux étudiant·e·s :** ils contiennent les corrections des cinq exercices.

Un thermostat, mille maisons : un labo de 2 heures qui monte du RL classique sur une maison à une politique robuste sans mémoire, puis à la mémoire, à l'apprentissage en contexte et à l'adaptation quand la maison change (méta-RL). Tout tourne sur CPU.

| Dossier | Langue | Commencer par |
| --- | --- | --- |
| [`fr/`](fr/) | français, plus court, sans notes pour l'animateur pour l'instant | [`fr/LISEZMOI.md`](fr/LISEZMOI.md) |
| [`en/`](en/) | anglais, avec notes pour l'animateur et réponses | [`en/README.md`](en/README.md) |

**Colab.** Français : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/prototype_lab_rl.ipynb) Anglais : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/en/rl_lab_prototype_en.ipynb)

Cliquez sur un badge et exécutez les cellules dans l'ordre. La première télécharge seule les fichiers du lab. Si elle n'atteint pas GitHub, elle demande le zip de votre langue (`fr/lab_rl_prototype.zip` ou `en/rl_lab_en.zip`).

**Après avoir modifié un fichier,** reconstruisez les zips avec `python tools/sync_to_team.py --zips`, car les notebooks les téléchargent.
