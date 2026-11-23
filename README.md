# Lab 10 · RL for home heating (prototype v3)

*Version française plus bas.*

**Status: prototype, not validated yet.** The numbers come from one seed, the weather is synthetic, and the content still needs to be reviewed by the teaching team.

**For students, use the student notebooks below:** the other notebooks contain the solutions to the five exercises.

One thermostat, a thousand houses: a 2-hour lab that climbs from classic RL on one house to a robust memoryless policy, then memory, in-context learning, and adaptation when the house changes (meta-RL). Everything runs on CPU. Memoryless agents use Stable-Baselines3 PPO; memory agents use a short in-house GRU PPO with a prediction head.

| Folder | Language | Start with |
| --- | --- | --- |
| [`en/`](en/) | English, with instructor notes and answers | [`en/README.md`](en/README.md) |
| [`fr/`](fr/) | French, shorter, no instructor notes yet | [`fr/LISEZMOI.md`](fr/LISEZMOI.md) |
| [`antiseche/`](antiseche/) | one-page cheat sheet, FR and EN (HTML for the screen, PDF to print) | `cheat_sheet_en.pdf`, `antiseche_fr.pdf` |

**Colab.** English: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/en/rl_lab_prototype_en.ipynb) French: [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/prototype_lab_rl.ipynb)

Click a badge and run the cells in order. The first cell installs Stable-Baselines3 if needed and downloads the lab's files by itself. If it cannot reach GitHub, it asks for the language's zip (`en/rl_lab_en.zip` or `fr/lab_rl_prototype.zip`).

**Student version, to fill in live on the board.** English: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/en/rl_lab_student_en.ipynb) French: [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/lab_rl_etudiant.ipynb)

Each exercise has `___` to replace. While it is empty, the notebook keeps running with the lab's version.

**After changing a file,** run `python tools/build_student.py`, then `python tools/sync_to_team.py --zips`, since the notebooks download the zips.

---

# Labo 10 · RL pour le chauffage (prototype v3)

**Statut : prototype, pas encore validé.** Les chiffres viennent d'une seule graine, la météo est synthétique et le contenu doit encore être relu par l'équipe.

**Pour les étudiant·e·s, utilisez les notebooks étudiants ci-dessous :** les autres contiennent les corrections des cinq exercices.

Un thermostat, mille maisons : un labo de 2 heures qui monte du RL classique sur une maison à une politique robuste sans mémoire, puis à la mémoire, à l'in-context learning et à l'adaptation quand la maison change (meta-RL). Tout tourne sur CPU. Les agents sans mémoire utilisent le PPO de Stable-Baselines3 ; les agents à mémoire, un court PPO à GRU maison avec une tête de prédiction.

| Dossier | Langue | Commencer par |
| --- | --- | --- |
| [`fr/`](fr/) | français, plus court, sans notes pour l'animateur pour l'instant | [`fr/LISEZMOI.md`](fr/LISEZMOI.md) |
| [`en/`](en/) | anglais, avec notes pour l'animateur et réponses | [`en/README.md`](en/README.md) |
| [`antiseche/`](antiseche/) | l'antisèche d'une page, FR et EN (HTML pour l'écran, PDF à imprimer) | `antiseche_fr.pdf`, `cheat_sheet_en.pdf` |

**Colab.** Français : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/prototype_lab_rl.ipynb) Anglais : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/en/rl_lab_prototype_en.ipynb)

Cliquez sur un badge et exécutez les cellules dans l'ordre. La première installe Stable-Baselines3 si besoin et télécharge seule les fichiers du lab. Si elle n'atteint pas GitHub, elle demande le zip de votre langue (`fr/lab_rl_prototype.zip` ou `en/rl_lab_en.zip`).

**Version étudiante, à compléter en direct au tableau.** Français : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/fr/lab_rl_etudiant.ipynb) Anglais : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/en/rl_lab_student_en.ipynb)

Chaque exercice contient des `___` à remplacer. Tant qu'il est vide, le notebook continue avec la version du labo.

**Après avoir modifié un fichier,** lancez `python tools/build_student.py`, puis `python tools/sync_to_team.py --zips`, car les notebooks téléchargent les zips.
