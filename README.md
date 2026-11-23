# Lab 10 · RL for home heating (prototype)

*Version française plus bas.*

**Status: prototype, not validated yet.** The numbers come from one seed, the weather is synthetic, and the content still needs to be reviewed by the teaching team.

**Do not share these notebooks with students as they are:** they contain the solutions to the five exercises.

One thermostat, a thousand houses: a 2-hour lab that climbs from classic RL on one house to a robust memoryless policy, then memory, in-context learning, and adaptation when the house changes (meta-RL). Everything runs on CPU.

| Folder | Language | Start with |
| --- | --- | --- |
| [`en/`](en/) | English, with instructor notes and answers | [`en/README.md`](en/README.md) |
| [`fr/`](fr/) | French, shorter, no instructor notes yet | [`fr/LISEZMOI.md`](fr/LISEZMOI.md) |

**Colab.** English: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milarobotlearningcourse/data_science_ift6758/blob/main/labs/Lab-10-RL/en/rl_lab_prototype_en.ipynb) French: [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milarobotlearningcourse/data_science_ift6758/blob/main/labs/Lab-10-RL/fr/prototype_lab_rl.ipynb)

This repository is private, so Colab asks once for permission to read your private GitHub repositories. It opens only the notebook, so also download the language's zip (`en/rl_lab_en.zip` or `fr/lab_rl_prototype.zip`) and pick it in the first code cell.

**After changing a file,** rebuild that language's zip so it stays in sync with the folder.

---

# Labo 10 · RL pour le chauffage (prototype)

**Statut : prototype, pas encore validé.** Les chiffres viennent d'une seule graine, la météo est synthétique et le contenu doit encore être relu par l'équipe.

**Ne pas distribuer ces notebooks tels quels aux étudiant·e·s :** ils contiennent les corrections des cinq exercices.

Un thermostat, mille maisons : un labo de 2 heures qui monte du RL classique sur une maison à une politique robuste sans mémoire, puis à la mémoire, à l'apprentissage en contexte et à l'adaptation quand la maison change (méta-RL). Tout tourne sur CPU.

| Dossier | Langue | Commencer par |
| --- | --- | --- |
| [`fr/`](fr/) | français, plus court, sans notes pour l'animateur pour l'instant | [`fr/LISEZMOI.md`](fr/LISEZMOI.md) |
| [`en/`](en/) | anglais, avec notes pour l'animateur et réponses | [`en/README.md`](en/README.md) |

**Colab.** Français : [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milarobotlearningcourse/data_science_ift6758/blob/main/labs/Lab-10-RL/fr/prototype_lab_rl.ipynb) Anglais : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milarobotlearningcourse/data_science_ift6758/blob/main/labs/Lab-10-RL/en/rl_lab_prototype_en.ipynb)

Ce dépôt étant privé, Colab demande une fois l'autorisation de lire vos dépôts GitHub privés. Il n'ouvre que le notebook : téléchargez aussi le zip de votre langue (`fr/lab_rl_prototype.zip` ou `en/rl_lab_en.zip`), puis choisissez-le dans la première cellule de code.

**Après avoir modifié un fichier,** reconstruisez le zip de la langue concernée pour qu'il reste à jour.
