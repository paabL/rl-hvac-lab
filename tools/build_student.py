#!/usr/bin/env python3
"""Build the student notebooks from the instructor notebooks.

The student notebooks have no outputs, no instructor notes and no answers. Each
exercise has ___ to fill in, and the notebook keeps running while they are empty.
Run this after changing an instructor notebook, then rebuild the zips:

    python tools/build_student.py
    python tools/sync_to_team.py --zips

The script stops if an instructor notebook no longer matches what it expects.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BASE = "https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main"
RAW = "https://raw.githubusercontent.com/paabL/rl-hvac-lab/main"

# Each exercise: (solution in the instructor notebook, blank in the student notebook)
EXERCISES = {
    "fr": [
        ('''        # VOTRE CODE ICI : bilan de l'air (pertes, échange avec la masse, pompe, gains) puis de la masse
        dTa = (xi["UA"] * (t_ext - Ta) + HAM * (Tm - Ta) + q_pompe + QINT) / CA
        dTm = HAM * (Ta - Tm) / xi["Cm"]
''', '''        # À COMPLÉTER : bilan de l'air (pertes, échange avec la masse, pompe, gains) puis de la masse
        # Disponibles : xi["UA"], xi["Cm"], HAM, CA, QINT, q_pompe, t_ext
        dTa = ___
        dTm = ___
'''),
        ('''    # VOTRE CODE ICI
    return w + eta * np.einsum("tn,tnj->j", G * (A - P), X) / (X.shape[0] * X.shape[1])
''', '''    # À COMPLÉTER : renvoyer les nouveaux poids (moyenne sur les heures et les maisons)
    return ___
'''),
        ('''    # VOTRE CODE ICI
    return {k: NOMINAL[k] * np.exp(rng.uniform(-1, 1, n) * largeur * np.log(np.sqrt(FACTEUR[k]))) for k in NOMINAL}
''', '''    # À COMPLÉTER : un dictionnaire {paramètre: n valeurs}, avec NOMINAL et FACTEUR
    return ___
'''),
        ('''    # VOTRE CODE ICI : dJ/db, dJ/dc
    return 2 * s_xi**2 * (b * c - 1) * c + 2 * b * s_eps**2, 2 * s_xi**2 * (b * c - 1) * b
''', '''    # À COMPLÉTER : dJ/db, dJ/dc
    return ___, ___
'''),
        ('''    # VOTRE CODE ICI : estimation cumulée ou exponentielle (demi-vie 12 h)
    est = x.expanding().mean() if methode == "cumulée" else x.ewm(halflife=12).mean()
''', '''    # À COMPLÉTER : moyenne cumulée ou moyenne exponentielle de x (demi-vie 12 h), avec pandas
    est = ___
'''),
    ],
    "en": [
        ('''        # YOUR CODE HERE: heat balance of the air (loss, exchange with the mass, pump, internal gains), then of the mass
        dTa = (xi["UA"] * (t_ext - Ta) + H_AM * (Tm - Ta) + q_pump + Q_INT) / C_AIR
        dTm = H_AM * (Ta - Tm) / xi["Cm"]
''', '''        # TO DO: heat balance of the air (loss, exchange with the mass, pump, internal gains), then of the mass
        # Available: xi["UA"], xi["Cm"], H_AM, C_AIR, Q_INT, q_pump, t_ext
        dTa = ___
        dTm = ___
'''),
        ('''    # YOUR CODE HERE
    return w + eta * np.einsum("tn,tnj->j", G * (A - P), X) / (X.shape[0] * X.shape[1])
''', '''    # TO DO: return the new weights (average over hours and houses)
    return ___
'''),
        ('''    # YOUR CODE HERE
    return {k: NOMINAL[k] * np.exp(rng.uniform(-1, 1, n) * width * np.log(np.sqrt(FACTOR[k]))) for k in NOMINAL}
''', '''    # TO DO: a dictionary {parameter: n values}, using NOMINAL and FACTOR
    return ___
'''),
        ('''    # YOUR CODE HERE: dJ/db, dJ/dc
    return 2 * s_xi**2 * (b * c - 1) * c + 2 * b * s_eps**2, 2 * s_xi**2 * (b * c - 1) * b
''', '''    # TO DO: dJ/db, dJ/dc
    return ___, ___
'''),
        ('''    # YOUR CODE HERE: cumulative or exponential estimate (half-life 12 h)
    est = x.expanding().mean() if method == "cumulative" else x.ewm(halflife=12).mean()
''', '''    # TO DO: cumulative mean or exponential mean of x (half-life 12 h), using pandas
    est = ___
'''),
    ],
}

# Lines around the exercises: the checks, and the lab's version while an exercise is empty
CHECKS = {
    "fr": [
        ('print("Test :", np.allclose(mon_step(20.0, 19.0, 0.5, -10.0, xi_test), step(20.0, 19.0, 0.5, -10.0, xi_test)))',
         'verifier("Exercice 1", lambda: all(np.allclose(m, s) for m, s in zip(mon_step(20.0, 19.0, 0.5, -10.0, xi_test), step(20.0, 19.0, 0.5, -10.0, xi_test))))'),
        ('w, hist = reinforce(meteo, iterations=100, eta=5.0, maj=ma_maj, verbose=False)',
         'r = np.random.default_rng(0); essai = (np.zeros(7), r.normal(size=(4, 3, 7)), r.integers(0, 2, (4, 3)), r.uniform(size=(4, 3)), r.normal(size=(4, 3)), 0.1)\n'
         'fait = verifier("Exercice 2", lambda: np.allclose(ma_maj(*essai), maj_reinforce(*essai)), secours=True)\n'
         'w, hist = reinforce(meteo, iterations=100, eta=5.0, maj=ma_maj if fait else None, verbose=False)'),
        ('xi_demo = mes_maisons(2000)',
         'graine = lambda: np.random.default_rng(0)\n'
         'fait = verifier("Exercice 3", lambda: all(np.allclose(mes_maisons(5, rng=graine())[k], tirer_maisons(5, rng=graine())[k]) for k in NOMINAL), secours=True)\n'
         'xi_demo = (mes_maisons if fait else tirer_maisons)(2000, rng=graine())'),
        ('print("Gradient au col (0, 0) :", mon_grad_J(0.0, 0.0), "   Test :", np.allclose(mon_grad_J(.3, -.7), grad_J(.3, -.7)))',
         'fait = verifier("Exercice 4", lambda: np.allclose(mon_grad_J(.3, -.7), grad_J(.3, -.7)), secours=True)\n'
         'gradient = mon_grad_J if fait else grad_J\n'
         'print("Gradient au col (0, 0) :", gradient(0.0, 0.0))'),
        ('gb, gc = mon_grad_J(b, c, s_xi)', 'gb, gc = gradient(b, c, s_xi)'),
        ('''print(pd.DataFrame({m: [demi_vie(t0, m) for t0 in (12, 24, 48, 96, 192)] for m in ("cumulée", "exponentielle")},
                   index=pd.Index([12, 24, 48, 96, 192], name="âge au saut (h)")))''',
         '''if verifier("Exercice 5", lambda: abs(demi_vie(96, "cumulée") - 96) <= 1 and abs(demi_vie(192, "exponentielle") - 12) <= 1) is not None:
    print(pd.DataFrame({m: [demi_vie(t0, m) for t0 in (12, 24, 48, 96, 192)] for m in ("cumulée", "exponentielle")},
                       index=pd.Index([12, 24, 48, 96, 192], name="âge au saut (h)")))'''),
    ],
    "en": [
        ('print("Test:", np.allclose(my_thermal_step(20.0, 19.0, 0.5, -10.0, xi_check), thermal_step(20.0, 19.0, 0.5, -10.0, xi_check)))',
         'check("Exercise 1", lambda: all(np.allclose(m, s) for m, s in zip(my_thermal_step(20.0, 19.0, 0.5, -10.0, xi_check), thermal_step(20.0, 19.0, 0.5, -10.0, xi_check))))'),
        ('w, hist = reinforce(weather, iterations=100, eta=5.0, update=my_update, verbose=False)',
         'r = np.random.default_rng(0); trial = (np.zeros(7), r.normal(size=(4, 3, 7)), r.integers(0, 2, (4, 3)), r.uniform(size=(4, 3)), r.normal(size=(4, 3)), 0.1)\n'
         'done = check("Exercise 2", lambda: np.allclose(my_update(*trial), reinforce_update(*trial)), fallback=True)\n'
         'w, hist = reinforce(weather, iterations=100, eta=5.0, update=my_update if done else None, verbose=False)'),
        ('xi_demo = my_houses(2000)',
         'seed = lambda: np.random.default_rng(0)\n'
         'done = check("Exercise 3", lambda: all(np.allclose(my_houses(5, rng=seed())[k], sample_houses(5, rng=seed())[k]) for k in NOMINAL), fallback=True)\n'
         'xi_demo = (my_houses if done else sample_houses)(2000, rng=seed())'),
        ('print("Gradient at the saddle (0, 0):", my_grad_J(0.0, 0.0), "   Test:", np.allclose(my_grad_J(.3, -.7), grad_J(.3, -.7)))',
         'done = check("Exercise 4", lambda: np.allclose(my_grad_J(.3, -.7), grad_J(.3, -.7)), fallback=True)\n'
         'gradient = my_grad_J if done else grad_J\n'
         'print("Gradient at the saddle (0, 0):", gradient(0.0, 0.0))'),
        ('gb, gc = my_grad_J(b, c, s_xi)', 'gb, gc = gradient(b, c, s_xi)'),
        ('''print(pd.DataFrame({m: [half_life(t0, m) for t0 in (12, 24, 48, 96, 192)] for m in ("cumulative", "exponential")},
                   index=pd.Index([12, 24, 48, 96, 192], name="age at the jump (h)")))''',
         '''if check("Exercise 5", lambda: abs(half_life(96, "cumulative") - 96) <= 1 and abs(half_life(192, "exponential") - 12) <= 1) is not None:
    print(pd.DataFrame({m: [half_life(t0, m) for t0 in (12, 24, 48, 96, 192)] for m in ("cumulative", "exponential")},
                       index=pd.Index([12, 24, 48, 96, 192], name="age at the jump (h)")))'''),
    ],
}

SETUP = {
    "fr": ('''# Fichiers compagnons : lab_utils.py, schemas.py, agents/, meteo_montreal.csv, tous dans lab_rl_prototype.zip
import os, sys, urllib.request, zipfile
if not os.path.exists("lab_utils.py"):
    paquet = "lab_rl_prototype.zip"
    if not os.path.exists(paquet):
        try:
            urllib.request.urlretrieve("''' + RAW + '''/fr/lab_rl_prototype.zip", paquet)
            print("Fichiers compagnons téléchargés depuis GitHub.")
        except OSError:                             # GitHub inaccessible : on demande le zip
            try:
                from google.colab import files      # n'existe que dans Colab
            except ImportError:
                raise FileNotFoundError("lab_utils.py introuvable : ouvrez ce notebook depuis le dossier décompressé.")
            print("Téléchargement impossible. Cliquez sur « Choisir des fichiers » et choisissez lab_rl_prototype.zip.")
            zips = [nom for nom in files.upload() if nom.endswith(".zip")]
            assert zips, "Ce n'était pas un .zip : téléversez lab_rl_prototype.zip."
            paquet = zips[0]
    zipfile.ZipFile(paquet).extractall(".")
''', '''# Fichiers compagnons : lab_utils.py, schemas.py, blanks.py, agents/, meteo_montreal.csv
import os, sys, urllib.request, zipfile
if not os.path.exists("blanks.py"):
    paquet = "lab_rl_prototype.zip"
    try:
        urllib.request.urlretrieve("''' + RAW + '''/fr/lab_rl_prototype.zip", paquet)
        print("Fichiers compagnons téléchargés depuis GitHub.")
    except OSError:                                 # GitHub inaccessible : on demande le zip
        try:
            from google.colab import files          # n'existe que dans Colab
        except ImportError:
            raise FileNotFoundError("blanks.py introuvable : ouvrez ce notebook depuis le dossier fr/ du lab.")
        print("Téléchargement impossible. Cliquez sur « Choisir des fichiers » et choisissez lab_rl_prototype.zip.")
        zips = [nom for nom in files.upload() if nom.endswith(".zip")]
        assert zips, "Ce n'était pas un .zip : téléversez lab_rl_prototype.zip."
        paquet = zips[0]
    archive = zipfile.ZipFile(paquet)
    archive.extractall(".", [f for f in archive.namelist() if not f.endswith(".ipynb")])
'''),
    "en": ('''# Companion files: lab_utils_en.py, schemas.py, agents/, montreal_weather.csv, all inside rl_lab_en.zip
import os, sys, urllib.request, zipfile
if not os.path.exists("lab_utils_en.py"):
    bundle = "rl_lab_en.zip"
    if not os.path.exists(bundle):
        try:
            urllib.request.urlretrieve("''' + RAW + '''/en/rl_lab_en.zip", bundle)
            print("Companion files downloaded from GitHub.")
        except OSError:                             # GitHub unreachable: ask for the zip
            try:
                from google.colab import files      # only exists in Colab
            except ImportError:
                raise FileNotFoundError("lab_utils_en.py not found: open this notebook from the unzipped rl_lab_en folder.")
            print("Download failed. Click 'Choose files' below and pick rl_lab_en.zip.")
            zips = [name for name in files.upload() if name.endswith(".zip")]
            assert zips, "That was not a .zip file: please upload rl_lab_en.zip."
            bundle = zips[0]
    zipfile.ZipFile(bundle).extractall(".")
''', '''# Companion files: lab_utils_en.py, schemas.py, blanks.py, agents/, montreal_weather.csv
import os, sys, urllib.request, zipfile
if not os.path.exists("blanks.py"):
    bundle = "rl_lab_en.zip"
    try:
        urllib.request.urlretrieve("''' + RAW + '''/en/rl_lab_en.zip", bundle)
        print("Companion files downloaded from GitHub.")
    except OSError:                                 # GitHub unreachable: ask for the zip
        try:
            from google.colab import files          # only exists in Colab
        except ImportError:
            raise FileNotFoundError("blanks.py not found: open this notebook from the en/ folder of the lab.")
        print("Download failed. Click 'Choose files' below and pick rl_lab_en.zip.")
        zips = [name for name in files.upload() if name.endswith(".zip")]
        assert zips, "That was not a .zip file: please upload rl_lab_en.zip."
        bundle = zips[0]
    archive = zipfile.ZipFile(bundle)
    archive.extractall(".", [f for f in archive.namelist() if not f.endswith(".ipynb")])
'''),
}

IMPORTS = {
    "fr": ("from schemas import schema_maison, schema_boucle, schema_agent\n",
           "from schemas import schema_maison, schema_boucle, schema_agent\nfrom blanks import ___, verifier\n"),
    "en": ("from schemas import house_diagram, loop_diagram, agent_diagram\n",
           "from schemas import house_diagram, loop_diagram, agent_diagram\nfrom blanks import ___, check\n"),
}

# Text that would give away an exercise or a question
TEXT = {
    "fr": [
        ("fr/prototype_lab_rl.ipynb", "fr/lab_rl_etudiant.ipynb"),
        ("**IFT3700/IFT6758 · Science des données · Labo RL · prototype v2, à tester et recadrer**",
         "**IFT3700/IFT6758 · Science des données · Labo RL · version étudiante (prototype)**"),
        ("> **Prototype.** Les corrections des cinq exercices sont déjà écrites. La météo",
         "> **Exercices.** Remplacez chaque `___` par votre code, puis exécutez la cellule. Tant qu'un exercice est vide, le notebook continue avec la version du labo.\n\n> **Prototype.** La météo"),
        ("(`expanding().mean()`) ", ""),
        ("(`ewm`, moyenne exponentielle)", "(moyenne exponentielle)"),
        ("\n\n> **Vers le contrôle général.** Avec la tête, la mémoire lit la maison dans l'historique, et la politique s'en sert : environ +3 % au-dessus de l'oracle, contre +10 % sans mémoire.", ""),
        ("\n\n> **Vers le contrôle général.** Entraîné sur des maisons qui changent, l'agent continue d'apprendre après son déploiement, avec des poids figés. C'est le méta-RL : la marche du haut.", ""),
    ],
    "en": [
        ("en/rl_lab_prototype_en.ipynb", "en/rl_lab_student_en.ipynb"),
        ("**IFT3700/IFT6758 · Data Science · RL lab · prototype v2, English version**",
         "**IFT3700/IFT6758 · Data Science · RL lab · student version (prototype)**"),
        ("> **Prototype.** The solutions to the five exercises are already written in. The weather",
         "> **Exercises.** Replace each `___` with your code, then run the cell. While an exercise is empty, the notebook keeps going with the lab's version.\n\n> **Prototype.** The weather"),
        ("(`expanding().mean()`) ", ""),
        ("(`ewm`, an exponentially weighted mean)", "(an exponentially weighted mean)"),
    ],
}

# English cells that state the answer to the question just above them
DROP_EN = ("> **Towards general control.** With the head, the memory reads the house",
           "> **Towards general control.** Trained on houses that change")

NOTEBOOKS = {"fr": ("fr/prototype_lab_rl.ipynb", "fr/lab_rl_etudiant.ipynb"),
             "en": ("en/rl_lab_prototype_en.ipynb", "en/rl_lab_student_en.ipynb")}


def swap(text, old, new, count=1):
    if text.count(old) != count:
        raise SystemExit(f"Instructor notebook changed, cannot find: {old[:70]!r}")
    return text.replace(old, new)


def build(lang):
    source, target = NOTEBOOKS[lang]
    nb = json.loads((HERE / source).read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if "<details" not in "".join(c["source"])
             and not "".join(c["source"]).startswith(DROP_EN)]
    code = "\n\x00\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    for old, new in EXERCISES[lang] + CHECKS[lang] + [SETUP[lang], IMPORTS[lang]]:
        code = swap(code, old, new)
    code_cells = iter(code.split("\n\x00\n"))
    text = "\n\x00\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    for old, new in TEXT[lang]:
        text = swap(text, old, new)
    text_cells = iter(text.split("\n\x00\n"))
    for cell in cells:
        cell["source"] = next(code_cells if cell["cell_type"] == "code" else text_cells).splitlines(keepends=True)
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    nb["cells"] = cells
    (HERE / target).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Built", target)


if __name__ == "__main__":
    (HERE / "en/blanks.py").write_bytes((HERE / "fr/blanks.py").read_bytes())
    for language in NOTEBOOKS:
        build(language)
