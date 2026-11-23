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
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
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
    X2 = X.reshape(-1, X.shape[-1])             # un exemple par ligne
    poids = (G * (A - P)).reshape(-1)           # retour × (action - probabilité)
    gradient = X2.T @ poids / len(poids)        # comme la régression logistique : Xᵀ (y - ŷ) / n
''', '''    # À COMPLÉTER : la régression logistique du cours 2, chaque exemple pondéré par son retour
    X2 = X.reshape(-1, X.shape[-1])             # un exemple par ligne
    poids = ___                                 # retour × (action - probabilité), un nombre par exemple
    gradient = ___                              # comme la régression logistique : Xᵀ (y - ŷ) / n
'''),
        ('''    # VOTRE CODE ICI
    maisons = {}
    for k in NOMINAL:
        w = largeur * np.log(np.sqrt(FACTEUR[k]))
        maisons[k] = NOMINAL[k] * np.exp(rng.uniform(-w, w, n))
''', '''    # À COMPLÉTER : pour chaque paramètre k, n valeurs avec NOMINAL[k] et FACTEUR[k]
    maisons = {}
    for k in NOMINAL:
        w = ___
        maisons[k] = ___
'''),
        ('''    # VOTRE CODE ICI : dJ/db, dJ/dc
    erreur = b * c - 1
    dJ_db = 2 * s_xi**2 * erreur * c + 2 * b * s_eps**2
    dJ_dc = 2 * s_xi**2 * erreur * b
''', '''    # À COMPLÉTER : dJ/db, dJ/dc
    erreur = b * c - 1
    dJ_db = ___
    dJ_dc = ___
'''),
        ('''    # VOTRE CODE ICI : estimation cumulée ou exponentielle (demi-vie 12 h)
    if methode == "cumulée":
        estimation = x.expanding().mean()
    else:
        estimation = x.ewm(halflife=12).mean()
''', '''    # À COMPLÉTER : moyenne cumulée ou moyenne exponentielle de x (demi-vie 12 h), avec pandas
    if methode == "cumulée":
        estimation = ___
    else:
        estimation = ___
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
    X2 = X.reshape(-1, X.shape[-1])             # one example per row
    weights = (G * (A - P)).reshape(-1)         # return × (action - probability)
    gradient = X2.T @ weights / len(weights)    # like logistic regression: Xᵀ (y - ŷ) / n
''', '''    # TO DO: the logistic regression of Lecture 2, each example weighted by its return
    X2 = X.reshape(-1, X.shape[-1])             # one example per row
    weights = ___                               # return × (action - probability), one number per example
    gradient = ___                              # like logistic regression: Xᵀ (y - ŷ) / n
'''),
        ('''    # YOUR CODE HERE
    houses = {}
    for k in NOMINAL:
        w = width * np.log(np.sqrt(FACTOR[k]))
        houses[k] = NOMINAL[k] * np.exp(rng.uniform(-w, w, n))
''', '''    # TO DO: for each parameter k, n values using NOMINAL[k] and FACTOR[k]
    houses = {}
    for k in NOMINAL:
        w = ___
        houses[k] = ___
'''),
        ('''    # YOUR CODE HERE: dJ/db, dJ/dc
    error = b * c - 1
    dJ_db = 2 * s_xi**2 * error * c + 2 * b * s_eps**2
    dJ_dc = 2 * s_xi**2 * error * b
''', '''    # TO DO: dJ/db, dJ/dc
    error = b * c - 1
    dJ_db = ___
    dJ_dc = ___
'''),
        ('''    # YOUR CODE HERE: cumulative or exponential estimate (half-life 12 h)
    if method == "cumulative":
        estimate = x.expanding().mean()
    else:
        estimate = x.ewm(halflife=12).mean()
''', '''    # TO DO: cumulative mean or exponential mean of x (half-life 12 h), using pandas
    if method == "cumulative":
        estimate = ___
    else:
        estimate = ___
'''),
    ],
}

# Lines around the exercises: the checks, and the lab's version while an exercise is empty
CHECKS = {
    "fr": [
        ('print("Test :", np.allclose(mon_step(20.0, 19.0, 0.5, -10.0, xi_essai), step(20.0, 19.0, 0.5, -10.0, xi_essai)))',
         '''
def test_exercice_1():
    return np.allclose(mon_step(20.0, 19.0, 0.5, -10.0, xi_essai), step(20.0, 19.0, 0.5, -10.0, xi_essai))

fait = verifier("Exercice 1", test_exercice_1)'''),
        ('w, hist = reinforce(meteo, iterations=100, eta=5.0, maj=ma_maj, verbose=False)',
         '''def test_exercice_2():
    r = np.random.default_rng(0)
    X = r.normal(size=(4, 3, 7))
    A = r.integers(0, 2, (4, 3))
    P = r.uniform(size=(4, 3))
    G = r.normal(size=(4, 3))
    return np.allclose(ma_maj(np.zeros(7), X, A, P, G, 0.1), maj_reinforce(np.zeros(7), X, A, P, G, 0.1))

fait = verifier("Exercice 2", test_exercice_2, secours=True)
if fait:
    w, hist = reinforce(meteo, iterations=100, eta=5.0, maj=ma_maj, verbose=False)
else:
    w, hist = reinforce(meteo, iterations=100, eta=5.0, verbose=False)        # la version du labo'''),
        ('xi_demo = mes_maisons(2000)',
         '''def test_exercice_3():
    les_miennes = mes_maisons(5, rng=np.random.default_rng(0))
    celles_du_labo = tirer_maisons(5, rng=np.random.default_rng(0))
    for k in NOMINAL:
        if not np.allclose(les_miennes[k], celles_du_labo[k]):
            return False
    return True

fait = verifier("Exercice 3", test_exercice_3, secours=True)
if fait:
    xi_demo = mes_maisons(2000, rng=np.random.default_rng(0))
else:
    xi_demo = tirer_maisons(2000, rng=np.random.default_rng(0))             # la version du labo'''),
        ('print("Gradient au col (0, 0) :", mon_grad_J(0.0, 0.0), "   Test :", np.allclose(mon_grad_J(.3, -.7), grad_J(.3, -.7)))',
         '''def test_exercice_4():
    return np.allclose(mon_grad_J(.3, -.7), grad_J(.3, -.7))

fait = verifier("Exercice 4", test_exercice_4, secours=True)
if fait:
    gradient = mon_grad_J
else:
    gradient = grad_J                                                       # la version du labo
print("Gradient au col (0, 0) :", gradient(0.0, 0.0))'''),
        ('gb, gc = mon_grad_J(b, c, s_xi)', 'gb, gc = gradient(b, c, s_xi)'),
        ('''for t0 in [12, 24, 48, 96, 192]:
    print(f"saut à l'heure {t0:3d} : demi-vie cumulée {demi_vie(t0, 'cumulée'):3d} h, exponentielle {demi_vie(t0, 'exponentielle'):3d} h")''',
         '''def test_exercice_5():
    return demi_vie(96, "cumulée") == 95 and demi_vie(192, "exponentielle") == 11

fait = verifier("Exercice 5", test_exercice_5)
if fait is not None:
    for t0 in [12, 24, 48, 96, 192]:
        print(f"saut à l'heure {t0:3d} : demi-vie cumulée {demi_vie(t0, 'cumulée'):3d} h, exponentielle {demi_vie(t0, 'exponentielle'):3d} h")'''),
    ],
    "en": [
        ('print("Test:", np.allclose(my_thermal_step(20.0, 19.0, 0.5, -10.0, xi_check), thermal_step(20.0, 19.0, 0.5, -10.0, xi_check)))',
         '''
def test_exercise_1():
    return np.allclose(my_thermal_step(20.0, 19.0, 0.5, -10.0, xi_check), thermal_step(20.0, 19.0, 0.5, -10.0, xi_check))

done = check("Exercise 1", test_exercise_1)'''),
        ('w, hist = reinforce(weather, iterations=100, eta=5.0, update=my_update, verbose=False)',
         '''def test_exercise_2():
    r = np.random.default_rng(0)
    X = r.normal(size=(4, 3, 7))
    A = r.integers(0, 2, (4, 3))
    P = r.uniform(size=(4, 3))
    G = r.normal(size=(4, 3))
    return np.allclose(my_update(np.zeros(7), X, A, P, G, 0.1), reinforce_update(np.zeros(7), X, A, P, G, 0.1))

done = check("Exercise 2", test_exercise_2, fallback=True)
if done:
    w, hist = reinforce(weather, iterations=100, eta=5.0, update=my_update, verbose=False)
else:
    w, hist = reinforce(weather, iterations=100, eta=5.0, verbose=False)     # the lab's version'''),
        ('xi_demo = my_houses(2000)',
         '''def test_exercise_3():
    mine = my_houses(5, rng=np.random.default_rng(0))
    lab = sample_houses(5, rng=np.random.default_rng(0))
    for k in NOMINAL:
        if not np.allclose(mine[k], lab[k]):
            return False
    return True

done = check("Exercise 3", test_exercise_3, fallback=True)
if done:
    xi_demo = my_houses(2000, rng=np.random.default_rng(0))
else:
    xi_demo = sample_houses(2000, rng=np.random.default_rng(0))             # the lab's version'''),
        ('print("Gradient at the saddle (0, 0):", my_grad_J(0.0, 0.0), "   Test:", np.allclose(my_grad_J(.3, -.7), grad_J(.3, -.7)))',
         '''def test_exercise_4():
    return np.allclose(my_grad_J(.3, -.7), grad_J(.3, -.7))

done = check("Exercise 4", test_exercise_4, fallback=True)
if done:
    gradient = my_grad_J
else:
    gradient = grad_J                                                       # the lab's version
print("Gradient at the saddle (0, 0):", gradient(0.0, 0.0))'''),
        ('gb, gc = my_grad_J(b, c, s_xi)', 'gb, gc = gradient(b, c, s_xi)'),
        ('''for t0 in [12, 24, 48, 96, 192]:
    print(f"jump at hour {t0:3d}: cumulative half-life {half_life(t0, 'cumulative'):3d} h, exponential {half_life(t0, 'exponential'):3d} h")''',
         '''def test_exercise_5():
    return half_life(96, "cumulative") == 95 and half_life(192, "exponential") == 11

done = check("Exercise 5", test_exercise_5)
if done is not None:
    for t0 in [12, 24, 48, 96, 192]:
        print(f"jump at hour {t0:3d}: cumulative half-life {half_life(t0, 'cumulative'):3d} h, exponential {half_life(t0, 'exponential'):3d} h")'''),
    ],
}

SETUP = {
    "fr": ('''# Fichiers compagnons : lab_utils.py, rl_sb3.py, gru_ppo.py, schemas.py, images/, agents/, meteo_montreal.csv, tous dans lab_rl_prototype.zip
import os, sys, subprocess, urllib.request, zipfile
try:
    import stable_baselines3                         # Stable-Baselines3 (PPO), absent de Colab par défaut
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "stable-baselines3"], check=True)
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
''', '''# Fichiers compagnons : lab_utils.py, rl_sb3.py, gru_ppo.py, schemas.py, blanks.py, images/, agents/, meteo_montreal.csv
import os, sys, subprocess, urllib.request, zipfile
try:
    import stable_baselines3                         # Stable-Baselines3 (PPO), absent de Colab par défaut
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "stable-baselines3"], check=True)
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
    for nom in archive.namelist():
        if not nom.endswith(".ipynb"):              # on garde ce notebook tel quel
            archive.extract(nom, ".")
'''),
    "en": ('''# Companion files: lab_utils_en.py, rl_sb3_en.py, gru_ppo_en.py, schemas.py, images/, agents/, montreal_weather.csv, all inside rl_lab_en.zip
import os, sys, subprocess, urllib.request, zipfile
try:
    import stable_baselines3                         # Stable-Baselines3 (PPO), not in Colab by default
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "stable-baselines3"], check=True)
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
''', '''# Companion files: lab_utils_en.py, rl_sb3_en.py, gru_ppo_en.py, schemas.py, blanks.py, images/, agents/, montreal_weather.csv
import os, sys, subprocess, urllib.request, zipfile
try:
    import stable_baselines3                         # Stable-Baselines3 (PPO), not in Colab by default
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "stable-baselines3"], check=True)
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
    for name in archive.namelist():
        if not name.endswith(".ipynb"):             # keep this notebook as it is
            archive.extract(name, ".")
'''),
}

IMPORTS = {
    "fr": ("from schemas import schema_maison, schema_boucle, schema_agent, schema_prechauffage, schema_flotte\n",
           "from schemas import schema_maison, schema_boucle, schema_agent, schema_prechauffage, schema_flotte\nfrom blanks import ___, verifier\n"),
    "en": ("from schemas import house_diagram, loop_diagram, agent_diagram, preheating_diagram, fleet_diagram\n",
           "from schemas import house_diagram, loop_diagram, agent_diagram, preheating_diagram, fleet_diagram\nfrom blanks import ___, check\n"),
}

# Text that would give away an exercise or a question, and the student's own setup
TEXT = {
    "fr": [
        ("blob/main/fr/prototype_lab_rl.ipynb", "blob/main/fr/lab_rl_etudiant.ipynb"),
        ("**IFT3700/IFT6758 · Science des données · Labo RL · prototype v3**",
         "**IFT3700/IFT6758 · Science des données · Labo RL · version étudiante (prototype v3)**"),
        ("> **Prototype.** Les corrections des cinq exercices sont déjà écrites. La météo",
         "> **Exercices.** Remplacez chaque `___` par votre code, puis exécutez la cellule. Tant qu'un exercice est vide, le notebook continue avec la version du labo.\n\n> **Prototype.** La météo"),
        ("| `schemas.py` | `schema_...` | les schémas du notebook | une figure |",
         "| `schemas.py` | `schema_...` | les schémas du notebook | une figure |\n| `blanks.py` | `___`, `verifier` | les blancs des exercices et leur vérification | |"),
        ("(`expanding().mean()`) ", ""),
        ("(`ewm`, moyenne exponentielle)", "(moyenne exponentielle)"),
        ("\n\n> **Vers le contrôle général.** Avec la tête, la mémoire lit la maison dans l'historique et la politique s'en sert : environ +3 % au-dessus de l'oracle, contre +6 % sans tête et +11 % sans mémoire.", ""),
        ("\n\n> **Vers le contrôle général.** Entraîné sur des maisons qui changent, l'agent continue d'apprendre après son déploiement, avec des poids figés. C'est le meta-RL : la marche du haut.", ""),
    ],
    "en": [
        ("blob/main/en/rl_lab_prototype_en.ipynb", "blob/main/en/rl_lab_student_en.ipynb"),
        ("**IFT3700/IFT6758 · Data Science · RL lab · prototype v3, English version**",
         "**IFT3700/IFT6758 · Data Science · RL lab · student version (prototype v3)**"),
        ("> **Prototype.** The solutions to the five exercises are already written in. The weather",
         "> **Exercises.** Replace each `___` with your code, then run the cell. While an exercise is empty, the notebook keeps going with the lab's version.\n\n> **Prototype.** The weather"),
        ("| `schemas.py` | `..._diagram` | the notebook's diagrams | a figure |",
         "| `schemas.py` | `..._diagram` | the notebook's diagrams | a figure |\n| `blanks.py` | `___`, `check` | the exercise blanks and their checks | |"),
        ("(`expanding().mean()`) ", ""),
        ("(`ewm`, an exponentially weighted mean)", "(an exponentially weighted mean)"),
        ("\n\n> **Towards general control.** With the head, the memory reads the house from its history and the policy uses it: about +3% above the oracle, against +6% without the head and +11% without memory.", ""),
        ("> **Towards general control.** Trained on houses that change, the agent keeps learning after deployment, with frozen weights. That is meta-RL: the top rung.", ""),
    ],
}

NOTEBOOKS = {"fr": ("fr/prototype_lab_rl.ipynb", "fr/lab_rl_etudiant.ipynb"),
             "en": ("en/rl_lab_prototype_en.ipynb", "en/rl_lab_student_en.ipynb")}


def swap(text, old, new, count=1):
    if text.count(old) != count:
        raise SystemExit(f"Instructor notebook changed, cannot find: {old[:70]!r}")
    return text.replace(old, new)


def build(lang):
    source, target = NOTEBOOKS[lang]
    nb = json.loads((HERE / source).read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if "<details" not in "".join(c["source"])]
    code = "\n\x00\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    for old, new in EXERCISES[lang] + CHECKS[lang] + [SETUP[lang], IMPORTS[lang]]:
        code = swap(code, old, new)
    code_cells = iter(code.split("\n\x00\n"))
    text = "\n\x00\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    for old, new in TEXT[lang]:
        text = swap(text, old, new)
    text_cells = iter(text.split("\n\x00\n"))
    kept = []
    for cell in cells:
        cell["source"] = next(code_cells if cell["cell_type"] == "code" else text_cells).splitlines(keepends=True)
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
        if "".join(cell["source"]).strip():
            kept.append(cell)
    nb["cells"] = kept
    (HERE / target).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Built", target)


if __name__ == "__main__":
    (HERE / "en/blanks.py").write_bytes((HERE / "fr/blanks.py").read_bytes())
    for language in NOTEBOOKS:
        build(language)
