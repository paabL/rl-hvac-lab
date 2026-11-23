"""Précalcule les agents du lab. Usage : python entrainer_agents.py NOM [ITERATIONS]"""
import json
import os
import sys

import numpy as np
import torch

from lab_utils import Agent, Maisons, PRIV_DIM, charger_meteo, entrainer, maison_nominale, tirer_maisons

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
N = 256


def maisons_fixes(k, seed=11):
    xi = tirer_maisons(k, 1.0, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    return lambda: {key: v[rng.integers(0, k, N)] for key, v in xi.items()}


CONFIGS = {
    # nom : (fabrique de maisons, mémoire, tête de prédiction, privilégié, dérive)
    "classique": (lambda: maison_nominale(N), True, False, False, False),
    "oracle": (None, False, False, True, False),
    "robuste": (None, False, False, False, False),
    "col": (None, True, False, False, False),
    "contexte": (None, True, True, False, False),
    "memoriser_4": (maisons_fixes(4), True, True, False, False),
    "memoriser_16": (maisons_fixes(16), True, True, False, False),
    "derive": (None, True, True, False, True),
    "oracle_derive": (None, False, False, True, True),
}

if __name__ == "__main__":
    nom = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    xi, memoire, tete, priv, derive = CONFIGS[nom]
    torch.manual_seed(0)
    env = Maisons(charger_meteo(), N, xi=xi, derive=derive, seed=1)
    agent = Agent(obs_dim=PRIV_DIM if priv else 9, memoire=memoire, tete_prediction=tete)
    chemin = os.path.join(HERE, "agents", f"{nom}.pt")
    avant = []
    if "--reprendre" in sys.argv and os.path.exists(chemin):
        agent.load_state_dict(torch.load(chemin, weights_only=True))
        avant = json.load(open(chemin.replace(".pt", ".json")))["cout_par_jour"]
        torch.manual_seed(len(avant))
        env = Maisons(charger_meteo(), N, xi=xi, derive=derive, seed=1 + len(avant))
    hist = avant + entrainer(agent, env, iterations=iters, privilegie=priv)
    os.makedirs(os.path.join(HERE, "agents"), exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(HERE, "agents", f"{nom}.pt"))
    json.dump(dict(nom=nom, iterations=len(hist), tete=tete, cout_par_jour=[float(c) for c in hist]),
              open(os.path.join(HERE, "agents", f"{nom}.json"), "w"))
