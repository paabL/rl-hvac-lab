"""Précalcule les agents du lab. Usage : python entrainer_agents.py NOM [ITERATIONS] [--reprendre]

Agents sans mémoire : PPO de Stable-Baselines3. Agents à mémoire : notre PPO à GRU (gru_ppo.py).
Une itération = une semaine jouée par 256 maisons, puis une mise à jour de PPO.
"""
import json
import os
import sys

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

from lab_utils import H, charger_meteo, maison_nominale, tirer_maisons
from rl_sb3 import creer_agent
from gru_ppo import AgentGRU, entrainer_gru
from lab_utils import Maisons

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


class Journal(BaseCallback):
    """Coût moyen par jour à chaque itération, et une sauvegarde toutes les `tous` itérations."""

    def __init__(self, nom, avant, tous=100):
        super().__init__()
        self.nom, self.hist, self.tous, self.somme = nom, list(avant), tous, 0.0

    def _on_step(self):
        self.somme += sum(i["cout"] for i in self.locals["infos"])
        return True

    def _on_rollout_end(self):
        self.hist.append(self.somme / self.training_env.num_envs / 7)
        self.somme = 0.0
        k = len(self.hist)
        if k % 20 == 0:
            print(f"{self.nom} itération {k:5d}  coût moyen {np.mean(self.hist[-20:]):6.2f} $/jour", flush=True)
        if k % self.tous == 0:
            sauver(self.model, self.nom, self.hist)


def sauver(agent, nom, hist):
    os.makedirs(os.path.join(HERE, "agents"), exist_ok=True)
    agent.save(os.path.join(HERE, "agents", f"{nom}.zip"))
    json.dump(dict(nom=nom, iterations=len(hist), cout_par_jour=[float(c) for c in hist]),
              open(os.path.join(HERE, "agents", f"{nom}.json"), "w"))


if __name__ == "__main__":
    nom = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 400
    xi, memoire, tete, priv, derive = CONFIGS[nom]
    meteo = charger_meteo()
    dossier = os.path.join(HERE, "agents")
    os.makedirs(dossier, exist_ok=True)
    reprendre = "--reprendre" in sys.argv
    if memoire:
        agent = AgentGRU(tete_prediction=tete)
        avant = []
        if reprendre and os.path.exists(os.path.join(dossier, f"{nom}.pt")):
            agent.load_state_dict(torch.load(os.path.join(dossier, f"{nom}.pt"), weights_only=True))
            avant = json.load(open(os.path.join(dossier, f"{nom}.json")))["cout_par_jour"]
        torch.manual_seed(len(avant))
        env = Maisons(meteo, N, xi=xi, derive=derive, seed=1 + len(avant))
        hist = avant + entrainer_gru(agent, env, iterations=iters)
        torch.save(agent.state_dict(), os.path.join(dossier, f"{nom}.pt"))
        json.dump(dict(nom=nom, iterations=len(hist), tete=tete, cout_par_jour=[float(c) for c in hist]),
                  open(os.path.join(dossier, f"{nom}.json"), "w"))
    else:
        agent = creer_agent(meteo, n=N, xi=xi, derive=derive, privilegie=priv, seed=1)
        avant = []
        if reprendre and os.path.exists(os.path.join(dossier, f"{nom}.zip")):
            agent = PPO.load(os.path.join(dossier, f"{nom}.zip"), env=agent.get_env(), device="cpu")
            avant = json.load(open(os.path.join(dossier, f"{nom}.json")))["cout_par_jour"]
        journal = Journal(nom, avant)
        agent.learn(total_timesteps=iters * N * H, callback=journal, reset_num_timesteps=not avant)
        sauver(agent, nom, journal.hist)
