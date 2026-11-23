"""La maison à la convention Gymnasium, les agents sans mémoire de Stable-Baselines3, et l'évaluation de tous les agents.

- MaisonGym : une maison, interface Gymnasium standard (pour comprendre la convention).
- MaisonsVec : N maisons simulées d'un coup, au format VecEnv de SB3 (pour entraîner vite).
- creer_agent : un PPO de SB3 sans mémoire. Les agents à mémoire sont dans gru_ppo.py.
- charger_agent, evaluer, et les diagnostics de mémoire (effacement, déménagement, sonde).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

import json
import os

from lab_utils import H, NIVEAUX, OBS_DIM, PRIV_DIM, Maisons, tirer_maisons, xi_normalise
from gru_ppo import AgentGRU, collecter_gru

HERE = os.path.dirname(os.path.abspath(__file__))


class MaisonGym(gym.Env):
    """Une maison, à la convention Gymnasium.

    reset() -> (observation, info) ; step(action) -> (observation, reward, terminated, truncated, info).
    Un épisode dure H = 168 heures : il finit par troncature (truncated=True), jamais par terminaison.
    """

    def __init__(self, meteo, xi=None, largeur=1.0, derive=False, privilegie=False, seed=0):
        self.maisons = Maisons(meteo, 1, xi=xi, largeur=largeur, derive=derive, seed=seed)
        self.privilegie = privilegie
        dim = PRIV_DIM if privilegie else OBS_DIM
        self.observation_space = spaces.Box(-50.0, 50.0, (dim,), np.float32)
        self.action_space = spaces.Discrete(len(NIVEAUX))

    def _obs(self):
        return (self.maisons.obs_privilegiee() if self.privilegie else self.maisons.obs())[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.maisons.rng = np.random.default_rng(seed)
        self.maisons.reset()
        return self._obs(), {}

    def step(self, action):
        _, r, c, inc, kwh = self.maisons.pas(np.array([int(action)]))
        info = {"cout": float(c[0]), "inconfort": float(inc[0]), "kwh": float(kwh[0])}
        return self._obs(), float(r[0]), False, self.maisons.t >= H, info


class MaisonsVec(VecEnv):
    """N maisons au format VecEnv de SB3 : les N avancent d'un pas en un seul appel NumPy.

    Tous les épisodes commencent et finissent ensemble ; à la fin, les maisons repartent seules (reset automatique).
    """

    def __init__(self, meteo, n, xi=None, largeur=1.0, derive=False, privilegie=False, seed=0):
        self.maisons = Maisons(meteo, n, xi=xi, largeur=largeur, derive=derive, seed=seed)
        self.privilegie = privilegie
        dim = PRIV_DIM if privilegie else OBS_DIM
        super().__init__(n, spaces.Box(-50.0, 50.0, (dim,), np.float32), spaces.Discrete(len(NIVEAUX)))

    def _obs(self):
        return self.maisons.obs_privilegiee() if self.privilegie else self.maisons.obs()

    def reset(self):
        self.maisons.reset()
        return self._obs()

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(-1).astype(int)

    def step_wait(self):
        _, r, c, inc, _ = self.maisons.pas(self._actions)
        obs, fini = self._obs(), self.maisons.t >= H
        infos = [{"cout": c[i], "inconfort": inc[i]} for i in range(self.num_envs)]
        if fini:
            for i, info in enumerate(infos):
                info["terminal_observation"], info["TimeLimit.truncated"] = obs[i], True
            obs = self.reset()
        return obs, r.astype(np.float32), np.full(self.num_envs, fini), infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


def creer_agent(meteo, n=256, xi=None, largeur=1.0, derive=False, privilegie=False, seed=0):
    """Un PPO de SB3 sans mémoire (un MLP) et ses N maisons d'entraînement."""
    env = VecMonitor(MaisonsVec(meteo, n, xi=xi, largeur=largeur, derive=derive, privilegie=privilegie, seed=seed))
    return PPO("MlpPolicy", env, n_steps=H, batch_size=H * 32, n_epochs=4, gamma=0.97, gae_lambda=0.95,
               learning_rate=3e-4, ent_coef=0.01, clip_range=0.2, max_grad_norm=0.5, seed=seed, verbose=0,
               device="cpu", policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh))


def entrainer_sb3(agent, iterations, callback=None):
    """Une itération = une semaine jouée par les N maisons, puis une mise à jour de PPO."""
    agent.learn(total_timesteps=iterations * agent.n_envs * H, callback=callback, progress_bar=False)
    return agent


def collecter_sb3(agent, env, privilegie=False, xi=None, debuts=None):
    """Joue un épisode complet avec un agent SB3 sans mémoire, sans exploration."""
    o = env.reset(xi=xi, debuts=debuts)
    C, INC, TA = [], [], []
    for t in range(H):
        ob = env.obs_privilegiee() if privilegie else o
        a, _ = agent.predict(ob, deterministic=True)
        o, _, c, inc, _ = env.pas(np.asarray(a).reshape(-1))
        C.append(c); INC.append(inc); TA.append(env.Ta.copy())
    return dict(cout=np.stack(C, 1), inconfort=np.stack(INC, 1), Ta=np.stack(TA, 1), xi=env.xi0)


# ---------------------------------------------------------------- évaluation et diagnostics

def charger_agent(nom, dossier=os.path.join(HERE, "agents")):
    """Charge un agent précalculé : un PPO de SB3 (.zip) ou un de nos agents GRU (.pt). Renvoie (agent, historique)."""
    hist = json.load(open(os.path.join(dossier, f"{nom}.json")))
    if os.path.exists(os.path.join(dossier, f"{nom}.zip")):
        return PPO.load(os.path.join(dossier, f"{nom}.zip"), device="cpu"), hist
    agent = AgentGRU(tete_prediction=hist.get("tete", False))
    agent.load_state_dict(torch.load(os.path.join(dossier, f"{nom}.pt"), weights_only=True))
    agent.eval()
    return agent, hist


def evaluer(politique, meteo, xi, debuts, privilegie=False, derive=False, seed=123, **kw):
    """Joue une semaine sur des maisons et des fenêtres météo fixées, sans exploration.

    `politique` : un agent GRU, un agent SB3, ou une fonction env -> actions (les thermostats). Retourne un dict de
    tableaux (N, H) : cout, inconfort, Ta, et m (la mémoire) pour un agent GRU.
    """
    env = Maisons(meteo, len(debuts), derive=derive, seed=seed)
    if isinstance(politique, AgentGRU):
        return collecter_gru(politique, env, greedy=True, xi=xi, debuts=debuts, **kw)
    if hasattr(politique, "predict"):
        return collecter_sb3(politique, env, privilegie, xi=xi, debuts=debuts)
    env.reset(xi=xi, debuts=debuts)
    C, INC, TA = [], [], []
    for t in range(H):
        _, _, c, inc, _ = env.pas(politique(env))
        C.append(c); INC.append(inc); TA.append(env.Ta.copy())
    return dict(cout=np.stack(C, 1), inconfort=np.stack(INC, 1), Ta=np.stack(TA, 1), xi=env.xi0)


def banc_de_test(meteo, n=128, largeur=1.0, seed=7):
    """n maisons de test et n semaines de météo, toujours les mêmes pour comparer les agents."""
    rng = np.random.default_rng(seed)
    T = Maisons(meteo, 1).T
    return tirer_maisons(n, largeur, rng), rng.integers(0, len(T) - H - 4, n)


def cout_par_jour(d):
    return d["cout"].sum(1) / (d["cout"].shape[1] / 24)


def _surcout(agent, meteo, xi, debuts, t, fenetre, **kw):
    d0 = evaluer(agent, meteo, xi, debuts)
    d1 = evaluer(agent, meteo, xi, debuts, **kw)
    c0, c1 = d0["cout"][:, t:t + fenetre].sum(1), d1["cout"][:, t:t + fenetre].sum(1)
    return 100 * (c1.sum() - c0.sum()) / c0.sum()


def effet_effacement(agent, meteo, xi, debuts, t=72, fenetre=24):
    """Surcoût (%) sur les `fenetre` heures qui suivent l'effacement de la mémoire à l'heure t."""
    return _surcout(agent, meteo, xi, debuts, t, fenetre, effacer_a=t)


def effet_demenagement(agent, meteo, xi, debuts, t=72, fenetre=24):
    """Surcoût (%) quand on échange les mémoires de deux maisons à l'heure t."""
    return _surcout(agent, meteo, xi, debuts, t, fenetre, echanger_a=t)


def sonde(d, heures=(0, 6, 12, 24, 48, 72, 120, 167)):
    """R² d'une régression Ridge qui lit ξ (normalisé) dans la mémoire, heure par heure (moitié des maisons pour ajuster)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    y = xi_normalise(d["xi"])
    n = len(y) // 2
    return {t: float(r2_score(y[n:], Ridge(alpha=1.0).fit(d["m"][:n, t], y[:n]).predict(d["m"][n:, t]))) for t in heures}
