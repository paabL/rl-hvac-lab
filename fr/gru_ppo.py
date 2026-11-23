"""Notre PPO à mémoire : un GRU, un actor, un critic et une tête de prédiction (self-predictive RL).

Pourquoi pas une bibliothèque ici : la tête de prédiction n'existe pas dans RecurrentPPO de sb3-contrib, et nos
essais avec RecurrentPPO restaient moins bons que l'agent sans mémoire. Ce fichier n'est pas à lire en classe :
on appelle AgentGRU, entrainer_gru et collecter_gru.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from lab_utils import H, NIVEAUX, OBS_DIM


class AgentGRU(nn.Module):
    """Actor-critic à mémoire GRU, h_t = f(W[x_t, h_{t-1}] + w0). Les têtes lisent [h_t, x_t].

    tete_prediction=True : une tête prédit la variation de température de l'heure suivante à partir de la mémoire
    et de l'action jouée ; sa perte s'ajoute à celle de PPO et écrit la maison dans la mémoire.
    """

    def __init__(self, obs_dim=OBS_DIM, tete_prediction=False, hidden=64):
        super().__init__()
        self.memoire, self.tete = True, tete_prediction
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi = nn.Sequential(nn.Linear(hidden + obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, len(NIVEAUX)))
        self.v = nn.Sequential(nn.Linear(hidden + obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.pred = nn.Sequential(nn.Linear(hidden + obs_dim + len(NIVEAUX), hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, obs, h=None):
        """obs : (B, T, obs_dim). Retourne logits, valeurs, mémoires (B, T, hidden), dernier h."""
        m, h = self.gru(self.enc(obs), h)
        x = torch.cat([m, obs], -1)
        return self.pi(x), self.v(x).squeeze(-1), m, h

    def predire(self, m, obs, a):
        return self.pred(torch.cat([m, obs, nn.functional.one_hot(a, len(NIVEAUX)).float()], -1)).squeeze(-1)


def _paires(n):
    """Échange les mémoires deux à deux : 0↔1, 2↔3, ..."""
    p = np.arange(n)
    p[0::2], p[1::2] = np.arange(1, n, 2), np.arange(0, n, 2)
    return torch.from_numpy(p)


def collecter_gru(agent, env, greedy=False, xi=None, debuts=None, effacer_a=None, echanger_a=None):
    """Joue un épisode complet sur les N maisons. Retourne un dict de tableaux (N, H, ...)."""
    o = env.reset(xi=xi, debuts=debuts)
    O, A, LP, V, R, C, INC, TA, M = [], [], [], [], [], [], [], [], []
    h = None
    with torch.no_grad():
        for t in range(H):
            if h is not None and t == effacer_a:
                h = torch.zeros_like(h)
            if h is not None and t == echanger_a:
                h = h[:, _paires(env.n)]
            logits, v, m, h = agent(torch.from_numpy(o)[:, None], h)
            dist = torch.distributions.Categorical(logits=logits[:, 0])
            a = logits[:, 0].argmax(-1) if greedy else dist.sample()
            O.append(o); A.append(a.numpy()); LP.append(dist.log_prob(a).numpy()); V.append(v[:, 0].numpy())
            M.append(m[:, 0].numpy())
            o, r, c, inc, _ = env.pas(a.numpy())
            R.append(r); C.append(c); INC.append(inc); TA.append(env.Ta.copy())
    st = lambda x: np.stack(x, 1)
    return dict(obs=st(O), a=st(A), logp=st(LP), v=st(V), r=st(R).astype(np.float32), cout=st(C), inconfort=st(INC),
                Ta=st(TA), m=st(M), xi=env.xi0)


def _gae(r, v, gamma=0.97, lam=0.95):
    adv = np.zeros_like(r)
    last = 0.0
    for t in reversed(range(r.shape[1])):
        nv = v[:, t + 1] if t + 1 < r.shape[1] else 0.0
        delta = r[:, t] + gamma * nv - v[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + v


def entrainer_gru(agent, env, iterations=200, lr=3e-4, epochs=4, mb=64, clip=0.2, ent=0.01, beta_pred=1.0, verbose=True):
    """PPO récurrent sur épisodes complets (une itération = une semaine jouée par les N maisons, puis une mise à jour).

    Retourne l'historique du coût moyen par jour.
    """
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    hist, t0 = [], time.time()
    for it in range(iterations):
        d = collecter_gru(agent, env)
        adv, ret = _gae(d["r"], d["v"])
        obs, a, lp0 = map(torch.from_numpy, (d["obs"], d["a"], d["logp"]))
        adv, ret = torch.from_numpy(adv), torch.from_numpy(ret)
        Ta_avant = 21.0 + 2.0 * d["obs"][:, :, 0]                       # cible de la tête : variation de l'heure suivante
        cible = torch.from_numpy((d["Ta"] - Ta_avant).astype(np.float32))
        n = obs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                logits, v, m, _ = agent(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ad = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
                ratio = torch.exp(dist.log_prob(a[idx]) - lp0[idx])
                perte = -torch.min(ratio * ad, torch.clamp(ratio, 1 - clip, 1 + clip) * ad).mean()
                perte = perte + 0.5 * ((v - ret[idx]) ** 2).mean() - ent * dist.entropy().mean()
                if agent.tete:
                    perte = perte + beta_pred * ((agent.predire(m, obs[idx], a[idx]) - cible[idx]) ** 2).mean()
                opt.zero_grad()
                perte.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
        hist.append(d["cout"].sum(1).mean() / 7)
        if verbose and (it % 20 == 0 or it == iterations - 1):
            print(f"itération {it:4d}  coût moyen {hist[-1]:6.2f} $/jour  ({time.time() - t0:4.0f} s)")
    return hist
