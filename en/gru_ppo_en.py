"""Our memory PPO: a GRU, an actor, a critic and a prediction head (self-predictive RL).

Why not a library here: the prediction head does not exist in sb3-contrib's RecurrentPPO, and our tests with
RecurrentPPO stayed worse than the memoryless agent. This file is not meant to be read in class: we call
GRUAgent, train_gru and rollout_gru.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from lab_utils_en import H, LEVELS, OBS_DIM


class GRUAgent(nn.Module):
    """Actor-critic with a GRU memory, h_t = f(W[x_t, h_{t-1}] + w0). The heads read [h_t, x_t].

    prediction_head=True: a head predicts next hour's temperature change from the memory and the action taken;
    its loss is added to PPO's and writes the house into the memory.
    """

    def __init__(self, obs_dim=OBS_DIM, prediction_head=False, hidden=64):
        super().__init__()
        self.memory, self.head = True, prediction_head
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi = nn.Sequential(nn.Linear(hidden + obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, len(LEVELS)))
        self.v = nn.Sequential(nn.Linear(hidden + obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.pred = nn.Sequential(nn.Linear(hidden + obs_dim + len(LEVELS), hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, obs, h=None):
        """obs: (B, T, obs_dim). Returns logits, values, memories (B, T, hidden), last h."""
        m, h = self.gru(self.enc(obs), h)
        x = torch.cat([m, obs], -1)
        return self.pi(x), self.v(x).squeeze(-1), m, h

    def predict_next(self, m, obs, a):
        return self.pred(torch.cat([m, obs, nn.functional.one_hot(a, len(LEVELS)).float()], -1)).squeeze(-1)


def _pairs(n):
    """Swaps memories two by two: 0↔1, 2↔3, ..."""
    p = np.arange(n)
    p[0::2], p[1::2] = np.arange(1, n, 2), np.arange(0, n, 2)
    return torch.from_numpy(p)


def rollout_gru(agent, env, greedy=False, xi=None, starts=None, erase_at=None, swap_at=None):
    """Plays one full episode on the N houses. Returns a dict of (N, H, ...) arrays."""
    o = env.reset(xi=xi, starts=starts)
    O, A, LP, V, R, C, DISC, TA, M = [], [], [], [], [], [], [], [], []
    h = None
    with torch.no_grad():
        for t in range(H):
            if h is not None and t == erase_at:
                h = torch.zeros_like(h)
            if h is not None and t == swap_at:
                h = h[:, _pairs(env.n)]
            logits, v, m, h = agent(torch.from_numpy(o)[:, None], h)
            dist = torch.distributions.Categorical(logits=logits[:, 0])
            a = logits[:, 0].argmax(-1) if greedy else dist.sample()
            O.append(o); A.append(a.numpy()); LP.append(dist.log_prob(a).numpy()); V.append(v[:, 0].numpy())
            M.append(m[:, 0].numpy())
            o, r, c, disc, _ = env.step(a.numpy())
            R.append(r); C.append(c); DISC.append(disc); TA.append(env.Ta.copy())
    st = lambda x: np.stack(x, 1)
    return dict(obs=st(O), a=st(A), logp=st(LP), v=st(V), r=st(R).astype(np.float32), cost=st(C), discomfort=st(DISC),
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


def train_gru(agent, env, iterations=200, lr=3e-4, epochs=4, mb=64, clip=0.2, ent=0.01, beta_pred=1.0, verbose=True):
    """Recurrent PPO on full episodes (one iteration = one week played by the N houses, then one update).

    Returns the history of the mean cost per day.
    """
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    hist, t0 = [], time.time()
    for it in range(iterations):
        d = rollout_gru(agent, env)
        adv, ret = _gae(d["r"], d["v"])
        obs, a, lp0 = map(torch.from_numpy, (d["obs"], d["a"], d["logp"]))
        adv, ret = torch.from_numpy(adv), torch.from_numpy(ret)
        Ta_before = 21.0 + 2.0 * d["obs"][:, :, 0]                      # head target: next hour's change
        target = torch.from_numpy((d["Ta"] - Ta_before).astype(np.float32))
        n = obs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                logits, v, m, _ = agent(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ad = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
                ratio = torch.exp(dist.log_prob(a[idx]) - lp0[idx])
                loss = -torch.min(ratio * ad, torch.clamp(ratio, 1 - clip, 1 + clip) * ad).mean()
                loss = loss + 0.5 * ((v - ret[idx]) ** 2).mean() - ent * dist.entropy().mean()
                if agent.head:
                    loss = loss + beta_pred * ((agent.predict_next(m, obs[idx], a[idx]) - target[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
        hist.append(d["cost"].sum(1).mean() / 7)
        if verbose and (it % 20 == 0 or it == iterations - 1):
            print(f"iteration {it:4d}  mean cost {hist[-1]:6.2f} $/day  ({time.time() - t0:4.0f} s)")
    return hist
