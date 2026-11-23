"""The house as a Gymnasium environment, the memoryless Stable-Baselines3 agents, and the evaluation of every agent.

- HouseGym: one house, standard Gymnasium interface (to understand the convention).
- HousesVec: N houses simulated at once, as an SB3 VecEnv (to train fast).
- make_agent: an SB3 PPO without memory. The memory agents live in gru_ppo_en.py.
- load_agent, evaluate, and the memory diagnostics (erase, swap, probe).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

from lab_utils_en import H, LEVELS, OBS_DIM, PRIV_DIM, Houses, normalize_xi, sample_houses
from gru_ppo_en import GRUAgent, rollout_gru

HERE = os.path.dirname(os.path.abspath(__file__))


class HouseGym(gym.Env):
    """One house, following the Gymnasium convention.

    reset() -> (observation, info); step(action) -> (observation, reward, terminated, truncated, info).
    An episode lasts H = 168 hours: it always ends by truncation (truncated=True), never by termination.
    """

    def __init__(self, weather, xi=None, width=1.0, drift=False, privileged=False, seed=0):
        self.houses = Houses(weather, 1, xi=xi, width=width, drift=drift, seed=seed)
        self.privileged = privileged
        dim = PRIV_DIM if privileged else OBS_DIM
        self.observation_space = spaces.Box(-50.0, 50.0, (dim,), np.float32)
        self.action_space = spaces.Discrete(len(LEVELS))

    def _obs(self):
        return (self.houses.obs_privileged() if self.privileged else self.houses.obs())[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.houses.rng = np.random.default_rng(seed)
        self.houses.reset()
        return self._obs(), {}

    def step(self, action):
        _, r, c, disc, kwh = self.houses.step(np.array([int(action)]))
        info = {"cost": float(c[0]), "discomfort": float(disc[0]), "kwh": float(kwh[0])}
        return self._obs(), float(r[0]), False, self.houses.t >= H, info


class HousesVec(VecEnv):
    """N houses as an SB3 VecEnv: all N advance one step in a single NumPy call.

    All episodes start and end together; at the end, the houses restart on their own (automatic reset).
    """

    def __init__(self, weather, n, xi=None, width=1.0, drift=False, privileged=False, seed=0):
        self.houses = Houses(weather, n, xi=xi, width=width, drift=drift, seed=seed)
        self.privileged = privileged
        dim = PRIV_DIM if privileged else OBS_DIM
        super().__init__(n, spaces.Box(-50.0, 50.0, (dim,), np.float32), spaces.Discrete(len(LEVELS)))

    def _obs(self):
        return self.houses.obs_privileged() if self.privileged else self.houses.obs()

    def reset(self):
        self.houses.reset()
        return self._obs()

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(-1).astype(int)

    def step_wait(self):
        _, r, c, disc, _ = self.houses.step(self._actions)
        obs, done = self._obs(), self.houses.t >= H
        infos = [{"cost": c[i], "discomfort": disc[i]} for i in range(self.num_envs)]
        if done:
            for i, info in enumerate(infos):
                info["terminal_observation"], info["TimeLimit.truncated"] = obs[i], True
            obs = self.reset()
        return obs, r.astype(np.float32), np.full(self.num_envs, done), infos

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


def make_agent(weather, n=256, xi=None, width=1.0, drift=False, privileged=False, seed=0):
    """An SB3 PPO without memory (an MLP) and its N training houses."""
    env = VecMonitor(HousesVec(weather, n, xi=xi, width=width, drift=drift, privileged=privileged, seed=seed))
    return PPO("MlpPolicy", env, n_steps=H, batch_size=H * 32, n_epochs=4, gamma=0.97, gae_lambda=0.95,
               learning_rate=3e-4, ent_coef=0.01, clip_range=0.2, max_grad_norm=0.5, seed=seed, verbose=0,
               device="cpu", policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh))


def train_sb3(agent, iterations, callback=None):
    """One iteration = one week played by the N houses, then one PPO update."""
    agent.learn(total_timesteps=iterations * agent.n_envs * H, callback=callback, progress_bar=False)
    return agent


def rollout_sb3(agent, env, privileged=False, xi=None, starts=None):
    """Plays one full episode with a memoryless SB3 agent, without exploration."""
    o = env.reset(xi=xi, starts=starts)
    C, DISC, TA = [], [], []
    for t in range(H):
        ob = env.obs_privileged() if privileged else o
        a, _ = agent.predict(ob, deterministic=True)
        o, _, c, disc, _ = env.step(np.asarray(a).reshape(-1))
        C.append(c); DISC.append(disc); TA.append(env.Ta.copy())
    return dict(cost=np.stack(C, 1), discomfort=np.stack(DISC, 1), Ta=np.stack(TA, 1), xi=env.xi0)


# ---------------------------------------------------------------- evaluation and diagnostics

def load_agent(name, folder=os.path.join(HERE, "agents")):
    """Loads a pre-trained agent: an SB3 PPO (.zip) or one of our GRU agents (.pt). Returns (agent, history)."""
    hist = json.load(open(os.path.join(folder, f"{name}.json")))
    if os.path.exists(os.path.join(folder, f"{name}.zip")):
        return PPO.load(os.path.join(folder, f"{name}.zip"), device="cpu"), hist
    agent = GRUAgent(prediction_head=hist.get("head", hist.get("tete", False)))
    agent.load_state_dict(torch.load(os.path.join(folder, f"{name}.pt"), weights_only=True))
    agent.eval()
    return agent, hist


def evaluate(policy, weather, xi, starts, privileged=False, drift=False, seed=123, **kw):
    """Plays one week on fixed houses and weather windows, without exploration.

    `policy`: a GRU agent, an SB3 agent, or a function env -> actions (the thermostats). Returns a dict of (N, H)
    arrays: cost, discomfort, Ta, and m (the memory) for a GRU agent.
    """
    env = Houses(weather, len(starts), drift=drift, seed=seed)
    if isinstance(policy, GRUAgent):
        return rollout_gru(policy, env, greedy=True, xi=xi, starts=starts, **kw)
    if hasattr(policy, "predict"):
        return rollout_sb3(policy, env, privileged, xi=xi, starts=starts)
    env.reset(xi=xi, starts=starts)
    C, DISC, TA = [], [], []
    for t in range(H):
        _, _, c, disc, _ = env.step(policy(env))
        C.append(c); DISC.append(disc); TA.append(env.Ta.copy())
    return dict(cost=np.stack(C, 1), discomfort=np.stack(DISC, 1), Ta=np.stack(TA, 1), xi=env.xi0)


def test_bench(weather, n=128, width=1.0, seed=7):
    """n test houses and n weather weeks, always the same, to compare agents."""
    rng = np.random.default_rng(seed)
    T = Houses(weather, 1).T
    return sample_houses(n, width, rng), rng.integers(0, len(T) - H - 4, n)


def cost_per_day(d):
    return d["cost"].sum(1) / (d["cost"].shape[1] / 24)


def _extra_cost(agent, weather, xi, starts, t, window, **kw):
    d0 = evaluate(agent, weather, xi, starts)
    d1 = evaluate(agent, weather, xi, starts, **kw)
    c0, c1 = d0["cost"][:, t:t + window].sum(1), d1["cost"][:, t:t + window].sum(1)
    return 100 * (c1.sum() - c0.sum()) / c0.sum()


def erase_effect(agent, weather, xi, starts, t=72, window=24):
    """Extra cost (%) over the `window` hours after wiping the memory at hour t."""
    return _extra_cost(agent, weather, xi, starts, t, window, erase_at=t)


def swap_effect(agent, weather, xi, starts, t=72, window=24):
    """Extra cost (%) when the memories of two houses are swapped at hour t."""
    return _extra_cost(agent, weather, xi, starts, t, window, swap_at=t)


def probe(d, hours=(0, 6, 12, 24, 48, 72, 120, 167)):
    """R² of a Ridge regression reading ξ (normalized) from the memory, hour by hour (half the houses to fit)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    y = normalize_xi(d["xi"])
    n = len(y) // 2
    return {t: float(r2_score(y[n:], Ridge(alpha=1.0).fit(d["m"][:n, t], y[:n]).predict(d["m"][n:, t]))) for t in hours}
