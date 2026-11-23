"""RL lab for home heating: two-temperature house simulator, agents and diagnostics (prototype).

English version of lab_utils.py (same code and numbers, English names).
Everything runs on CPU. The simulator is NumPy, vectorized over N houses.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- weather

def synthetic_weather(n_januaries=20, seed=0):
    """Hourly outdoor temperatures (°C) for synthetic Montreal Januaries, one row per hour.

    Daily mean is AR(1) around -10 °C (std 6, day-to-day correlation 0.8),
    4 °C daily cycle (coldest at 6 am, warmest at 3 pm), 0.5 °C hourly noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(n_januaries):
        m = rng.normal(0, 6)
        for d in range(31):
            m = 0.8 * m + math.sqrt(1 - 0.8**2) * rng.normal(0, 6)
            for h in range(24):
                daily = -2.0 * math.cos(2 * math.pi * (h - 3) / 24)
                rows.append((y, d, h, -10 + m + daily + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["year", "day", "hour", "t_ext"])


def download_open_meteo(years=range(2015, 2025)):
    """Hourly Montreal Januaries from the Open-Meteo archive (CC BY 4.0 data, cite it)."""
    frames = []
    for i, a in enumerate(years):
        url = ("https://archive-api.open-meteo.com/v1/archive?latitude=45.50&longitude=-73.57"
               f"&start_date={a}-01-01&end_date={a}-01-31&hourly=temperature_2m&timezone=America/Toronto")
        data = json.load(urllib.request.urlopen(url, timeout=20))["hourly"]
        t = pd.to_datetime(data["time"])
        frames.append(pd.DataFrame({"year": i, "day": t.day - 1, "hour": t.hour, "t_ext": data["temperature_2m"]}))
    return pd.concat(frames, ignore_index=True)


def load_weather(path=os.path.join(HERE, "montreal_weather.csv")):
    if os.path.exists(path):
        return pd.read_csv(path)
    try:
        df = download_open_meteo()
    except Exception:
        df = synthetic_weather()
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------- houses

NOMINAL = dict(UA=150.0, Cm=3.0e7, mu=1.6)          # W/K, J/K, heat-pump oversizing factor
FACTOR = dict(UA=3.0, Cm=3.0, mu=3.0)               # box width: from nominal/sqrt(f) to nominal*sqrt(f)
C_AIR, H_AM, Q_INT = 8.0e6, 1000.0, 400.0           # air heat capacity (J/K), air-mass coupling (W/K), internal gains (W)
T_DESIGN, T_SETPOINT = -25.0, 21.0                  # °C
COMFORT_BAND = (20.0, 23.0)                         # °C
PRICE_OFFPEAK, PRICE_PEAK = 0.07, 0.50              # $/kWh
LAMBDA = 1.0                                        # $ per K·h of discomfort
LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])      # actions: fraction of heat-pump power
H = 168                                             # one episode = 7 days at a 1-hour step


def sample_houses(n, width=1.0, rng=None):
    """ξ = nominal × exp(U(-w, w)) in log space, w = width × log(sqrt(factor)). Returns a dict of (n,) arrays."""
    rng = rng or np.random.default_rng()
    return {k: NOMINAL[k] * np.exp(rng.uniform(-1, 1, n) * width * math.log(math.sqrt(FACTOR[k]))) for k in NOMINAL}


def nominal_house(n):
    return {k: np.full(n, NOMINAL[k]) for k in NOMINAL}


def normalize_xi(xi):
    """log(ξ/nominal) / log(sqrt(factor)): each parameter in [-1, 1] over the box."""
    return np.stack([np.log(xi[k] / NOMINAL[k]) / math.log(math.sqrt(FACTOR[k])) for k in NOMINAL], -1)


def is_peak(hour_of_week):
    h, day = hour_of_week % 24, (hour_of_week // 24) % 7
    return ((day < 5) & (((h >= 6) & (h < 9)) | ((h >= 16) & (h < 20)))).astype(np.float32)


def cop(t_ext):
    """Coefficient of performance: units of heat delivered per unit of electricity. Worse when it is colder."""
    return np.clip(2.8 + 0.05 * t_ext, 1.5, 4.0)


def thermal_step(Ta, Tm, u, t_ext, xi, substeps=6):
    """One hour of the two-temperature (2R2C) model (explicit Euler, 6 substeps of 10 min).

    Ta, Tm: air and mass temperatures (°C); u: fraction of heat-pump power in [0, 1].
    Returns Ta, Tm one hour later and the electricity used (kWh).
    """
    q_pump = u * xi["mu"] * xi["UA"] * (T_SETPOINT - T_DESIGN)   # thermal W
    dt = 3600.0 / substeps
    for _ in range(substeps):
        dTa = (xi["UA"] * (t_ext - Ta) + H_AM * (Tm - Ta) + q_pump + Q_INT) / C_AIR
        dTm = H_AM * (Ta - Tm) / xi["Cm"]
        Ta, Tm = Ta + dt * dTa, Tm + dt * dTm
    kwh = q_pump / cop(t_ext) / 1000.0
    return Ta, Tm, kwh


def hourly_cost(kwh, price, Ta):
    """Hourly cost in $: bill + λ × discomfort (K·h outside the 20-23 °C band)."""
    discomfort = np.maximum(COMFORT_BAND[0] - Ta, 0) + np.maximum(Ta - COMFORT_BAND[1], 0)
    return price * kwh + LAMBDA * discomfort, discomfort


class Houses:
    """N houses simulated together. Public observation, and privileged observation (oracle only)."""

    def __init__(self, weather, n, xi=None, width=1.0, drift=False, seed=0, noise=0.1):
        self.rng = np.random.default_rng(seed)
        self.T = weather.pivot_table(index=["year", "day", "hour"], values="t_ext").values.ravel()
        self.n, self.width, self.drift, self.noise = n, width, drift, noise
        self.xi_fixed = xi

    def reset(self, xi=None, starts=None):
        n = self.n
        self.xi0 = xi if xi is not None else (self.xi_fixed() if callable(self.xi_fixed) else
                                              self.xi_fixed if self.xi_fixed is not None else sample_houses(n, self.width, self.rng))
        self.xi = {k: v.copy() for k, v in self.xi0.items()}
        self.start = starts if starts is not None else self.rng.integers(0, len(self.T) - H - 4, n)
        self.h0 = self.rng.integers(0, 7, n) * 24 if starts is None else np.zeros(n, int)
        self.t = 0
        self.Ta = self.rng.uniform(20.0, 22.0, n)
        self.Tm = self.Ta - self.rng.uniform(0.0, 1.0, n)
        self.u_prev = np.zeros(n)
        # drift: at a random hour, the heat pump loses 50% of its power or the insulation degrades (×1.8)
        self.t_jump = self.rng.integers(24, H - 24, n) if self.drift else np.full(n, H + 1)
        self.jump_type = self.rng.integers(0, 2, n)
        return self.obs()

    def obs(self):
        t_ext = self.T[self.start + self.t]
        hw = self.h0 + self.t
        upcoming = np.stack([is_peak(hw + k) for k in (1, 2, 3)], -1)
        o = np.column_stack([
            (self.Ta + self.rng.normal(0, self.noise, self.n) - 21.0) / 2.0,
            t_ext / 10.0,
            is_peak(hw), upcoming,
            np.sin(2 * np.pi * (hw % 24) / 24), np.cos(2 * np.pi * (hw % 24) / 24),
            self.u_prev,
        ]).astype(np.float32)
        return o

    def obs_privileged(self):
        return np.column_stack([self.obs(), normalize_xi(self.xi), (self.Tm - 21.0) / 2.0]).astype(np.float32)

    def apply_drift(self):
        jump = self.t == self.t_jump
        if jump.any():
            pump = jump & (self.jump_type == 0)
            leak = jump & (self.jump_type == 1)
            self.xi["mu"] = np.where(pump, self.xi["mu"] * 0.5, self.xi["mu"])
            self.xi["UA"] = np.where(leak, self.xi["UA"] * 1.8, self.xi["UA"])

    def step(self, a):
        """a: action indices (n,). Returns obs, reward, cost, discomfort, kWh."""
        self.apply_drift()
        u = LEVELS[a]
        t_ext = self.T[self.start + self.t]
        price = np.where(is_peak(self.h0 + self.t) > 0, PRICE_PEAK, PRICE_OFFPEAK)
        self.Ta, self.Tm, kwh = thermal_step(self.Ta, self.Tm, u, t_ext, self.xi)
        c, disc = hourly_cost(kwh, price, self.Ta)
        self.u_prev = u
        self.t += 1
        return self.obs(), -c, c, disc, kwh


OBS_DIM = 9
PRIV_DIM = OBS_DIM + 4


# ---------------------------------------------------------------- baseline policies

def _level(u):
    return np.abs(np.clip(u, 0, 1)[:, None] - LEVELS[None]).argmin(1)


def thermostat(env, target=None):
    """Proportional thermostat at 21 °C, no anticipation: u = 0.4 + 0.4 × (21 - Ta)."""
    target = T_SETPOINT if target is None else target
    return _level(0.4 + 0.4 * (target - env.Ta))


def preheating_thermostat(env):
    """Same, but aims for 22.5 °C in the 3 hours before a peak and drops to 20 °C during it."""
    hw = env.h0 + env.t
    target = np.where(is_peak(hw + 1) + is_peak(hw + 2) + is_peak(hw + 3) > 0, 22.5, T_SETPOINT)
    target = np.where(is_peak(hw) > 0, 20.0, target)
    return thermostat(env, target)


# ---------------------------------------------------------------- networks

class Agent(nn.Module):
    """Actor-critic. memory=False: memoryless MLP. memory=True: GRU h = f(W[x, h_prev] + w0).

    prediction_head=True: a head predicts next hour's temperature change from the memory
    and the action taken; it trains the memory directly, without going through the actor.
    """

    def __init__(self, obs_dim=OBS_DIM, memory=True, prediction_head=False, hidden=64):
        super().__init__()
        self.memory, self.head = memory, prediction_head
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True) if memory else None
        self.pi = nn.Sequential(nn.Linear(hidden + obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, len(LEVELS)))
        self.v = nn.Sequential(nn.Linear(hidden + obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.pred = nn.Sequential(nn.Linear(hidden + obs_dim + len(LEVELS), hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, obs, h=None):
        """obs: (B, T, obs_dim). Returns logits, values, memories (B, T, hidden), last h."""
        z = self.enc(obs)
        if self.memory:
            m, h = self.gru(z, h)
        else:
            m = z
        x = torch.cat([m, obs], -1)
        return self.pi(x), self.v(x).squeeze(-1), m, h

    def predict(self, m, obs, a):
        return self.pred(torch.cat([m, obs, nn.functional.one_hot(a, len(LEVELS)).float()], -1)).squeeze(-1)


# ---------------------------------------------------------------- PPO (full episodes, memory reset at the start)

def rollout(agent, env, privileged=False, greedy=False, xi=None, starts=None, erase_at=None, swap_at=None):
    """Plays one full episode on the N houses. Returns a dict of (N, H, ...) arrays."""
    o = env.reset(xi=xi, starts=starts)
    O, A, LP, V, R, C, DISC, KWH, TA, M = [], [], [], [], [], [], [], [], [], []
    h = None
    with torch.no_grad():
        for t in range(H):
            if erase_at is not None and t == erase_at and h is not None:
                h = torch.zeros_like(h)
            if swap_at is not None and t == swap_at and h is not None:
                h = h[:, swap_pairs(env.n)]
            ob = env.obs_privileged() if privileged else o
            logits, v, m, h = agent(torch.from_numpy(ob)[:, None], h)
            dist = torch.distributions.Categorical(logits=logits[:, 0])
            a = logits[:, 0].argmax(-1) if greedy else dist.sample()
            O.append(ob); A.append(a.numpy()); LP.append(dist.log_prob(a).numpy()); V.append(v[:, 0].numpy())
            M.append(m[:, 0].numpy())
            o, r, c, disc, kwh = env.step(a.numpy())
            R.append(r); C.append(c); DISC.append(disc); KWH.append(kwh); TA.append(env.Ta.copy())
    st = lambda x: np.stack(x, 1)
    return dict(obs=st(O), a=st(A), logp=st(LP), v=st(V), r=st(R).astype(np.float32), cost=st(C), discomfort=st(DISC),
                kwh=st(KWH), Ta=st(TA), m=st(M), xi=env.xi0, start=env.start)


def swap_pairs(n):
    """Swaps memories in pairs: 0↔1, 2↔3, ..."""
    p = np.arange(n)
    p[0::2], p[1::2] = np.arange(1, n, 2), np.arange(0, n, 2)
    return torch.from_numpy(p)


def gae(r, v, gamma=0.97, lam=0.95):
    adv = np.zeros_like(r)
    last = 0.0
    for t in reversed(range(r.shape[1])):
        nv = v[:, t + 1] if t + 1 < r.shape[1] else 0.0
        delta = r[:, t] + gamma * nv - v[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + v


def train(agent, env, iterations=200, privileged=False, lr=3e-4, epochs=4, mb=64, clip=0.2, ent=0.01,
          beta_pred=1.0, verbose=True):
    """Recurrent PPO on full episodes. Returns the history of the mean cost per day."""
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    hist = []
    t0 = time.time()
    for it in range(iterations):
        d = rollout(agent, env, privileged)
        adv, ret = gae(d["r"], d["v"])
        obs, a, lp0 = map(torch.from_numpy, (d["obs"], d["a"], d["logp"]))
        adv, ret = torch.from_numpy(adv), torch.from_numpy(ret)
        # head target: next hour's (normalized) change in air temperature
        Ta_now = 21.0 + 2.0 * d["obs"][:, :, 0]
        target = torch.from_numpy(((d["Ta"] - Ta_now) / 1.0).astype(np.float32))
        n = obs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                logits, v, m, _ = agent(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(a[idx])
                ad = adv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                ratio = torch.exp(lp - lp0[idx])
                l_pi = -torch.min(ratio * ad, torch.clamp(ratio, 1 - clip, 1 + clip) * ad).mean()
                l_v = ((v - ret[idx]) ** 2).mean()
                loss = l_pi + 0.5 * l_v - ent * dist.entropy().mean()
                if agent.head:
                    p = agent.predict(m, obs[idx], a[idx])
                    loss = loss + beta_pred * ((p - target[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
        c_day = d["cost"].sum(1).mean() / 7
        hist.append(c_day)
        if verbose and (it % 20 == 0 or it == iterations - 1):
            print(f"iteration {it:4d}  mean cost {c_day:6.2f} $/day  discomfort {d['discomfort'].sum(1).mean()/7:5.2f} K·h/day  "
                  f"({time.time()-t0:4.0f} s)")
    return hist


# ---------------------------------------------------------------- evaluation

def evaluate(policy, weather, xi, starts, privileged=False, drift=False, seed=123, **kw):
    """Runs a policy greedily on fixed houses and fixed weather windows. Returns a dict of (N, H) arrays."""
    env = Houses(weather, len(starts), drift=drift, seed=seed)
    if isinstance(policy, nn.Module):
        return rollout(policy, env, privileged, greedy=True, xi=xi, starts=starts, **kw)
    env.reset(xi=xi, starts=starts)
    C, DISC, TA = [], [], []
    for t in range(H):
        _, _, c, disc, _ = env.step(policy(env))
        C.append(c); DISC.append(disc); TA.append(env.Ta.copy())
    return dict(cost=np.stack(C, 1), discomfort=np.stack(DISC, 1), Ta=np.stack(TA, 1))


def test_bench(weather, n=128, width=1.0, seed=7):
    """Fixed test houses and weather windows, so every agent is compared on the same ground."""
    rng = np.random.default_rng(seed)
    T = Houses(weather, 1).T
    return sample_houses(n, width, rng), rng.integers(0, len(T) - H - 4, n)


# ---------------------------------------------------------------- toy model of the saddle

def J(b, c, s_xi=1.0, s_eps=1.0):
    """Toy cost: memory writes m = c ξ + ε, the action reads a = b m, the ideal action is ξ."""
    return s_xi**2 * (b * c - 1) ** 2 + b**2 * s_eps**2


def grad_J(b, c, s_xi=1.0, s_eps=1.0):
    return 2 * s_xi**2 * (b * c - 1) * c + 2 * b * s_eps**2, 2 * s_xi**2 * (b * c - 1) * b


def descent(b, c, eta=0.05, steps=400, **kw):
    traj = [(b, c)]
    for _ in range(steps):
        gb, gc = grad_J(b, c, **kw)
        b, c = b - eta * gb, c - eta * gc
        traj.append((b, c))
    return np.array(traj)


# ---------------------------------------------------------------- memory diagnostics

def load_agent(name, folder=os.path.join(HERE, "agents")):
    from sklearn.linear_model import Ridge  # noqa: F401  (checks the dependency early)
    cfg = json.load(open(os.path.join(folder, f"{name}.json")))
    sd = torch.load(os.path.join(folder, f"{name}.pt"), weights_only=True)
    memory = any(k.startswith("gru.") for k in sd)
    obs_dim = sd["enc.0.weight"].shape[1]
    head = cfg.get("prediction_head", "in_context" in name or "memorize" in name or name == "drift")
    ag = Agent(obs_dim=obs_dim, memory=memory, prediction_head=head)
    ag.load_state_dict(sd)
    ag.eval()
    return ag, cfg


def cost_per_day(d):
    return d["cost"].sum(1) / (d["cost"].shape[1] / 24)


def erase_effect(agent, weather, xi, starts, t=72, window=24, **kw):
    """Mean extra cost (%) over the `window` hours after the memory is wiped at hour t."""
    d0 = evaluate(agent, weather, xi, starts, **kw)
    d1 = evaluate(agent, weather, xi, starts, erase_at=t, **kw)
    c0, c1 = d0["cost"][:, t:t + window].sum(1), d1["cost"][:, t:t + window].sum(1)
    return 100 * (c1.sum() - c0.sum()) / c0.sum()


def swap_effect(agent, weather, xi, starts, t=72, window=24, **kw):
    """Mean extra cost (%) when the memories of two houses are swapped at hour t."""
    d0 = evaluate(agent, weather, xi, starts, **kw)
    d1 = evaluate(agent, weather, xi, starts, swap_at=t, **kw)
    c0, c1 = d0["cost"][:, t:t + window].sum(1), d1["cost"][:, t:t + window].sum(1)
    return 100 * (c1.sum() - c0.sum()) / c0.sum()


def probe(d, hours=(0, 6, 12, 24, 48, 72, 120, 167)):
    """R² of a Ridge regression that reads ξ (normalized) from the memory, hour by hour.

    Half the houses to fit, half to test. Returns {hour: R² averaged over the 3 parameters}.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    y = normalize_xi(d["xi"])
    n = len(y) // 2
    out = {}
    for t in hours:
        X = d["m"][:, t]
        r = Ridge(alpha=1.0).fit(X[:n], y[:n])
        out[t] = float(r2_score(y[n:], r.predict(X[n:])))
    return out


# ---------------------------------------------------------------- REINFORCE: logistic regression weighted by the return

def features(env):
    """x = [1, Ta - 21, T_ext/10, peak now, peak in 1-3 h, sin, cos of the hour]."""
    o = env.obs()
    return np.column_stack([np.ones(env.n), o[:, 0] * 2.0, o[:, 1], o[:, 2], o[:, 3:6].max(1), o[:, 6], o[:, 7]])


def reinforce(weather, iterations=60, eta=0.05, n=512, seed=0, update=None, verbose=True):
    """Bernoulli policy π(heat | x) = σ(wᵀx) trained by REINFORCE on the nominal house.

    update(w, X, A, P, G, eta) applies the update (the exercise); by default, the solution.
    """
    rng = np.random.default_rng(seed)
    env = Houses(weather, n, xi=lambda: nominal_house(n), seed=seed)
    w = np.zeros(7)
    hist = []
    for it in range(iterations):
        env.reset()
        X, A, P, R = [], [], [], []
        for t in range(H):
            x = features(env)
            p = 1 / (1 + np.exp(-x @ w))
            a = (rng.random(n) < p).astype(float)
            _, r, _, _, _ = env.step(np.where(a > 0, 4, 0))
            X.append(x); A.append(a); P.append(p); R.append(r)
        X, A, P, R = map(np.array, (X, A, P, R))                 # (H, n, ...)
        G = np.cumsum(R[::-1], 0)[::-1]                           # return from each hour on
        G = (G - G.mean(1, keepdims=True)) / (G.std() + 1e-8)     # baseline: the mean over houses at the same hour
        w = (update or reinforce_update)(w, X, A, P, G, eta)
        hist.append(-R.sum(0).mean() / 7)
        if verbose and (it % 10 == 0 or it == iterations - 1):
            print(f"iteration {it:3d}  mean cost {hist[-1]:6.2f} $/day")
    return w, hist


def reinforce_update(w, X, A, P, G, eta):
    """w_j ← w_j + η Σ G (a - p) x_j: the logistic-regression gradient step, weighted by the return."""
    return w + eta * np.einsum("tn,tnj->j", G * (A - P), X) / (X.shape[0] * X.shape[1])
