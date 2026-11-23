"""Pre-trains the lab's agents. Usage: python train_agents_en.py NAME [ITERATIONS] [--resume]

Memoryless agents: Stable-Baselines3 PPO. Memory agents: our GRU PPO (gru_ppo_en.py).

One iteration = one week played by 256 houses, then one PPO update. Not needed to teach: the agents are provided.
"""
import json
import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from lab_utils_en import H, Houses, load_weather, nominal_house, sample_houses
from rl_sb3_en import make_agent
from gru_ppo_en import GRUAgent, train_gru

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
N = 256


def fixed_houses(k, seed=11):
    xi = sample_houses(k, 1.0, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    return lambda: {key: v[rng.integers(0, k, N)] for key, v in xi.items()}


CONFIGS = {
    # name: (house factory, memory, prediction head, privileged, drift)
    "classic": (lambda: nominal_house(N), True, False, False, False),
    "oracle": (None, False, False, True, False),
    "robust": (None, False, False, False, False),
    "saddle": (None, True, False, False, False),
    "in_context": (None, True, True, False, False),
    "memorize_4": (fixed_houses(4), True, True, False, False),
    "memorize_16": (fixed_houses(16), True, True, False, False),
    "drift": (None, True, True, False, True),
    "oracle_drift": (None, False, False, True, True),
}


class Log(BaseCallback):
    """Mean cost per day at each iteration, and a checkpoint every `every` iterations."""

    def __init__(self, name, before, every=100):
        super().__init__()
        self.name, self.hist, self.every, self.total = name, list(before), every, 0.0

    def _on_step(self):
        self.total += sum(i["cost"] for i in self.locals["infos"])
        return True

    def _on_rollout_end(self):
        self.hist.append(self.total / self.training_env.num_envs / 7)
        self.total = 0.0
        if len(self.hist) % 20 == 0:
            print(f"{self.name} iteration {len(self.hist):5d}  mean cost {np.mean(self.hist[-20:]):6.2f} $/day", flush=True)
        if len(self.hist) % self.every == 0:
            save(self.model, self.name, self.hist)


def save(agent, name, hist):
    os.makedirs(os.path.join(HERE, "agents"), exist_ok=True)
    agent.save(os.path.join(HERE, "agents", f"{name}.zip"))
    json.dump(dict(name=name, iterations=len(hist), cost_per_day=[float(c) for c in hist]),
              open(os.path.join(HERE, "agents", f"{name}.json"), "w"))


if __name__ == "__main__":
    name = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 400
    xi, memory, head, priv, drift = CONFIGS[name]
    weather = load_weather()
    folder = os.path.join(HERE, "agents")
    os.makedirs(folder, exist_ok=True)
    resume = "--resume" in sys.argv
    if memory:
        agent = GRUAgent(prediction_head=head)
        before = []
        if resume and os.path.exists(os.path.join(folder, f"{name}.pt")):
            agent.load_state_dict(torch.load(os.path.join(folder, f"{name}.pt"), weights_only=True))
            saved = json.load(open(os.path.join(folder, f"{name}.json")))
            before = saved.get("cost_per_day", saved.get("cout_par_jour", []))
        torch.manual_seed(len(before))
        env = Houses(weather, N, xi=xi, drift=drift, seed=1 + len(before))
        hist = before + train_gru(agent, env, iterations=iters)
        torch.save(agent.state_dict(), os.path.join(folder, f"{name}.pt"))
        json.dump(dict(name=name, iterations=len(hist), head=head, cost_per_day=[float(c) for c in hist]),
                  open(os.path.join(folder, f"{name}.json"), "w"))
    else:
        agent = make_agent(weather, n=N, xi=xi, drift=drift, privileged=priv, seed=1)
        before = []
        if resume and os.path.exists(os.path.join(folder, f"{name}.zip")):
            agent = PPO.load(os.path.join(folder, f"{name}.zip"), env=agent.get_env(), device="cpu")
            before = json.load(open(os.path.join(folder, f"{name}.json"))).get("cost_per_day", [])
        log = Log(name, before)
        agent.learn(total_timesteps=iters * N * H, callback=log, reset_num_timesteps=not before)
        save(agent, name, log.hist)
