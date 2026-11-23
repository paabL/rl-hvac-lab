"""Pre-trains the lab's agents. Usage: python train_agents_en.py NAME [ITERATIONS] [--resume]

Not needed to teach the lab: the trained agents are already in agents/.
Memory agents take about 20-30 min on CPU for 1,200 iterations.
"""
import json
import os
import sys

import numpy as np
import torch

from lab_utils_en import Agent, Houses, PRIV_DIM, load_weather, nominal_house, sample_houses, train

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
N = 256


def fixed_houses(k, seed=11):
    """Each episode draws its 256 houses from the same k houses (the 'memorize' worlds)."""
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

if __name__ == "__main__":
    name = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 150
    xi, memory, head, priv, drift = CONFIGS[name]
    torch.manual_seed(0)
    env = Houses(load_weather(), N, xi=xi, drift=drift, seed=1)
    agent = Agent(obs_dim=PRIV_DIM if priv else 9, memory=memory, prediction_head=head)
    path = os.path.join(HERE, "agents", f"{name}.pt")
    before = []
    if "--resume" in sys.argv and os.path.exists(path):
        agent.load_state_dict(torch.load(path, weights_only=True))
        before = json.load(open(path.replace(".pt", ".json")))["cost_per_day"]
        torch.manual_seed(len(before))
        env = Houses(load_weather(), N, xi=xi, drift=drift, seed=1 + len(before))
    hist = before + train(agent, env, iterations=iters, privileged=priv)
    os.makedirs(os.path.join(HERE, "agents"), exist_ok=True)
    torch.save(agent.state_dict(), path)
    json.dump(dict(name=name, iterations=len(hist), prediction_head=head, cost_per_day=[float(c) for c in hist]),
              open(path.replace(".pt", ".json"), "w"))
