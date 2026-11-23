# RL lab, English version: one thermostat, a thousand houses

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/en/rl_lab_prototype_en.ipynb)


A 2-hour Jupyter lab on reinforcement learning for home heating (IFT3700/IFT6758). This is a **prototype**, not the final version. Everything runs on CPU, in Colab or locally.

## Run it in Colab

Click the badge above, then run the cells in order. The first code cell (**0 · Setup**) downloads the companion files by itself. If it cannot reach GitHub, it shows a **Choose files** button: pick `rl_lab_en.zip`.

## Run it locally

Unzip `rl_lab_en.zip`, then open `rl_lab_prototype_en.ipynb` in Jupyter from that folder. You need Python 3.10 or newer with `numpy pandas scikit-learn matplotlib torch ipywidgets`.

## Settings (top of the Setup cell)

- `SHOW_WIDGETS = False` replaces the sliders with static figures.
- `RUN_TRAINING = True` retrains the "classic" agent live, about 2 minutes on CPU. All other agents are always loaded pre-trained.
- `REAL_WEATHER = True` replaces the synthetic weather with real Montreal Januaries 2015-2024 from Open-Meteo (needs Internet). The provided agents were trained on the synthetic weather.

## Preparing to teach

- Open `rl_lab_prototype_en_executed.ipynb` first: it is the same notebook already run, so you can see every figure and number without running anything.
- The notebook has collapsible **Instructor notes** and **Answer** blocks (click to open). Start with "Instructor notes: the lab in two minutes" at the top. They give the story, what to say, the expected results and the answers to the questions.
- The five `EXERCISE k/5` cells contain the solution right after `# YOUR CODE HERE`, so you can live-code it or hand it to students.

## Files

| File | What it is |
| --- | --- |
| `rl_lab_prototype_en.ipynb` | the notebook, without outputs |
| `rl_lab_prototype_en_executed.ipynb` | the same notebook, already run (not in the zip) |
| `rl_lab_en.zip` | everything Colab needs: the files below plus the notebook |
| `lab_utils_en.py` | house simulator, thermostats, agents, PPO, REINFORCE, memory diagnostics, saddle toy |
| `schemas.py` | the three diagrams (house, RL loop, memory agent), shared with the French version |
| `agents/` | the pre-trained agents (`.pt`) and their learning curves (`.json`) |
| `montreal_weather.csv` | the hourly January weather used (synthetic for now) |
| `train_agents_en.py` | retrains one agent, e.g. `python train_agents_en.py in_context 400` (add `--resume` to continue). Not needed to teach. |
