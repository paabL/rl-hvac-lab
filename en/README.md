# RL lab, English version: one thermostat, a thousand houses (prototype v3)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/en/rl_lab_prototype_en.ipynb) instructor version, with the solutions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paabL/rl-hvac-lab/blob/main/en/rl_lab_student_en.ipynb) student version, to fill in on the board

A 2-hour Jupyter lab on reinforcement learning for home heating (IFT3700/IFT6758). This is a **prototype**, not the final version. Everything runs on CPU, in Colab or locally.

## Run it in Colab

Click a badge, then run the cells in order. The first code cell (**0 · Setup**) installs Stable-Baselines3 if needed and downloads the companion files by itself. If it cannot reach GitHub, it shows a **Choose files** button: pick `rl_lab_en.zip`.

## Run it locally

Open the notebook in Jupyter from this folder. You need Python 3.10 or newer with `numpy pandas scikit-learn matplotlib torch ipywidgets gymnasium stable-baselines3`.

## Settings (top of the Setup cell)

- `SHOW_WIDGETS = False` replaces the sliders with static figures.
- `RUN_TRAINING = True` retrains the "classic" agent live, about 1.5 minutes on CPU. All other agents are always loaded pre-trained.
- `REAL_WEATHER = True` replaces the synthetic weather with real Montreal Januaries 2015-2024 from Open-Meteo (needs Internet). The provided agents were trained on the synthetic weather.

## Preparing to teach

- Open `rl_lab_prototype_en_executed.ipynb` first: it is the same notebook already run, so you can see every figure and number without running anything.
- The notebook has collapsible **Instructor notes** and **Answer** blocks (click to open). Start with "Instructor notes: the lab in two minutes" at the top. They give the story, what to say, the expected results and the answers to the questions.
- In class, project the student version and fill in its `___` live. The one-page cheat sheet in `../antiseche/` (`cheat_sheet_en.pdf` or `.html`) has the code to type, with a comment on each line saying what to say.

## Files

| File | What it is |
| --- | --- |
| `rl_lab_student_en.ipynb` | **the student version to fill in on the board**: each exercise has `___`; while it is empty, the rest runs with the lab's version |
| `rl_lab_prototype_en.ipynb` | the notebook with the solutions, without outputs |
| `rl_lab_prototype_en_executed.ipynb` | the same notebook, already run (not in the zip) |
| `rl_lab_en.zip` | everything Colab needs, without the executed notebook |
| `lab_utils_en.py` | house simulator, thermostats, REINFORCE, saddle toy |
| `rl_sb3_en.py` | the house as a Gymnasium environment, the memoryless Stable-Baselines3 PPO agents, evaluation and memory diagnostics |
| `gru_ppo_en.py` | our memory PPO: GRU, actor, critic and prediction head (to call, not to read) |
| `blanks.py` | the `___` and the exercise checks of the student version |
| `schemas.py`, `images/` | the notebook's diagrams, shared with the French version (robot and building drawings from the PluRL deck) |
| `agents/` | the pre-trained agents (SB3 `.zip` without memory, `.pt` for our GRUs) and their learning curves (`.json`) |
| `montreal_weather.csv` | the hourly January weather used (synthetic for now) |
| `train_agents_en.py` | retrains one agent, e.g. `python train_agents_en.py in_context 400` (add `--resume` to continue). Not needed to teach. |
