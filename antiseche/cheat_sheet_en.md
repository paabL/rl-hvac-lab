# Cheat sheet · RL lab, prototype v3

Keep it next to Colab, in the student notebook (`rl_lab_student_en.ipynb`). In each `EXERCISE`, replace the `___` with the code below. The `#` comments say roughly what to say while typing each line. While an exercise is empty, the rest runs with the lab's version.

**Run order:** cell 0 (Setup, downloads the files) · Gym loop · §1 diagrams · §2 vocabulary, house, ex. 1, thermostat · §3 ex. 2, classic agent · §4 ex. 3, robust vs oracle · §5 agent, ex. 4, saddle, memory table · §6 memorize, PCA · §7 ex. 5, drift · §8 wrap-up.

## 1/5 · `my_thermal_step` (Section 2)

```python
dTa = (xi["UA"] * (t_ext - Ta)     # air leaks out: UA × gap
       + H_AM * (Tm - Ta)          # it exchanges with the walls
       + q_pump + Q_INT) / C_AIR   # + pump + people, ÷ capacity
dTm = H_AM * (Ta - Tm) / xi["Cm"]  # walls follow the air, slowly
```

**Say:** heat capacity × rate of change = sum of heat flows, then one Euler step (already written). The mass is slow and hidden: it is what makes preheating possible.

**Expect:** `Exercise 1: ✓ correct`.

## 2/5 · `my_update` (Section 3)

```python
weights = (G * (A - P)).reshape(-1)       # return × (action − prob)
gradient = X2.T @ weights / len(weights)  # Lecture 2's Xᵀ(y − ŷ)/n
```

**Say:** the logistic regression of Lecture 2: the label is the action taken, and each example is weighted by its return. Actions followed by a good return become more likely.

**Expect:** cost from 116 to about 12 $/day; weight "Ta − 21" ≈ −2.1, "peak now" ≈ −0.4, "peak soon" ≈ 0 (no preheating yet).

## 3/5 · `my_houses` (Section 4)

```python
w = width * np.log(np.sqrt(FACTOR[k]))  # half-box width, in log
houses[k] = NOMINAL[k] * np.exp(        # nominal × a random factor
    rng.uniform(-w, w, n))              # drawn uniformly in log
```

**Say:** domain randomization is data augmentation for RL. In log space, "twice as much" and "half as much" are equally likely.

**Expect:** UA from 87 to 260, Cm from 1.7e7 to 5.2e7, mu from 0.92 to 2.77.

## 4/5 · `my_grad_J` (Section 5)

```python
dJ_db = (2 * s_xi**2 * error * c  # reading pays if memory holds ξ
         + 2 * b * s_eps**2)      # but reading noise costs
dJ_dc = 2 * s_xi**2 * error * b   # writing pays if someone reads
```

**Say:** at (0, 0) both derivatives are zero: reading an empty memory is useless, and so is writing a memory nobody reads. It is a saddle. The green cross is the target.

**Expect:** gradient at the saddle `(0.0, -0.0)`. Head slider at 0.5: every run goes to the cross.

## 5/5 · `half_life` (Section 7)

```python
estimate = x.expanding().mean()       # cumulative: forgets nothing
estimate = x.ewm(halflife=12).mean()  # forgets: half-life 12 h
```

**Say:** the cumulative mean takes as long to correct as its age; the exponential mean forgets, and its half-life stays constant.

**Expect:** cumulative 11, 23, 47, 95, 191; exponential 7, 9, then 11.

## Quick answers and numbers

- **PPO (Section 3):** REINFORCE's policy gradient with the advantage Â (better or worse than expected, thanks to the critic) instead of the return, plus a clip: L = E[min(ρ·Â, clip(ρ, 0.8, 1.2)·Â)], ρ = π_new / π_old. One update cannot move the policy too much.
- **REINFORCE weights:** the warmer, the less it heats; it avoids the peak; it does not preheat yet.
- **Classic agent (nominal house):** thermostat 7.7, preheating thermostat 6.5, agent 6.0 $/day.
- **Gap to the oracle (5.5 $/day):** thermostat +88%, classic elsewhere +126%, robust +11%, GRU without head +6%, GRU with head +3%.
- **Saddle:** a wider box escapes faster; the head gives writing a gradient of its own, and reading follows.
- **Which agent listens to its memory?** The one whose cost jumps most when handed another house's memory (swap column: +23% with head, +11% without).
- **Memorize:** 4 houses, +3.5% at home and +15% elsewhere; 16 houses, +2% everywhere.
- **Half-life after the jump:** trained without drift, it behaves like the cumulative mean (+13% after the jump); with drift, like the exponential mean (+7%).
