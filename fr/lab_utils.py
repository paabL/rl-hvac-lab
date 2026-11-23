"""Lab RL pour le chauffage : le simulateur de maisons, les thermostats de référence, REINFORCE et le jouet du col.

Tout tourne sur CPU, en NumPy, vectorisé sur N maisons. Les agents PPO sont dans rl_sb3.py (Stable-Baselines3).
"""
from __future__ import annotations

import io
import json
import math
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- météo

def meteo_synthetique(n_janviers=20, seed=0):
    """Températures horaires de janviers montréalais synthétiques (°C), une ligne par heure.

    Moyenne journalière AR(1) autour de -10 °C (écart-type 6, corrélation 0,8 par jour),
    cycle diurne de 4 °C (minimum à 6 h, maximum à 15 h), bruit horaire de 0,5 °C.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(n_janviers):
        m = rng.normal(0, 6)
        for d in range(31):
            m = 0.8 * m + math.sqrt(1 - 0.8**2) * rng.normal(0, 6)
            for h in range(24):
                diurne = -2.0 * math.cos(2 * math.pi * (h - 3) / 24)
                rows.append((y, d, h, -10 + m + diurne + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["annee", "jour", "heure", "t_ext"])


def telecharger_open_meteo(annees=range(2015, 2025)):
    """Janviers horaires de Montréal depuis l'archive Open-Meteo (données CC BY 4.0, à citer)."""
    frames = []
    for i, a in enumerate(annees):
        url = ("https://archive-api.open-meteo.com/v1/archive?latitude=45.50&longitude=-73.57"
               f"&start_date={a}-01-01&end_date={a}-01-31&hourly=temperature_2m&timezone=America/Toronto")
        data = json.load(urllib.request.urlopen(url, timeout=20))["hourly"]
        t = pd.to_datetime(data["time"])
        frames.append(pd.DataFrame({"annee": i, "jour": t.day - 1, "heure": t.hour, "t_ext": data["temperature_2m"]}))
    return pd.concat(frames, ignore_index=True)


def charger_meteo(chemin=os.path.join(HERE, "meteo_montreal.csv")):
    if os.path.exists(chemin):
        return pd.read_csv(chemin)
    try:
        df = telecharger_open_meteo()
    except Exception:
        df = meteo_synthetique()
    df.to_csv(chemin, index=False)
    return df


# ---------------------------------------------------------------- maisons

NOMINAL = dict(UA=150.0, Cm=3.0e7, mu=1.6)          # W/K, J/K, surdimensionnement de la pompe
FACTEUR = dict(UA=3.0, Cm=3.0, mu=3.0)              # largeur de la boîte : de nominal/sqrt(f) à nominal*sqrt(f)
CA, HAM, QINT = 8.0e6, 1000.0, 400.0                 # capacité de l'air, couplage air-masse, gains internes
T_DESIGN, T_CONSIGNE = -25.0, 21.0
BANDE = (20.0, 23.0)
PRIX_BASE, PRIX_POINTE = 0.07, 0.50                 # $/kWh
LAMBDA = 1.0                                        # $ par K·h d'inconfort
NIVEAUX = np.array([0.0, 0.25, 0.5, 0.75, 1.0])     # actions : fraction de la puissance de la pompe
H = 168                                             # un épisode = 7 jours au pas d'une heure


def tirer_maisons(n, largeur=1.0, rng=None):
    """ξ = nominal × exp(U(-w, w)) en log, w = largeur × log(sqrt(facteur)). Retourne un dict de tableaux (n,)."""
    rng = rng or np.random.default_rng()
    return {k: NOMINAL[k] * np.exp(rng.uniform(-1, 1, n) * largeur * math.log(math.sqrt(FACTEUR[k]))) for k in NOMINAL}


def maison_nominale(n):
    return {k: np.full(n, NOMINAL[k]) for k in NOMINAL}


def xi_normalise(xi):
    """log(ξ/nominal) / log(sqrt(facteur)) : chaque paramètre dans [-1, 1] sur la boîte."""
    return np.stack([np.log(xi[k] / NOMINAL[k]) / math.log(math.sqrt(FACTEUR[k])) for k in NOMINAL], -1)


def est_pointe(heure_semaine):
    h, jour = heure_semaine % 24, (heure_semaine // 24) % 7
    return ((jour < 5) & (((h >= 6) & (h < 9)) | ((h >= 16) & (h < 20)))).astype(np.float32)


def cop(t_ext):
    return np.clip(2.8 + 0.05 * t_ext, 1.5, 4.0)


def step(Ta, Tm, u, t_ext, xi, sous_pas=6):
    """Un pas d'une heure du modèle 2R2C (Euler explicite, 6 sous-pas de 10 min).

    Ta, Tm : températures de l'air et de la masse (°C) ; u : fraction de puissance dans [0, 1].
    Retourne Ta, Tm à l'heure suivante et l'énergie électrique consommée (kWh).
    """
    q_pompe = u * xi["mu"] * xi["UA"] * (T_CONSIGNE - T_DESIGN)   # W thermiques
    dt = 3600.0 / sous_pas
    for _ in range(sous_pas):
        dTa = (xi["UA"] * (t_ext - Ta) + HAM * (Tm - Ta) + q_pompe + QINT) / CA
        dTm = HAM * (Ta - Tm) / xi["Cm"]
        Ta, Tm = Ta + dt * dTa, Tm + dt * dTm
    kwh = q_pompe / cop(t_ext) / 1000.0
    return Ta, Tm, kwh


def cout(kwh, prix, Ta):
    """Coût horaire en $ : facture + λ × inconfort (K·h hors de la bande 20-23 °C)."""
    inconfort = np.maximum(BANDE[0] - Ta, 0) + np.maximum(Ta - BANDE[1], 0)
    return prix * kwh + LAMBDA * inconfort, inconfort


class Maisons:
    """N maisons simulées ensemble. Observation publique et observation privilégiée (oracle)."""

    def __init__(self, meteo, n, xi=None, largeur=1.0, derive=False, seed=0, bruit=0.1):
        self.rng = np.random.default_rng(seed)
        self.T = meteo.pivot_table(index=["annee", "jour", "heure"], values="t_ext").values.ravel()
        self.n, self.largeur, self.derive, self.bruit = n, largeur, derive, bruit
        self.xi_fixe = xi

    def reset(self, xi=None, debuts=None):
        n = self.n
        self.xi0 = xi if xi is not None else (self.xi_fixe() if callable(self.xi_fixe) else
                                              self.xi_fixe if self.xi_fixe is not None else tirer_maisons(n, self.largeur, self.rng))
        self.xi = {k: v.copy() for k, v in self.xi0.items()}
        self.debut = debuts if debuts is not None else self.rng.integers(0, len(self.T) - H - 4, n)
        self.h0 = self.rng.integers(0, 7, n) * 24 if debuts is None else np.zeros(n, int)
        self.t = 0
        self.Ta = self.rng.uniform(20.0, 22.0, n)
        self.Tm = self.Ta - self.rng.uniform(0.0, 1.0, n)
        self.u_prec = np.zeros(n)
        # dérive : à une heure tirée au hasard, la pompe perd 50 % ou l'isolation se dégrade (×1,8)
        self.t_saut = self.rng.integers(24, H - 24, n) if self.derive else np.full(n, H + 1)
        self.type_saut = self.rng.integers(0, 2, n)
        return self.obs()

    def obs(self):
        t_ext = self.T[self.debut + self.t]
        hs = self.h0 + self.t
        futur = np.stack([est_pointe(hs + k) for k in (1, 2, 3)], -1)
        o = np.column_stack([
            (self.Ta + self.rng.normal(0, self.bruit, self.n) - 21.0) / 2.0,
            t_ext / 10.0,
            est_pointe(hs), futur,
            np.sin(2 * np.pi * (hs % 24) / 24), np.cos(2 * np.pi * (hs % 24) / 24),
            self.u_prec,
        ]).astype(np.float32)
        return o

    def obs_privilegiee(self):
        return np.column_stack([self.obs(), xi_normalise(self.xi), (self.Tm - 21.0) / 2.0]).astype(np.float32)

    def appliquer_derive(self):
        saut = self.t == self.t_saut
        if saut.any():
            pompe = saut & (self.type_saut == 0)
            fuite = saut & (self.type_saut == 1)
            self.xi["mu"] = np.where(pompe, self.xi["mu"] * 0.5, self.xi["mu"])
            self.xi["UA"] = np.where(fuite, self.xi["UA"] * 1.8, self.xi["UA"])

    def pas(self, a):
        """a : indices d'action (n,). Retourne obs, reward, coût, inconfort, kWh."""
        self.appliquer_derive()
        u = NIVEAUX[a]
        t_ext = self.T[self.debut + self.t]
        prix = np.where(est_pointe(self.h0 + self.t) > 0, PRIX_POINTE, PRIX_BASE)
        self.Ta, self.Tm, kwh = step(self.Ta, self.Tm, u, t_ext, self.xi)
        c, inc = cout(kwh, prix, self.Ta)
        self.u_prec = u
        self.t += 1
        return self.obs(), -c, c, inc, kwh


OBS_DIM = 9
PRIV_DIM = OBS_DIM + 4


# ---------------------------------------------------------------- politiques de référence

def _niveau(u):
    return np.abs(np.clip(u, 0, 1)[:, None] - NIVEAUX[None]).argmin(1)


def thermostat(env, cible=None):
    """Thermostat proportionnel à 21 °C, sans anticipation : u = 0,4 + 0,4 × (21 - Ta)."""
    cible = T_CONSIGNE if cible is None else cible
    return _niveau(0.4 + 0.4 * (cible - env.Ta))


def thermostat_malin(env):
    """Le même, mais monte à 22,5 °C les 3 heures avant la pointe et descend à 20 °C pendant."""
    hs = env.h0 + env.t
    cible = np.where(est_pointe(hs + 1) + est_pointe(hs + 2) + est_pointe(hs + 3) > 0, 22.5, T_CONSIGNE)
    cible = np.where(est_pointe(hs) > 0, 20.0, cible)
    return thermostat(env, cible)


# ---------------------------------------------------------------- jouets

def J(b, c, s_xi=1.0, s_eps=1.0):
    """Coût du jouet : la mémoire écrit m = c ξ + ε, l'action lit a = b m, l'action idéale est ξ."""
    return s_xi**2 * (b * c - 1) ** 2 + b**2 * s_eps**2


def grad_J(b, c, s_xi=1.0, s_eps=1.0):
    return 2 * s_xi**2 * (b * c - 1) * c + 2 * b * s_eps**2, 2 * s_xi**2 * (b * c - 1) * b


def descente(b, c, eta=0.05, pas=400, **kw):
    traj = [(b, c)]
    for _ in range(pas):
        gb, gc = grad_J(b, c, **kw)
        b, c = b - eta * gb, c - eta * gc
        traj.append((b, c))
    return np.array(traj)


# ---------------------------------------------------------------- REINFORCE : la régression logistique pondérée par le retour

def caracteristiques(env):
    """x = [1, Ta - 21, T_ext/10, pointe maintenant, pointe dans 1-3 h, sin, cos de l'heure]."""
    o = env.obs()
    return np.column_stack([np.ones(env.n), o[:, 0] * 2.0, o[:, 1], o[:, 2], o[:, 3:6].max(1), o[:, 6], o[:, 7]])


def reinforce(meteo, iterations=60, eta=0.05, n=512, seed=0, maj=None, verbose=True):
    """Politique de Bernoulli π(chauffer | x) = σ(wᵀx + w0) entraînée par REINFORCE sur une maison nominale.

    maj(w, X, A, P, G, eta) applique la mise à jour (l'exercice) ; par défaut, la solution.
    """
    rng = np.random.default_rng(seed)
    env = Maisons(meteo, n, xi=lambda: maison_nominale(n), seed=seed)
    w = np.zeros(7)
    hist = []
    for it in range(iterations):
        env.reset()
        X, A, P, R = [], [], [], []
        for t in range(H):
            x = caracteristiques(env)
            p = 1 / (1 + np.exp(-x @ w))
            a = (rng.random(n) < p).astype(float)
            _, r, _, _, _ = env.pas(np.where(a > 0, 4, 0))
            X.append(x); A.append(a); P.append(p); R.append(r)
        X, A, P, R = map(np.array, (X, A, P, R))                 # (H, n, ...) 
        G = np.cumsum(R[::-1], 0)[::-1]                           # retour à partir de chaque heure
        G = (G - G.mean(1, keepdims=True)) / (G.std() + 1e-8)     # baseline : la moyenne des maisons à la même heure
        w = (maj or maj_reinforce)(w, X, A, P, G, eta)
        hist.append(-R.sum(0).mean() / 7)
        if verbose and (it % 10 == 0 or it == iterations - 1):
            print(f"itération {it:3d}  coût moyen {hist[-1]:6.2f} $/jour")
    return w, hist


def maj_reinforce(w, X, A, P, G, eta):
    """w_j ← w_j + η Σ G (a - p) x_j : la descente de gradient de la régression logistique, pondérée par le retour."""
    X2 = X.reshape(-1, X.shape[-1])             # un exemple par ligne
    poids = (G * (A - P)).reshape(-1)           # retour × (action - probabilité)
    gradient = X2.T @ poids / len(poids)        # comme la régression logistique : Xᵀ (y - ŷ) / n
    return w + eta * gradient
