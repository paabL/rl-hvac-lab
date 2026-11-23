"""Schémas explicatifs du lab, dessinés avec matplotlib (lang="fr" ou "en")."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ENCRE, DOUX, CACHE, ACCENT = "#222222", "#6b6b6b", "#c45a1a", "#2b5fab"

TXT = {
    "fr": dict(
        maison_titre="La maison : deux températures, trois paramètres cachés",
        ext="Extérieur\n$T_{ext}$", air="Air\n$T_a$, capacité $C_a$", masse="Masse (murs, planchers)\n$T_m$, capacité $C_m$",
        pompe="Pompe à chaleur\n$Q = u \\cdot \\mu \\cdot UA \\cdot 46\\,K$", gains="Occupants,\nappareils",
        pertes="pertes\n$UA\\,(T_{ext}-T_a)$", echange="échange\n$H_{am}\\,(T_m-T_a)$",
        cache="cachés à l'agent : $UA$ (isolation), $C_m$ (inertie), $\\mu$ (taille de la pompe), et $T_m$",
        boucle_titre="La boucle du RL : l'agent ne voit jamais la maison, seulement ses réactions",
        agent="Agent\npolitique $\\pi(a_t \\mid s_t)$", env="Maison\nparamètres cachés $\\xi$",
        action="action $a_t$ : puissance de la pompe (0 à 100 %)",
        obs="observation $s_{t+1}$ : $T_a$, $T_{ext}$, prix, heure",
        rec="reward $r_t = -$(facture + $\\lambda \\times$ inconfort)",
        agent_titre="L'agent à mémoire : une mémoire, trois têtes",
        x="observation $x_t$", enc="encodeur\n(MLP)", gru="mémoire GRU\n$h_t = f(W[x_t, h_{t-1}] + w_0)$",
        acteur="actor\n$\\pi(a_t \\mid h_t)$", critique="critic\n$V(h_t)$ : retour attendu", pred="prédiction\n$\\hat T_{a,t+1}$",
        rl="appris par RL (PPO)", auto="auto-supervisé :\nla cible est le thermomètre", boucle_h="$h_{t-1}$",
        act_in="+ action $a_t$"),
    "en": dict(
        maison_titre="The house: two temperatures, three hidden parameters",
        ext="Outside\n$T_{ext}$", air="Air\n$T_a$, capacity $C_a$", masse="Mass (walls, floors)\n$T_m$, capacity $C_m$",
        pompe="Heat pump\n$Q = u \\cdot \\mu \\cdot UA \\cdot 46\\,K$", gains="People,\nappliances",
        pertes="loss\n$UA\\,(T_{ext}-T_a)$", echange="exchange\n$H_{am}\\,(T_m-T_a)$",
        cache="hidden from the agent: $UA$ (insulation), $C_m$ (thermal mass), $\\mu$ (heat-pump size), and $T_m$",
        boucle_titre="The RL loop: the agent never sees the house, only how it responds",
        agent="Agent\npolicy $\\pi(a_t \\mid s_t)$", env="House\nhidden parameters $\\xi$",
        action="action $a_t$: heat-pump power (0 to 100%)",
        obs="observation $s_{t+1}$: $T_a$, $T_{ext}$, price, hour",
        rec="reward $r_t = -$(bill + $\\lambda \\times$ discomfort)",
        agent_titre="The memory agent: one memory, three heads",
        x="observation $x_t$", enc="encoder\n(MLP)", gru="GRU memory\n$h_t = f(W[x_t, h_{t-1}] + w_0)$",
        acteur="actor\n$\\pi(a_t \\mid h_t)$", critique="critic\n$V(h_t)$: expected return", pred="prediction\n$\\hat T_{a,t+1}$",
        rl="learned by RL (PPO)", auto="self-supervised:\nthe target is the thermometer", boucle_h="$h_{t-1}$",
        act_in="+ action $a_t$"),
}


def _boite(ax, x, y, w, h, texte, couleur=ENCRE, fond="white", gras=False, taille=10.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fond, ec=couleur, lw=1.4))
    ax.text(x, y, texte, ha="center", va="center", fontsize=taille, color=ENCRE, fontweight="bold" if gras else "normal")


def _fleche(ax, a, b, texte="", couleur=ENCRE, decal=(0, 0.08), style="-|>", courbe=0.0, taille=9.5, va="bottom"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=couleur, lw=1.3,
                                                    connectionstyle=f"arc3,rad={courbe}", shrinkA=2, shrinkB=2))
    if texte:
        ax.text((a[0] + b[0]) / 2 + decal[0], (a[1] + b[1]) / 2 + decal[1], texte, ha="center", va=va,
                fontsize=taille, color=couleur)


def _cadre(titre, taille=(11, 4.2)):
    fig, ax = plt.subplots(figsize=taille)
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title(titre, fontsize=12.5, loc="left", color=ENCRE)
    return fig, ax


def schema_maison(lang="fr"):
    t = TXT[lang]
    fig, ax = _cadre(t["maison_titre"])
    ax.set_xlim(0, 12)
    _boite(ax, 1.1, 2.3, 1.8, 1.0, t["ext"])
    _boite(ax, 5.0, 2.3, 2.4, 1.0, t["air"], couleur=ACCENT, gras=True)
    _boite(ax, 10.2, 2.3, 2.8, 1.0, t["masse"], couleur=CACHE)
    _boite(ax, 5.0, 0.6, 2.9, 0.9, t["pompe"], couleur=CACHE)
    _boite(ax, 1.9, 0.6, 1.6, 0.8, t["gains"], couleur=DOUX)
    _fleche(ax, (3.8, 2.3), (2.0, 2.3), t["pertes"], couleur=CACHE, style="<|-|>")
    _fleche(ax, (6.2, 2.3), (8.8, 2.3), t["echange"], style="<|-|>")
    _fleche(ax, (5.0, 1.05), (5.0, 1.8), couleur=CACHE)
    _fleche(ax, (2.7, 0.8), (4.0, 1.8), couleur=DOUX, courbe=-0.2)
    ax.text(6.0, 3.7, t["cache"], ha="center", fontsize=10, color=CACHE)
    return fig


def schema_boucle(lang="fr"):
    t = TXT[lang]
    fig, ax = _cadre(t["boucle_titre"], (11, 3.4))
    ax.set_ylim(0.4, 3.2)
    _boite(ax, 2.0, 1.8, 2.6, 1.2, t["agent"], couleur=ACCENT, gras=True)
    _boite(ax, 9.0, 1.8, 2.6, 1.2, t["env"], couleur=CACHE, gras=True)
    _fleche(ax, (3.3, 2.25), (7.7, 2.25), t["action"])
    _fleche(ax, (7.7, 1.8), (3.3, 1.8), t["obs"])
    _fleche(ax, (7.7, 1.35), (3.3, 1.35), t["rec"], couleur=ACCENT, decal=(0, -0.08), va="top")
    return fig


def schema_agent(lang="fr"):
    t = TXT[lang]
    fig, ax = _cadre(t["agent_titre"], (11, 4.6))
    ax.set_ylim(0, 4.6)
    _boite(ax, 1.0, 2.3, 1.6, 0.8, t["x"], couleur=DOUX)
    _boite(ax, 3.0, 2.3, 1.4, 0.9, t["enc"])
    _boite(ax, 5.6, 2.3, 2.8, 1.0, t["gru"], couleur=ACCENT, gras=True)
    _boite(ax, 9.4, 3.7, 2.4, 0.8, t["acteur"], couleur=ACCENT)
    _boite(ax, 9.4, 2.5, 2.4, 0.8, t["critique"], couleur=ACCENT)
    _boite(ax, 9.4, 1.1, 2.4, 0.8, t["pred"], couleur=CACHE)
    _fleche(ax, (1.8, 2.3), (2.3, 2.3))
    _fleche(ax, (3.7, 2.3), (4.2, 2.3))
    _fleche(ax, (7.0, 2.5), (8.2, 3.6))
    _fleche(ax, (7.0, 2.4), (8.2, 2.5))
    _fleche(ax, (7.0, 2.1), (8.2, 1.2))
    ax.annotate("", xy=(4.9, 2.8), xytext=(6.3, 2.8), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2,
                                                                       connectionstyle="arc3,rad=1.6"))
    ax.text(5.6, 3.55, t["boucle_h"], ha="center", fontsize=10, color=ACCENT)
    ax.text(7.3, 1.0, t["act_in"], ha="center", fontsize=9.5, color=CACHE)
    ax.text(9.4, 4.32, t["rl"], ha="center", fontsize=9.5, color=ACCENT)
    ax.text(9.4, 0.2, t["auto"], ha="center", fontsize=9.5, color=CACHE)
    return fig


# English names for the English notebook
def house_diagram(lang="en"):
    return schema_maison(lang)


def loop_diagram(lang="en"):
    return schema_boucle(lang)


def agent_diagram(lang="en"):
    return schema_agent(lang)


# ---------------------------------------------------------------- préchauffage

TXT_PRE = {
    "fr": dict(
        titre="Le bon plan de chauffage dépend de la dynamique de la maison",
        gauche="Maison lourde, bien isolée : préchauffer, puis laisser filer",
        droite="Maison légère, passoire : le même plan échoue",
        pointe="pointe : électricité\n7 fois plus chère", confort="confort",
        naif="thermostat : chauffe pendant la pointe", pre="préchauffage : chauffe avant, coupe pendant",
        meme="même plan : trop froid", adapte="plan adapté : chauffer aussi pendant la pointe",
        stocke="la chaleur\ns'accumule", filer="refroidit lentement\n(grande inertie)", chute="refroidit vite\n(petite inertie, fuites)",
        heure="heure du matin", temp="température intérieure (°C)",
        bas="Pour préchauffer juste assez, au bon moment, il faut savoir à quelle vitesse la maison se réchauffe et se refroidit : "
            "connaître sa dynamique, ou l'apprendre."),
    "en": dict(
        titre="The right heating plan depends on the dynamics of the house",
        gauche="Heavy, well-insulated house: preheat, then coast",
        droite="Light, leaky house: the same plan fails",
        pointe="peak: electricity\n7 times more expensive", confort="comfort",
        naif="thermostat: heats during the peak", pre="preheating: heat before, cut during",
        meme="same plan: too cold", adapte="adapted plan: also heat during the peak",
        stocke="heat\nbuilds up", filer="cools slowly\n(high thermal mass)", chute="cools fast\n(low mass, leaks)",
        heure="hour of the morning", temp="indoor temperature (°C)",
        bas="To preheat just enough, at the right time, you need to know how fast the house warms up and cools down: "
            "know its dynamics, or learn them."),
}


def _maison_1t(UA, C, plan, t_ext=-10.0, pas=60.0, heures=12.0):
    """Une température, C dT/dt = UA (T_ext - T) + u P_max : juste ce qu'il faut pour le schéma."""
    T, P, sortie = 21.0, 1.6 * UA * 46.0, []
    for k in range(int(heures * 3600 / pas)):
        T = T + pas * (UA * (t_ext - T) + plan(k * pas / 3600, T) * P) / C
        sortie.append(T)
    return np.arange(len(sortie)) * pas / 3600, np.array(sortie)


def schema_prechauffage(lang="fr"):
    t = TXT_PRE[lang]
    pointe = lambda h: 6 <= h < 9
    garde = lambda cible: (lambda h, T: float(np.clip(2 * (cible - T), 0, 1)))
    def plan(debut, haut=23.0, pendant=None):
        def p(h, T):
            if pointe(h):
                return 0.0 if pendant is None else float(np.clip(2 * (pendant - T), 0, 1))
            return garde(haut)(h, T) if debut <= h < 6 else garde(21.0)(h, T)
        return p
    lourde, legere = (100.0, 1.6e7), (250.0, 6e6)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, titre in zip(axs, (t["gauche"], t["droite"])):
        ax.axvspan(6, 9, color="#999999", alpha=0.18, lw=0)
        ax.axhspan(20, 23, color="#4caf50", alpha=0.10, lw=0)
        ax.text(7.5, 23.75, t["pointe"], ha="center", va="center", fontsize=10.5, color="#555555")
        ax.text(11.9, 20.15, t["confort"], ha="right", va="bottom", fontsize=10, color="#2e7d32")
        ax.set_title(titre, fontsize=12, loc="left")
        ax.set_xticks([0, 3, 6, 9, 12]); ax.set_xticklabels(["0 h", "3 h", "6 h", "9 h", "12 h"])
        ax.set_xlim(0, 12); ax.set_ylim(15, 24.4); ax.set_xlabel(t["heure"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, T = _maison_1t(*lourde, garde(21.0)); axs[0].plot(h, T, "--", color="#777777", lw=2, label=t["naif"])
    h, T = _maison_1t(*lourde, plan(3.0)); axs[0].plot(h, T, color=ACCENT, lw=2.6, label=t["pre"])
    axs[0].annotate(t["stocke"], xy=(4.6, 22.2), xytext=(1.2, 23.2), fontsize=10, color=ACCENT,
                    arrowprops=dict(arrowstyle="->", color=ACCENT))
    axs[0].annotate(t["filer"], xy=(8.2, 20.6), xytext=(9.3, 18.0), fontsize=10, color=ACCENT,
                    arrowprops=dict(arrowstyle="->", color=ACCENT))
    h, T = _maison_1t(*legere, plan(3.0)); axs[1].plot(h, T, color=CACHE, lw=2.6, label=t["meme"])
    h, T = _maison_1t(*legere, plan(5.0, pendant=20.6)); axs[1].plot(h, T, color="#2e7d32", lw=2.2, label=t["adapte"])
    axs[1].annotate(t["chute"], xy=(6.75, 18.3), xytext=(2.4, 18.4), fontsize=10, color=CACHE,
                    arrowprops=dict(arrowstyle="->", color=CACHE))
    axs[0].set_ylabel(t["temp"])
    for ax in axs:
        ax.legend(loc="lower left", fontsize=9.5, frameon=False)
    fig.suptitle(t["titre"], fontsize=13.5, x=0.01, ha="left", y=1.0)
    fig.text(0.01, -0.03, t["bas"], fontsize=10.5, color=ENCRE, ha="left", wrap=True)
    fig.tight_layout()
    return fig


def preheating_diagram(lang="en"):
    return schema_prechauffage(lang)


# ---------------------------------------------------------------- un agent, plusieurs maisons

TXT_FLOTTE = {
    "fr": dict(
        titre="Un seul agent pour toutes les maisons : il règle chaque thermostat sans voir la maison",
        agent="Agent", agent2="une seule politique,\nles mêmes poids partout",
        action="action :\npuissance de la pompe",
        m1="immeuble en béton : lourd, bien isolé", m2="vieille maison : une passoire", m3="bungalow : petite pompe à chaleur",
        autres="... et 997 autres maisons, toutes différentes",
        obs="observation de chaque maison : températures, prix, heure", rec="reward : − (facture + inconfort)",
        cache="en orange : caché à l'agent"),
    "en": dict(
        titre="One agent for every house: it sets each thermostat without seeing the house",
        agent="Agent", agent2="one policy,\nthe same weights everywhere",
        action="action:\nheat-pump power",
        m1="concrete apartment block: heavy, well insulated", m2="old house: very leaky", m3="bungalow: small heat pump",
        autres="... and 997 more houses, all different",
        obs="observation from each house: temperatures, price, hour", rec="reward: − (bill + discomfort)",
        cache="in orange: hidden from the agent"),
}
IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def _image(ax, nom, x0, y0, largeur):
    img = plt.imread(os.path.join(IMAGES, nom + ".png"))
    h = largeur * img.shape[0] / img.shape[1]
    ax.imshow(img, extent=[x0, x0 + largeur, y0, y0 + h], zorder=3)
    return h


def schema_flotte(lang="fr"):
    t = TXT_FLOTTE[lang]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12.6); ax.set_ylim(-0.35, 6.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(t["titre"], fontsize=13, loc="left", color=ENCRE)
    ax.text(12.55, 6.35, t["cache"], ha="right", fontsize=10, color=CACHE)
    _image(ax, "robot", 0.4, 3.0, 2.2)
    ax.text(1.5, 2.75, t["agent"], ha="center", va="top", fontsize=13, fontweight="bold", color=ENCRE)
    ax.text(1.5, 2.35, t["agent2"], ha="center", va="top", fontsize=10.5, color=ENCRE)
    virgule = (lambda x: x.replace(".", ",")) if lang == "fr" else (lambda x: x)
    lignes = [(5.4, "immeuble", 1.25, t["m1"], "21.4 °C"), (3.85, "maison", 1.75, t["m2"], "19.6 °C"),
              (2.35, "bungalow", 2.3, t["m3"], "22.1 °C")]
    for y, nom, larg, legende, temp in lignes:
        th = _image(ax, "thermostat_vide", 5.2, y - 0.5, 1.25)
        ax.text(5.83, y - 0.5 + th * 0.69, virgule(temp), ha="center", va="center", fontsize=9.5, zorder=4, color=ENCRE)
        _image(ax, nom, 8.1, y - 0.45, larg)
        ax.text(8.1, y - 0.62, legende, ha="left", va="top", fontsize=10, color=CACHE)
        ax.annotate("", xy=(8.0, y), xytext=(6.55, y), arrowprops=dict(arrowstyle="-|>", color=ENCRE, lw=1.4))
        ax.annotate("", xy=(5.1, y), xytext=(2.8, 4.1), arrowprops=dict(arrowstyle="-|>", color=ENCRE, lw=1.6))
    ax.text(3.0, 5.0, t["action"], ha="center", va="bottom", fontsize=10.5, color=ENCRE)
    ax.text(8.1, 1.2, t["autres"], ha="left", fontsize=10.5, color=DOUX, style="italic")
    for y, couleur, style, texte in [(0.75, ACCENT, "--", t["obs"]), (0.05, "#2e7d32", ":", t["rec"])]:
        ax.annotate("", xy=(1.5 if y > 0.5 else 1.0, 1.5), xytext=(11.8, y),
                    arrowprops=dict(arrowstyle="-|>", color=couleur, lw=1.7, ls=style,
                                    connectionstyle="angle,angleA=180,angleB=-90,rad=8"))
        ax.text(6.6, y + 0.07, texte, ha="center", va="bottom", fontsize=10.5, color=couleur)
    return fig


def fleet_diagram(lang="en"):
    return schema_flotte(lang)
