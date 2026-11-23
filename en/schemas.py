"""Schémas explicatifs du lab, dessinés avec matplotlib (lang="fr" ou "en")."""
import matplotlib.pyplot as plt
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
        rec="récompense $r_t = -$(facture + $\\lambda \\times$ inconfort)",
        agent_titre="L'agent à mémoire : une mémoire, trois têtes",
        x="observation $x_t$", enc="encodeur\n(MLP)", gru="mémoire GRU\n$h_t = f(W[x_t, h_{t-1}] + w_0)$",
        acteur="acteur\n$\\pi(a_t \\mid h_t)$", critique="critique\n$V(h_t)$ : retour attendu", pred="prédiction\n$\\hat T_{a,t+1}$",
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
