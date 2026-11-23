# Antisèche · Labo RL

À garder à côté de Colab, dans le notebook étudiant (`lab_rl_etudiant.ipynb`). Dans chaque `EXERCICE`, remplacer les `___` par le code ci-dessous. Les `#` disent en gros quoi dire en tapant chaque ligne. Tant qu'un exercice est vide, la suite tourne avec la version du labo.

**Déroulé :** cellule 0 (installation, télécharge les fichiers) · boucle Gym · §1 schémas · §2 vocabulaire, maison, ex. 1, thermostat · §3 ex. 2, agent classique · §4 ex. 3, robuste contre oracle · §5 agent, ex. 4, col, tableau mémoire · §6 mémoriser, ACP · §7 ex. 5, dérive · §8 synthèse.

## 1/5 · `mon_step` (section 2)

```python
dTa = (xi["UA"] * (t_ext - Ta)    # l'air fuit dehors : UA × écart
       + HAM * (Tm - Ta)          # il échange avec les murs
       + q_pompe + QINT) / CA     # + pompe + gens, ÷ capacité
dTm = HAM * (Ta - Tm) / xi["Cm"]  # les murs suivent l'air, lentement
```

**Dire :** capacité × vitesse = somme des flux, puis un pas d'Euler (déjà écrit). La masse est lente et cachée : c'est elle qui permet de préchauffer.

**Attendu :** `Exercice 1 : ✓ correct`.

## 2/5 · `ma_maj` (section 3)

```python
poids = (G * (A - P)).reshape(-1)     # retour × (action − proba)
gradient = X2.T @ poids / len(poids)  # le Xᵀ(y − ŷ)/n du cours 2
```

**Dire :** la régression logistique du cours 2 : l'étiquette est l'action jouée, et chaque exemple pèse son retour. Une action suivie d'un bon retour devient plus probable.

**Attendu :** coût de 116 à environ 12 $/jour ; poids « Ta − 21 » ≈ −2,1, « pointe maintenant » ≈ −0,4, « pointe bientôt » ≈ 0 (il ne préchauffe pas).

## 3/5 · `mes_maisons` (section 4)

```python
w = largeur * np.log(np.sqrt(FACTEUR[k]))  # demi-boîte, en log
maisons[k] = NOMINAL[k] * np.exp(          # nominal × un facteur
    rng.uniform(-w, w, n))                 # tiré uniforme en log
```

**Dire :** la domain randomization, c'est l'augmentation de données du RL. En log, « deux fois plus » et « deux fois moins » sont aussi probables.

**Attendu :** UA de 87 à 260, Cm de 1,7e7 à 5,2e7, mu de 0,92 à 2,77.

## 4/5 · `mon_grad_J` (section 5)

```python
dJ_db = (2 * s_xi**2 * erreur * c  # lire paie si la mémoire a ξ
         + 2 * b * s_eps**2)       # mais lire du bruit coûte
dJ_dc = 2 * s_xi**2 * erreur * b   # écrire paie si quelqu'un lit
```

**Dire :** en (0, 0), les deux dérivées sont nulles : lire une mémoire vide ne sert à rien, écrire une mémoire que personne ne lit non plus. C'est un col. La croix verte est l'objectif.

**Attendu :** gradient au col `(0.0, -0.0)`. Curseur « tête » à 0,5 : toutes les trajectoires vont vers la croix.

## 5/5 · `demi_vie` (section 7)

```python
estimation = x.expanding().mean()       # cumulée : n'oublie rien
estimation = x.ewm(halflife=12).mean()  # oublie : demi-vie 12 h
```

**Dire :** la moyenne cumulée met autant de temps à se corriger que son âge ; la moyenne exponentielle oublie, sa demi-vie reste constante.

**Attendu :** cumulée 11, 23, 47, 95, 191 ; exponentielle 7, 9, puis 11.

## Réponses rapides et chiffres

- **PPO (section 3) :** le policy gradient de REINFORCE, avec l'advantage Â (mieux ou moins bien que prévu, grâce au critic) à la place du retour, et un clip : L = E[min(ρ·Â, clip(ρ, 0,8, 1,2)·Â)], ρ = π_nouv / π_anc. Une mise à jour ne peut pas trop changer la politique.
- **Poids de REINFORCE :** plus il fait chaud, moins il chauffe ; il évite la pointe ; il ne préchauffe pas encore.
- **Agent classique (maison nominale) :** thermostat 7,7, thermostat qui préchauffe 6,5, agent 6,0 $/jour.
- **Écart à l'oracle (5,5 $/jour) :** thermostat +88 %, classique ailleurs +126 %, robuste +11 %, GRU sans tête +6 %, GRU avec tête +3 %.
- **Col :** une boîte plus large fait sortir plus vite ; la tête donne un gradient à l'écriture, et la lecture suit.
- **Quel agent écoute sa mémoire ?** Celui dont le coût monte le plus quand on lui donne la mémoire d'une autre maison (colonne déménager : +23 % avec tête, +11 % sans).
- **Mémoriser :** 4 maisons, +3,5 % chez lui et +15 % ailleurs ; 16 maisons, +2 % partout.
- **Demi-vie après le saut :** sans dérive à l'entraînement, il se comporte comme la moyenne cumulée (+13 % après le saut) ; avec dérive, comme la moyenne exponentielle (+7 %).
