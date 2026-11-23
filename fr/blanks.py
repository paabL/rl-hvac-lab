"""Blanks for the exercises of the student notebooks: replace each ___ with your code."""


class BlankLeft(Exception):
    """An exercise still contains a ___."""


class Blank:
    """Stops the computation as soon as a ___ is used."""

    __array_ufunc__ = None

    def _stop(self, *args, **kwargs):
        raise BlankLeft

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _stop
    __truediv__ = __rtruediv__ = __pow__ = __rpow__ = __neg__ = __abs__ = _stop
    __getitem__ = __iter__ = __call__ = __float__ = __array__ = _stop

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise BlankLeft

    def __repr__(self):
        return "___"


___ = Blank()


def _status(test):
    """True (correct), False (wrong) or None (a ___ is left). Real errors in the code stay visible."""
    try:
        return bool(test())
    except BlankLeft:
        return None


def verifier(nom, test, secours=False):
    """Affiche l'état de l'exercice. Renvoie True (correct), False (à revoir) ou None (à compléter)."""
    etat = _status(test)
    vide = "à compléter" + (", la suite utilise la version du labo" if secours else "")
    print(f"{nom} : " + {True: "✓ correct", False: "✗ à revoir", None: vide}[etat])
    return etat


def check(name, test, fallback=False):
    """Prints the exercise status. Returns True (correct), False (try again) or None (to do)."""
    status = _status(test)
    todo = "to do" + (", the rest uses the lab's version" if fallback else "")
    print(f"{name}: " + {True: "✓ correct", False: "✗ try again", None: todo}[status])
    return status
