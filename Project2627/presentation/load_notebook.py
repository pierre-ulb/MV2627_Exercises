"""
Récupère des variables calculées dans project.ipynb, sans modifier le notebook.

Les cellules de code sont exécutées dans l'ordre (graphiques désactivés,
commandes `!` et `%` ignorées) et l'exécution s'arrête dès que toutes les
variables demandées existent. Le notebook reste la seule source de calcul.

    from load_notebook import load_results
    res = load_results("psi", "freq_nat")
    psi = res["psi"]
"""

import contextlib
import io
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTEBOOK = Path(__file__).resolve().parent.parent / "project.ipynb"
_MAGIC = re.compile(r"^(\s*)[!%].*$", re.MULTILINE)


def load_results(*names, notebook=NOTEBOOK):
    nb = json.loads(Path(notebook).read_text(encoding="utf-8"))
    ns = {"__name__": "__notebook__"}
    show = plt.show
    plt.show = lambda *a, **k: None
    try:
        with contextlib.chdir(Path(notebook).parent), contextlib.redirect_stdout(io.StringIO()):
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                src = _MAGIC.sub(r"\1pass", "".join(cell["source"]))
                try:
                    exec(compile(src, f"{Path(notebook).name}[cellule {i}]", "exec"), ns)
                except Exception as e:
                    raise RuntimeError(f"Erreur dans la cellule {i} du notebook : {e!r}") from e
                finally:
                    plt.close("all")
                if all(n in ns for n in names):
                    return {n: ns[n] for n in names}
    finally:
        plt.show = show
    missing = [n for n in names if n not in ns]
    raise KeyError(f"Variables introuvables dans le notebook : {missing}")
