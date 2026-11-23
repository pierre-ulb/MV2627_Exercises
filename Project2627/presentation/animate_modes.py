"""
Animations du modèle de voiture à 4 DDL pour la présentation.

Ce script ne calcule rien : les modes propres (psi, freq_nat) sont lus dans
project.ipynb via load_notebook.py. Il ne fait que dessiner.

Usage (depuis la racine du dépôt) :
    uv run python Project2627/presentation/animate_modes.py            # MP4
    uv run python Project2627/presentation/animate_modes.py --gif      # GIF en plus

Sorties dans Project2627/presentation/figures/ :
    mode_1.mp4 ... mode_4.mp4   (une animation par mode)
    modes_overview.png / .svg   (vue statique 2x2 pour une diapositive)

DDL : x1, x2 (roues), x3 (translation caisse), q3 (rotation caisse) ;
entrées : x01, x02 (déplacements imposés par la route sous chaque roue).
Convention du notebook : la caisse au-dessus de la roue 1 se déplace de
x3 - L/2*q3, celle au-dessus de la roue 2 de x3 + L/2*q3.

Pour une réponse temporelle, utiliser animate_response(t, x, x0, fichier).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.transforms import Affine2D

# ----------------------------------------------------------------------------
# Style (palette du modèle PowerPoint)
# ----------------------------------------------------------------------------
NAVY, ROYAL, MID, SKY, ICE, INK, GREY = (
    "#1F3864", "#2E5C8A", "#4A86C5", "#9DC3E6", "#EAF1FA", "#1B2A41", "#6B7A8F")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "DejaVu Serif"],
    "axes.titlesize": 15,
    "axes.titlecolor": NAVY,
    "axes.titleweight": "bold",
})

# ----------------------------------------------------------------------------
# Géométrie du dessin (unités du dessin ; XW = L/2 = 2 m pour que la rotation
# de la caisse soit cohérente avec les déplacements)
# ----------------------------------------------------------------------------
XW = 2.0                   # position horizontale des roues : ±L/2
Y_GROUND = -0.45           # sol fixe (référence)
Y_ROAD = 0.15              # surface de la route au repos (x0 = 0)
ROAD_W, ROAD_H = 1.1, 0.12
Y_WHEEL = 1.25             # centre des masses non suspendues
WHEEL_W, WHEEL_H = 0.7, 0.4
Y_BODY = 2.85              # centre de la caisse
BODY_W, BODY_H = 5.6, 0.7
AMP = 0.4                  # déplacement maximal dessiné


def mode_name(phi):
    x1, x2, x3, q3 = phi
    body = abs(x3) + abs(XW * q3)
    wheels = abs(x1) + abs(x2)
    if body > wheels:
        return "Pompage de la caisse" if abs(x3) > abs(XW * q3) else "Tangage de la caisse"
    return "Roues en phase" if abs(x1 + x2) > abs(x1 - x2) else "Roues en opposition"


def display_scale(x, x0=None):
    """Facteur d'amplification graphique : le plus grand déplacement dessiné vaut AMP."""
    x = np.atleast_2d(np.asarray(x).T).T          # (4,) -> (4,1)
    x1, x2, x3, q3 = x
    pts = [x1, x2, x3 - XW * q3, x3 + XW * q3]
    if x0 is not None:
        pts += list(np.atleast_2d(np.asarray(x0).T).T)
    return AMP / np.max(np.abs(pts))


def spring_xy(x, y0, y1, n=5, width=0.11):
    """Ressort en zigzag vertical entre y0 et y1."""
    ys = np.linspace(y0, y1, 2 * n + 3)
    xs = np.full_like(ys, x)
    xs[2:-2] += width * np.array([(-1) ** i for i in range(2 * n - 1)])
    return xs, ys


def dashpot_xy(x, y0, y1, w=0.11):
    """Amortisseur entre y0 (bas) et y1 (haut), proportionnel à sa longueur."""
    h = y1 - y0
    yc0, yc1 = y0 + 0.25 * h, y0 + 0.65 * h       # cylindre
    yp = y0 + 0.45 * h                            # piston
    cyl = ([x, x, np.nan, x - w, x - w, x + w, x + w],
           [y0, yc0, np.nan, yc1, yc0, yc0, yc1])
    rod = ([x, x, np.nan, x - 0.7 * w, x + 0.7 * w], [y1, yp, np.nan, yp, yp])
    return rod, cyl


class CarDrawing:
    """Dessine la voiture sur un axe ; update(x, x0) la place dans un état donné."""

    def __init__(self, ax, ghost=True):
        self.ax = ax
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-0.75, 4.0)
        ax.set_aspect("equal")
        ax.axis("off")

        # sol fixe hachuré
        ax.add_patch(Rectangle((-3.6, Y_GROUND - 0.25), 7.2, 0.25, fc=ICE, ec="none", hatch="////",
                               lw=0, zorder=0))
        ax.plot([-3.6, 3.6], [Y_GROUND, Y_GROUND], color=GREY, lw=1.2, zorder=1)

        # position d'équilibre (fantôme)
        if ghost:
            dash = dict(fill=False, ec=SKY, lw=1, ls="--", zorder=1)
            ax.add_patch(Rectangle((-BODY_W / 2, Y_BODY - BODY_H / 2), BODY_W, BODY_H, **dash))
            for xw in (-XW, XW):
                ax.add_patch(Rectangle((xw - WHEEL_W / 2, Y_WHEEL - WHEEL_H / 2), WHEEL_W, WHEEL_H, **dash))
                ax.add_patch(Rectangle((xw - ROAD_W / 2, Y_ROAD - ROAD_H), ROAD_W, ROAD_H, **dash))

        kw = dict(color=INK, lw=1.4, zorder=2)
        line = lambda: ax.plot([], [], **kw)[0]
        self.tire_k = [line() for _ in range(2)]
        self.tire_b = [(line(), line()) for _ in range(2)]
        self.susp_k = [line() for _ in range(2)]
        self.susp_b = [(line(), line()) for _ in range(2)]
        self.road = [ax.add_patch(Rectangle((0, 0), ROAD_W, ROAD_H, fc=GREY, ec=INK, lw=1, zorder=3))
                     for _ in range(2)]
        self.wheels = [ax.add_patch(Rectangle((0, 0), WHEEL_W, WHEEL_H, fc=MID, ec=NAVY, lw=1.2, zorder=3))
                       for _ in range(2)]
        self.body = ax.add_patch(Rectangle((-BODY_W / 2, -BODY_H / 2), BODY_W, BODY_H,
                                           fc=NAVY, ec=NAVY, zorder=3))
        self.cg = ax.add_patch(Circle((0, 0), 0.09, fc="white", ec="white", zorder=4))

        # flèches ↕ des déplacements imposés par la route (côté intérieur)
        self.road_arrow = [ax.add_patch(FancyArrowPatch((0, 0), (0, 1), arrowstyle="<|-|>", mutation_scale=10,
                                                        color=ROYAL, lw=1.3, zorder=4)) for _ in range(2)]
        lab = dict(color=ROYAL, fontsize=12, va="center", zorder=4)
        self.road_lab = [ax.text(0, 0, "$x_{01}$", ha="left", **lab),
                         ax.text(0, 0, "$x_{02}$", ha="right", **lab)]
        self.wlab = [ax.text(0, 0, t, ha=ha, va="center", color=GREY, fontsize=12, zorder=4)
                     for t, ha in (("$m_1$", "right"), ("$m_2$", "left"))]
        self.blab = ax.text(0, 0, "$M_0,\\ I_G$", ha="left", va="center", color="white", fontsize=12, zorder=4)
        self.update(np.zeros(4))

    def update(self, x, x0=(0.0, 0.0)):
        x1, x2, x3, q3 = x
        yb = Y_BODY + x3
        tr = Affine2D().rotate(q3).translate(0, yb)
        self.body.set_transform(tr + self.ax.transData)
        self.cg.center = (0, yb)
        self.blab.set_position(tr.transform((0.3, 0)))
        self.blab.set_rotation(np.degrees(q3))

        for i, (xw, xi, x0i) in enumerate(((-XW, x1, x0[0]), (XW, x2, x0[1]))):
            side = 1 if i == 0 else -1                        # +1 : intérieur vers la droite
            yr = Y_ROAD + x0i
            yw = Y_WHEEL + xi
            self.road[i].set_xy((xw - ROAD_W / 2, yr - ROAD_H))
            self.road_arrow[i].set_positions((xw + side * 0.75, yr - 0.32), (xw + side * 0.75, yr + 0.32))
            self.road_lab[i].set_position((xw + side * 0.9, yr))
            self.wheels[i].set_xy((xw - WHEEL_W / 2, yw - WHEEL_H / 2))
            self.wlab[i].set_position((xw - side * 0.5, yw))

            # pneu : route -> roue (raideur k, amortissement b)
            ybot = yw - WHEEL_H / 2
            self.tire_k[i].set_data(*spring_xy(xw - 0.17, yr, ybot, n=4))
            rod, cyl = dashpot_xy(xw + 0.17, yr, ybot)
            self.tire_b[i][0].set_data(*rod)
            self.tire_b[i][1].set_data(*cyl)

            # suspension : roue -> caisse (raideur K0, amortissement B0)
            ax_, ay_ = tr.transform((xw, -BODY_H / 2))
            ytop = yw + WHEEL_H / 2
            self.susp_k[i].set_data(*spring_xy(ax_ - 0.17, ytop, ay_))
            rod, cyl = dashpot_xy(ax_ + 0.17, ytop, ay_)
            self.susp_b[i][0].set_data(*rod)
            self.susp_b[i][1].set_data(*cyl)

        return [*self.tire_k, *[l for d in self.tire_b for l in d], *self.susp_k,
                *[l for d in self.susp_b for l in d], *self.road, *self.road_arrow, *self.road_lab,
                *self.wheels, self.body, self.cg, self.blab, *self.wlab]


def _save(anim, path, fps, gif):
    anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    print("écrit :", path)
    if gif:
        g = path.with_suffix(".gif")
        anim.save(g, writer=animation.PillowWriter(fps=fps), dpi=90)
        print("écrit :", g)


def animate_mode(phi, freq, num, out, fps=30, periods=2, frames_per_period=60, gif=False):
    """Mode propre phi (4,) animé en phi*sin(wt) ; route immobile."""
    phi = np.real(phi) * display_scale(np.real(phi))
    if phi[np.argmax(np.abs(phi))] < 0:            # signe arbitraire : premier pic vers le haut
        phi = -phi
    fig, ax = plt.subplots(figsize=(8, 5.3), dpi=150)
    car = CarDrawing(ax)
    ax.set_title(f"Mode {num} — {mode_name(phi)}  ($f_{num}$ = {freq:.2f} Hz)", loc="left")
    t = np.linspace(0, periods * 2 * np.pi, periods * frames_per_period, endpoint=False)
    anim = animation.FuncAnimation(fig, lambda j: car.update(phi * np.sin(t[j])),
                                   frames=len(t), interval=1000 / fps, blit=True)
    fig.tight_layout()
    _save(anim, out, fps, gif)
    plt.close(fig)


def animate_response(t, x, x0, out, title="Réponse à la route", fps=30, speed=1.0, gif=False):
    """
    Réponse temporelle déjà calculée (par ex. dans le notebook).
    t  : (N,) temps [s]      x : (4, N) [x1, x2, x3, q3]      x0 : (2, N) [x01, x02]
    speed : facteur de lecture (0.5 = ralenti x2).
    Les déplacements sont amplifiés du même facteur que la route (rapport conservé).
    """
    t, x, x0 = np.asarray(t), np.real(np.asarray(x)), np.real(np.asarray(x0))
    s = display_scale(x, x0)
    frames = np.searchsorted(t, np.arange(t[0], t[-1], speed / fps))
    fig, ax = plt.subplots(figsize=(8, 5.3), dpi=150)
    car = CarDrawing(ax)
    ax.set_title(title, loc="left")
    clock = ax.text(3.5, 3.85, "", ha="right", va="top", color=GREY, fontsize=11)

    def frame(j):
        k = frames[j]
        clock.set_text(f"t = {t[k]:.2f} s")
        return [*car.update(s * x[:, k], s * x0[:, k]), clock]

    anim = animation.FuncAnimation(fig, frame, frames=len(frames), interval=1000 / fps, blit=True)
    fig.tight_layout()
    _save(anim, out, fps, gif)
    plt.close(fig)


def overview(psi, freqs, outdir):
    """Figure statique 2x2 : position déformée maximale de chaque mode."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
    for i, ax in enumerate(axs.flat):
        phi = np.real(psi[:, i]) * display_scale(np.real(psi[:, i]))
        if phi[np.argmax(np.abs(phi))] < 0:
            phi = -phi
        CarDrawing(ax).update(phi)
        ax.set_title(f"Mode {i + 1} — {mode_name(phi)}\n$f_{i + 1}$ = {freqs[i]:.2f} Hz", loc="left", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "svg"):
        p = outdir / f"modes_overview.{ext}"
        fig.savefig(p, facecolor="white")
        print("écrit :", p)
    plt.close(fig)


if __name__ == "__main__":
    from load_notebook import load_results

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gif", action="store_true", help="exporter aussi en GIF")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "figures")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    res = load_results("psi", "freq_nat")              # calculés dans project.ipynb
    psi, freqs = np.real(res["psi"]), np.abs(res["freq_nat"])
    for i, f in enumerate(freqs):
        print(f"Mode {i + 1}: {f:6.2f} Hz  {mode_name(psi[:, i])}")

    overview(psi, freqs, args.outdir)
    for i in range(psi.shape[1]):
        animate_mode(psi[:, i], freqs[i], i + 1, args.outdir / f"mode_{i + 1}.mp4", fps=args.fps, gif=args.gif)
