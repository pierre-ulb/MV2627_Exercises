"""
Génère les animations des modes propres du modèle de voiture à 4 DDL
pour la présentation (hors notebook).

Usage (depuis la racine du dépôt) :
    uv run python Project2627/presentation/animate_modes.py            # MP4
    uv run python Project2627/presentation/animate_modes.py --gif      # GIF en plus

Sorties dans Project2627/presentation/figures/ :
    mode_1.mp4 ... mode_4.mp4   (une animation par mode)
    modes_overview.png          (vue statique 2x2 pour une diapositive)

DDL : x1, x2 (roues), x3 (translation caisse), q3 (rotation caisse).
Convention du notebook : la caisse au-dessus de la roue 1 se déplace de
x3 - L/2*q3, celle au-dessus de la roue 2 de x3 + L/2*q3.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
from scipy import linalg

# ----------------------------------------------------------------------------
# Modèle (mêmes paramètres que project.ipynb)
# ----------------------------------------------------------------------------
M0, K0, B0 = 1500, 65000, 65
m, k, b = 10, 100000, 10
L, IG = 4, 5700

M = np.diag([m, m, M0, IG])
K = np.array([
    [k + K0, 0, -K0, L / 2 * K0],
    [0, k + K0, -K0, -L / 2 * K0],
    [-K0, -K0, 2 * K0, 0],
    [L / 2 * K0, -L / 2 * K0, 0, L**2 / 2 * K0],
])

lamb, psi = linalg.eigh(K, M)          # K, M symétriques -> valeurs réelles triées
freqs = np.sqrt(lamb) / (2 * np.pi)

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
# Géométrie du dessin (unités du dessin, pas physiques)
# ----------------------------------------------------------------------------
XW = L / 2                 # position horizontale des roues : ±2
Y_ROAD = 0.0
Y_WHEEL = 1.0              # centre des masses non suspendues
WHEEL_W, WHEEL_H = 0.7, 0.4
Y_BODY = 2.6               # centre de la caisse
BODY_W, BODY_H = 5.6, 0.7
AMP = 0.45                 # déplacement maximal dessiné


def mode_name(phi):
    x1, x2, x3, q3 = phi
    body = abs(x3) + abs(XW * q3)
    wheels = abs(x1) + abs(x2)
    if body > wheels:
        return "Pompage de la caisse" if abs(x3) > abs(XW * q3) else "Tangage de la caisse"
    return "Roues en phase" if abs(x1 + x2) > abs(x1 - x2) else "Roues en opposition"


def scaled_mode(phi):
    """Mise à l'échelle pour que le plus grand déplacement dessiné vaille AMP."""
    x1, x2, x3, q3 = phi
    pts = np.array([x1, x2, x3 - XW * q3, x3 + XW * q3])
    s = AMP / np.max(np.abs(pts))
    if pts[np.argmax(np.abs(pts))] < 0:   # signe arbitraire : premier pic vers le haut
        s = -s
    return s * phi


def spring_xy(x, y0, y1, n=6, width=0.14):
    """Ressort en zigzag vertical entre y0 et y1."""
    ys = np.linspace(y0, y1, 2 * n + 3)
    xs = np.full_like(ys, x)
    xs[2:-2] += width * np.array([(-1) ** i for i in range(2 * n - 1)])
    return xs, ys


def dashpot_xy(x, y0, y1, w=0.12):
    """Amortisseur : tige + cylindre ouvert, renvoie deux segments."""
    ym = 0.5 * (y0 + y1)
    rod = ([x, x, x - w * 0.7, x + w * 0.7], [y1, ym + 0.02, ym + 0.02, ym + 0.02])
    cyl = ([x - w, x - w, x + w, x + w, np.nan, x, x], [ym + 0.15, ym - 0.12, ym - 0.12, ym + 0.15, np.nan, ym - 0.12, y0])
    return rod, cyl


class CarDrawing:
    """Dessine la voiture sur un axe et met à jour la géométrie pour un état q."""

    def __init__(self, ax, ghost=True):
        self.ax = ax
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-0.45, 3.75)
        ax.set_aspect("equal")
        ax.axis("off")

        # route
        ax.fill_between([-3.6, 3.6], -0.45, Y_ROAD, color=ICE, zorder=0)
        ax.plot([-3.6, 3.6], [Y_ROAD, Y_ROAD], color=GREY, lw=1.2, zorder=1)

        # position d'équilibre (fantôme)
        if ghost:
            ax.add_patch(Rectangle((-BODY_W / 2, Y_BODY - BODY_H / 2), BODY_W, BODY_H, fill=False,
                                   ec=SKY, lw=1, ls="--", zorder=1))
            for xw in (-XW, XW):
                ax.add_patch(Rectangle((xw - WHEEL_W / 2, Y_WHEEL - WHEEL_H / 2), WHEEL_W, WHEEL_H,
                                       fill=False, ec=SKY, lw=1, ls="--", zorder=1))

        kw = dict(color=INK, lw=1.4, zorder=2)
        self.tire = [ax.plot([], [], **kw)[0] for _ in range(2)]
        self.susp = [ax.plot([], [], **kw)[0] for _ in range(2)]
        self.damp = [(ax.plot([], [], **kw)[0], ax.plot([], [], **kw)[0]) for _ in range(2)]
        self.wheels = [ax.add_patch(Rectangle((0, 0), WHEEL_W, WHEEL_H, fc=MID, ec=NAVY, lw=1.2, zorder=3))
                       for _ in range(2)]
        self.body = ax.add_patch(Rectangle((-BODY_W / 2, -BODY_H / 2), BODY_W, BODY_H,
                                           fc=NAVY, ec=NAVY, zorder=3))
        self.cg = ax.add_patch(Circle((0, 0), 0.09, fc="white", ec="white", zorder=4))
        self.wlab = [ax.text(0, 0, lab, ha=ha, va="center", color=GREY, fontsize=12, zorder=4)
                     for lab, ha in (("$m_1$", "right"), ("$m_2$", "left"))]
        self.blab = ax.text(0, 0, "$M_0,\\ I_G$", ha="left", va="center", color="white", fontsize=12, zorder=4)

    def update(self, q):
        x1, x2, x3, q3 = q
        yb = Y_BODY + x3
        tr = Affine2D().rotate(q3).translate(0, yb)
        self.body.set_transform(tr + self.ax.transData)
        self.cg.center = (0, yb)
        self.blab.set_position(tr.transform((0.3, 0)))
        self.blab.set_rotation(np.degrees(q3))
        for i, (xw, xi) in enumerate(((-XW, x1), (XW, x2))):
            yw = Y_WHEEL + xi
            self.wheels[i].set_xy((xw - WHEEL_W / 2, yw - WHEEL_H / 2))
            self.wlab[i].set_position((xw - 0.5 if i == 0 else xw + 0.5, yw))
            # pneu : route -> roue
            self.tire[i].set_data(*spring_xy(xw, Y_ROAD, yw - WHEEL_H / 2, n=4, width=0.12))
            # suspension : roue -> dessous de caisse (point attaché en rotation)
            ax_, ay_ = tr.transform((xw, -BODY_H / 2))
            ytop = yw + WHEEL_H / 2
            self.susp[i].set_data(*spring_xy(ax_ - 0.17, ytop, ay_, n=5, width=0.11))
            rod, cyl = dashpot_xy(ax_ + 0.2, ytop, ay_)
            self.damp[i][0].set_data(*rod)
            self.damp[i][1].set_data(*cyl)
        return [*self.tire, *self.susp, *[l for d in self.damp for l in d], *self.wheels, self.body, self.cg, self.blab, *self.wlab]


def animate_mode(i, outdir, fps=30, periods=2, frames_per_period=60, gif=False):
    phi = scaled_mode(np.real(psi[:, i]))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("white")
    car = CarDrawing(ax)
    ax.set_title(f"Mode {i + 1} — {mode_name(phi)}  ($f_{i + 1}$ = {freqs[i]:.2f} Hz)", loc="left")

    n = periods * frames_per_period
    t = np.linspace(0, periods * 2 * np.pi, n, endpoint=False)

    def frame(j):
        return car.update(phi * np.sin(t[j]))

    anim = animation.FuncAnimation(fig, frame, frames=n, interval=1000 / fps, blit=True)
    fig.tight_layout()
    mp4 = outdir / f"mode_{i + 1}.mp4"
    anim.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    print("écrit :", mp4)
    if gif:
        g = outdir / f"mode_{i + 1}.gif"
        anim.save(g, writer=animation.PillowWriter(fps=fps), dpi=90)
        print("écrit :", g)
    plt.close(fig)


def overview(outdir):
    """Figure statique 2x2 : position déformée maximale de chaque mode."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), dpi=200)
    for i, ax in enumerate(axs.flat):
        phi = scaled_mode(np.real(psi[:, i]))
        CarDrawing(ax).update(phi)
        ax.set_title(f"Mode {i + 1} — {mode_name(phi)}\n$f_{i + 1}$ = {freqs[i]:.2f} Hz", loc="left", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "svg"):
        p = outdir / f"modes_overview.{ext}"
        fig.savefig(p, facecolor="white")
        print("écrit :", p)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gif", action="store_true", help="exporter aussi en GIF")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "figures")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(freqs):
        print(f"Mode {i + 1}: {f:6.2f} Hz  {mode_name(psi[:, i])}  phi = {np.round(psi[:, i], 4)}")
    overview(args.outdir)
    for i in range(4):
        animate_mode(i, args.outdir, fps=args.fps, gif=args.gif)
