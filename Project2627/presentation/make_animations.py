"""
Animations du modèle de voiture à 4 DDL pour la présentation.

Ce script ne calcule rien : toutes les grandeurs physiques sont lues dans
project.ipynb (via load_notebook.py). Il ne fait que dessiner.

Usage (depuis la racine du dépôt) :
    uv run python Project2627/presentation/make_animations.py
    uv run python Project2627/presentation/make_animations.py modes
    uv run python Project2627/presentation/make_animations.py temps --speed 0.5
  (ffmpeg doit être installé ; --etiquettes ajoute les symboles m1, m2, M0, x01, x02)

Aucun texte n'est écrit sur les animations : les informations sont dans les noms.
Sorties dans Project2627/presentation/figures/ (chaque dossier est vidé puis régénéré) :
    1_modes/                mode1_pompage-de-la-caisse_f1.15Hz.mp4, ..., modes_apercu_*.png/.svg
    2_transmissibilite/     X3-sur-X01_X01=X02_f1.10Hz_gain11.8_route-dessinee-x6.mp4, ...
    3_reponse_temporelle/   reponse-route_v05ms_temps-reel_deplacements-x7_duree40s.mp4, ...

Variables lues dans le notebook :
    modes             psi, freq_nat
    transmissibilité  f, X_sym, X_anti, freq_nat
    temporel          X_t, t, N, v, x_01, x_02, x, w_road

DDL : x1, x2 (roues), x3 (translation caisse), q3 (rotation caisse) ;
entrées : x01, x02 (déplacements imposés par la route sous chaque roue).
Convention du notebook : la caisse au-dessus de la roue 1 se déplace de
x3 - L/2*q3, celle au-dessus de la roue 2 de x3 + L/2*q3 ; la roue 2 est
devant (x_02 = w_road décalé de L).
"""

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
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
LABELS = False             # aucun texte sur les animations (option --etiquettes pour les symboles)


def mode_name(phi):
    x1, x2, x3, q3 = phi
    body = abs(x3) + abs(XW * q3)
    wheels = abs(x1) + abs(x2)
    if body > wheels:
        return "Pompage de la caisse" if abs(x3) > abs(XW * q3) else "Tangage de la caisse"
    return "Roues en phase" if abs(x1 + x2) > abs(x1 - x2) else "Roues en opposition"


def display_scale(x, x0=None):
    """Facteur d'amplification graphique : le plus grand déplacement dessiné vaut AMP."""
    x = np.atleast_2d(np.asarray(x).T).T          # (4,) -> (4,1) ; complexe accepté
    x1, x2, x3, q3 = x
    pts = [x1, x2, x3 - XW * q3, x3 + XW * q3]
    if x0 is not None:
        pts += list(np.atleast_2d(np.asarray(x0).T).T)
    return AMP / np.max(np.abs(np.array(pts)))


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

    def __init__(self, ax, ghost=True, labels=False):
        self.ax = ax
        self.labels = labels
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
        # étiquettes des symboles (désactivées par défaut : aucun texte sur les animations)
        self.road_lab, self.wlab, self.blab = [], [], None
        if labels:
            lab = dict(color=ROYAL, fontsize=12, va="center", zorder=4)
            self.road_lab = [ax.text(0, 0, "$x_{01}$", ha="left", **lab),
                             ax.text(0, 0, "$x_{02}$", ha="right", **lab)]
            self.wlab = [ax.text(0, 0, t, ha=ha, va="center", color=GREY, fontsize=12, zorder=4)
                         for t, ha in (("$m_1$", "right"), ("$m_2$", "left"))]
            self.blab = ax.text(0, 0, "$M_0,\\ I_G$", ha="left", va="center", color="white", fontsize=12,
                                zorder=4)
        self.update(np.zeros(4))

    def update(self, x, x0=(0.0, 0.0)):
        x1, x2, x3, q3 = x
        yb = Y_BODY + x3
        tr = Affine2D().rotate(q3).translate(0, yb)
        self.body.set_transform(tr + self.ax.transData)
        self.cg.center = (0, yb)
        if self.blab is not None:
            self.blab.set_position(tr.transform((0.3, 0)))
            self.blab.set_rotation(np.degrees(q3))

        for i, (xw, xi, x0i) in enumerate(((-XW, x1, x0[0]), (XW, x2, x0[1]))):
            side = 1 if i == 0 else -1                        # +1 : intérieur vers la droite
            yr = Y_ROAD + x0i
            yw = Y_WHEEL + xi
            self.road[i].set_xy((xw - ROAD_W / 2, yr - ROAD_H))
            self.road_arrow[i].set_positions((xw + side * 0.75, yr - 0.32), (xw + side * 0.75, yr + 0.32))
            if self.labels:
                self.road_lab[i].set_position((xw + side * 0.9, yr))
                self.wlab[i].set_position((xw - side * 0.5, yw))
            self.wheels[i].set_xy((xw - WHEEL_W / 2, yw - WHEEL_H / 2))

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
                *self.wheels, self.body, self.cg, *([self.blab] if self.blab else []), *self.wlab]


def slug(text):
    """Texte -> morceau de nom de fichier sans accents ni espaces."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "-".join(t.lower().split())


def clean_dir(d):
    """Vide un dossier de sortie des anciennes animations (les noms contiennent des valeurs)."""
    d.mkdir(parents=True, exist_ok=True)
    for old in [*d.glob("*.mp4"), *d.glob("*.gif"), *d.glob("*.png"), *d.glob("*.svg")]:
        old.unlink()


def _save(anim, path, fps, gif):
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    print("écrit :", path)
    if gif:
        g = path.with_suffix(".gif")
        anim.save(g, writer=animation.PillowWriter(fps=fps), dpi=90)
        print("écrit :", g)


def _upward(phi):
    """Signe arbitraire d'un mode : le plus grand déplacement part vers le haut."""
    return -phi if phi[np.argmax(np.abs(phi))] < 0 else phi


# ----------------------------------------------------------------------------
# 1. Modes propres
# ----------------------------------------------------------------------------
def animate_mode(phi, freq, num, out, fps=30, periods=2, frames_per_period=60, gif=False):
    """Mode propre phi (4,) animé en phi*sin(wt) ; route immobile. Boucle parfaite."""
    phi = _upward(np.real(phi) * display_scale(np.real(phi)))
    fig, ax = plt.subplots(figsize=(8, 5.3), dpi=150)
    car = CarDrawing(ax, labels=LABELS)
    th = np.linspace(0, periods * 2 * np.pi, periods * frames_per_period, endpoint=False)
    anim = animation.FuncAnimation(fig, lambda j: car.update(phi * np.sin(th[j])),
                                   frames=len(th), interval=1000 / fps, blit=True)
    fig.tight_layout()
    _save(anim, out, fps, gif)
    plt.close(fig)


def overview(psi, freqs, outdir):
    """Figure statique 2x2 : position déformée maximale de chaque mode."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
    for i, ax in enumerate(axs.flat):
        phi = _upward(np.real(psi[:, i]) * display_scale(np.real(psi[:, i])))
        CarDrawing(ax, labels=LABELS).update(phi)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fs = "_".join(f"{fi:.2f}" for fi in freqs)
    for ext in ("png", "svg"):
        p = outdir / f"modes_apercu_1-2-3-4_f{fs}Hz.{ext}"
        fig.savefig(p, facecolor="white")
        print("écrit :", p)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2. Réponse harmonique (transmissibilité) à une fréquence donnée
# ----------------------------------------------------------------------------
ROAD_AMP = 0.22            # amplitude dessinée de la route en harmonique


def animate_harmonic(X, X0, out, fps=30, periods=2, frames_per_period=60, gif=False):
    """
    X  : (4,) réponse complexe [X1, X2, X3, Q3] pour l'entrée X0 (2,) complexe.
    Animation de Re(X e^{iwt}) et Re(X0 e^{iwt}) sur un nombre entier de périodes (boucle parfaite).
    La route et la voiture ont chacune leur facteur d'amplification graphique :
    renvoie le rapport (route / voiture), à indiquer dans le nom du fichier.
    """
    X, X0 = np.asarray(X), np.asarray(X0)
    s_car = display_scale(X)
    s_road = ROAD_AMP / np.max(np.abs(X0))
    fig, ax = plt.subplots(figsize=(8, 5.3), dpi=150)
    car = CarDrawing(ax, labels=LABELS)
    th = np.linspace(0, periods * 2 * np.pi, periods * frames_per_period, endpoint=False)

    def frame(j):
        e = np.exp(1j * th[j])
        return car.update(s_car * np.real(X * e), s_road * np.real(X0 * e))

    anim = animation.FuncAnimation(fig, frame, frames=len(th), interval=1000 / fps, blit=True)
    fig.tight_layout()
    _save(anim, out, fps, gif)
    plt.close(fig)
    return s_road / s_car


# ----------------------------------------------------------------------------
# 3. Réponse temporelle sur le profil de route
# ----------------------------------------------------------------------------
def animate_response(t, x, x0, out, fps=30, speed=1.0, gif=False,
                     pos=None, road_x=None, road_w=None):
    """
    Réponse temporelle déjà calculée (dans le notebook).
    t  : (N,) temps [s]      x : (4, N) [x1, x2, x3, q3]      x0 : (2, N) [x01, x02]
    speed : facteur de lecture (0.5 = ralenti x2).
    Voiture et route amplifiées du même facteur (rapport d'amplitude conservé).
    Optionnel (panneau du bas) : pos (N,) position de la roue 1 sur la route [m],
    road_x, road_w : profil de route w_road(x) [m].
    """
    t, x, x0 = np.asarray(t), np.real(np.asarray(x)), np.real(np.asarray(x0))
    s = display_scale(x, x0)
    frames = np.searchsorted(t, np.arange(t[0], t[-1], speed / fps))
    with_road = pos is not None

    if with_road:
        fig = plt.figure(figsize=(8, 7), dpi=150)
        gs = GridSpec(2, 1, height_ratios=[4.75, 1.6], hspace=0.08, figure=fig)
        ax = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(8, 5.3), dpi=150)
    car = CarDrawing(ax, labels=LABELS)
    artists = []

    if with_road:
        road_x, road_w = np.asarray(road_x), 1e3 * np.asarray(road_w)       # mm
        pad = 0.1 * np.ptp(road_w)
        axr.set_xlim(*ax.get_xlim())
        axr.set_ylim(road_w.min() - pad, road_w.max() + pad)
        axr.fill_between(axr.get_xlim(), road_w.min() - pad, road_w.max() + pad, color=ICE, zorder=0)
        for side in (-XW, XW):
            axr.axvline(side, color=SKY, lw=1, ls="--", zorder=1)
        for sp in ("top", "right"):
            axr.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axr.spines[sp].set_color(GREY)
        axr.tick_params(colors=GREY, labelsize=9)
        if LABELS:
            axr.set_ylabel("$w_{road}$ [mm]", color=GREY, fontsize=10)
            axr.set_xlabel("Position par rapport au centre de la voiture [m]  —  sens de marche →",
                           color=GREY, fontsize=10)
        else:
            axr.tick_params(labelleft=False, labelbottom=False)
        prof, = axr.plot([], [], color=NAVY, lw=1.4, zorder=2)
        contact, = axr.plot([], [], "o", ms=6, color=ROYAL, zorder=3)
        artists += [prof, contact]

    def frame(j):
        k = frames[j]
        out_art = car.update(s * x[:, k], s * x0[:, k])
        if with_road:
            centre = pos[k] + XW                               # roue 1 en -L/2, roue 2 en +L/2
            i0, i1 = np.searchsorted(road_x, [centre - 4, centre + 4])
            prof.set_data(road_x[i0:i1] - centre, road_w[i0:i1])
            contact.set_data([-XW, XW], 1e3 * x0[:, k])
        return [*out_art, *artists]

    anim = animation.FuncAnimation(fig, frame, frames=len(frames), interval=1000 / fps, blit=True)
    if not with_road:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08 if LABELS else 0.03)
        fig.canvas.draw()                      # aligne le profil sous la voiture (même échelle en x)
        p, q = ax.get_position(), axr.get_position()
        axr.set_position([p.x0, q.y0, p.width, q.height])
    _save(anim, out, fps, gif)
    plt.close(fig)
    return s


# ----------------------------------------------------------------------------
# Programme principal : lit les résultats du notebook et génère les vidéos
# ----------------------------------------------------------------------------
def make_modes(outdir, fps, gif):
    r = load_results("psi", "freq_nat")
    psi, freqs = np.real(r["psi"]), np.abs(r["freq_nat"])
    d = outdir / "1_modes"
    clean_dir(d)
    overview(psi, freqs, d)
    for i in range(psi.shape[1]):
        name = f"mode{i + 1}_{slug(mode_name(psi[:, i]))}_f{freqs[i]:.2f}Hz.mp4"
        animate_mode(psi[:, i], freqs[i], i + 1, d / name, fps=fps, gif=gif)


def make_transmissibility(outdir, fps, gif):
    r = load_results("f", "X_sym", "X_anti", "freq_nat")
    f, freqs = np.asarray(r["f"]), np.abs(r["freq_nat"])
    d = outdir / "2_transmissibilite"
    clean_dir(d)
    cases = [
        # (réponse, entrée [X01, X02], mode de la 1re résonance, préfixe, gain affiché dans le nom)
        (r["X_sym"], [1, 1], 0, "X3-sur-X01_X01=X02", lambda X: f"gain{abs(X[2]):.3g}"),
        (r["X_anti"], [1, -1], 1, "q3-sur-X01_X01=-X02", lambda X: f"gain{abs(X[3]):.3g}rad-par-m"),
    ]
    for Xall, X0, mode, prefix, gain in cases:
        # fréquence de la grille f la plus proche de la résonance, par en dessous
        n = max(np.searchsorted(f, freqs[mode], side="right") - 1, 0)
        X = Xall[:, n]
        tmp = d / "_en_cours.mp4"
        ratio = animate_harmonic(X, np.array(X0, dtype=complex), tmp, fps=fps, gif=gif)
        name = f"{prefix}_f{f[n]:.2f}Hz_{gain(X)}_route-dessinee-x{ratio:.0f}"
        for ext in (".mp4", ".gif"):
            if tmp.with_suffix(ext).exists():
                tmp.with_suffix(ext).replace(d / (name + ext))
                print("renommé :", d / (name + ext))


def make_time(outdir, fps, gif, speed):
    r = load_results("X_t", "t", "N", "v", "x_01", "x_02", "x", "w_road")
    N, v, x_road, w_road = r["N"], np.ravel(r["v"]), np.asarray(r["x"]), np.asarray(r["w_road"])
    d = outdir / "3_reponse_temporelle"
    clean_dir(d)
    x0 = np.vstack([r["x_01"], r["x_02"]])
    lecture = "temps-reel" if speed == 1 else (f"ralenti-x{1 / speed:g}" if speed < 1 else f"accelere-x{speed:g}")
    for s, vs in enumerate(v):
        ts = r["t"][s, :N]
        tmp = d / "_en_cours.mp4"
        amp = animate_response(ts, r["X_t"][s], x0, tmp, fps=fps, speed=speed, gif=gif,
                               pos=x_road[:N], road_x=x_road, road_w=w_road)
        name = f"reponse-route_v{vs:02.0f}ms_{lecture}_deplacements-x{amp:.0f}_duree{(ts[-1] - ts[0]) / speed:.0f}s"
        for ext in (".mp4", ".gif"):
            if tmp.with_suffix(ext).exists():
                tmp.with_suffix(ext).replace(d / (name + ext))
                print("renommé :", d / (name + ext))


if __name__ == "__main__":
    from load_notebook import load_results

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parties", nargs="*", choices=["modes", "transmissibilite", "temps"],
                    help="parties à générer (par défaut : toutes)")
    ap.add_argument("--gif", action="store_true", help="exporter aussi en GIF")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0, help="vitesse de lecture des réponses temporelles")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "figures")
    ap.add_argument("--etiquettes", action="store_true",
                    help="ajouter les symboles (m1, m2, M0, x01, x02) et les axes du profil de route")
    a = ap.parse_args()
    LABELS = a.etiquettes
    todo = a.parties or ["modes", "transmissibilite", "temps"]

    if "modes" in todo:
        make_modes(a.outdir, a.fps, a.gif)
    if "transmissibilite" in todo:
        make_transmissibility(a.outdir, a.fps, a.gif)
    if "temps" in todo:
        make_time(a.outdir, a.fps, a.gif, a.speed)
