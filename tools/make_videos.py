"""Render the short explainer videos shown in the app (Notes & Blog > Videos).
All videos use SIMULATED data for illustration.

    pip install imageio-ffmpeg matplotlib numpy
    python tools/make_videos.py            # writes assets/videos/*.mp4
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

try:
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # system ffmpeg
    pass

OUT = Path(__file__).resolve().parents[1] / "assets" / "videos"
FPS = 20
W, H = 9.6, 5.4
BLUE, ORANGE, RED, GREEN, INK, MUTED, BG = "#1a56ff", "#f08a24", "#e5484d", "#2da44e", "#1b2433", "#5b6678", "#f0f3f8"


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def seg(f, a, b):
    """progress 0..1 of frame f inside [a,b) seconds"""
    return ease((f / FPS - a) / (b - a))


def new_fig():
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor=BG)
    return fig


def caption(fig, title, text):
    fig.text(0.04, 0.93, title, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.86, text, fontsize=12.5, color=MUTED, va="top", wrap=True)


def footer(fig):
    fig.text(0.96, 0.008, "XBCI · illustration with simulated data", fontsize=8, color=MUTED, ha="right")


def head(ax, c=(0, 0), r=1.0):
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(c[0] + r * np.cos(t), c[1] + r * np.sin(t), color=INK, lw=2)
    ax.plot([-.12, 0, .12], [r * .99, r * 1.13, r * .99], color=INK, lw=2)
    for s in (-1, 1):
        ey = np.array([-.12, -.05, .05, .12])
        ax.plot(s * (r + .06 * (1 - (ey / .12) ** 2)), ey, color=INK, lw=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.2, 1.3)


ELEC = np.array([[-.55, .1], [-.28, .1], [0, .1], [.28, .1], [.55, .1], [-.4, .45], [0, .5], [.4, .45], [-.4, -.3], [0, -.3], [.4, -.3],
                 [-.75, .35], [.75, .35], [-.75, -.1], [.75, -.1], [-.2, .75], [.2, .75], [-.2, -.6], [.2, -.6], [0, .85], [0, -.8], [0, .3]])


def writer():
    return animation.FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "23"], bitrate=-1)


def render(name, nframes, draw):
    OUT.mkdir(parents=True, exist_ok=True)
    fig = new_fig()
    path = OUT / f"{name}.mp4"
    w = writer()
    with w.saving(fig, str(path), dpi=100):
        for f in range(nframes):
            fig.clf()
            fig.set_facecolor(BG)
            draw(fig, f)
            footer(fig)
            w.grab_frame(facecolor=BG)
    plt.close(fig)
    print("wrote", path, f"{path.stat().st_size / 1e6:.1f} MB")


# ----------------------------------------------------------------------------- 1 pipeline
def video_pipeline():
    rng = np.random.RandomState(0)
    t = np.linspace(0, 6, 600)
    sig = [np.sin(2 * np.pi * (9 + i) * t + i) * (1 + .4 * np.sin(t + i)) + .5 * rng.randn(600) for i in range(6)]
    probs = np.array([.07, .71, .15, .07])
    cls = ["feet", "left hand", "right hand", "tongue"]

    def draw(fig, f):
        s = f / FPS
        if s < 4:
            fig.text(.5, .58, "From brain signal to decision", ha="center", fontsize=34, fontweight="bold", color=INK, alpha=ease(s / 1.5))
            fig.text(.5, .45, "How a brain-computer interface works — in 30 seconds", ha="center", fontsize=16, color=MUTED, alpha=ease((s - .8) / 1.5))
        elif s < 10:
            caption(fig, "1 · Record", "22 scalp electrodes measure microvolt-sized voltage changes produced by the brain.")
            ax = fig.add_axes([.25, .02, .5, .8]); head(ax)
            n = int(len(ELEC) * seg(f, 4.3, 8.5))
            ax.scatter(ELEC[:n, 0], ELEC[:n, 1], s=90, color=BLUE, zorder=3, edgecolor="white")
            ax.text(0, -1.1, "user imagines moving the LEFT hand", ha="center", fontsize=12, color=INK)
        elif s < 17:
            caption(fig, "2 · Filter & cut", "Keep 4–38 Hz (where motor rhythms live) and cut a 4-second window that starts at the cue.")
            ax = fig.add_axes([.06, .1, .88, .66])
            shift = int(60 * (s - 10))
            for i in range(6):
                ax.plot(t, sig[i] + 3.2 * (5 - i), color="#2a3a56", lw=1.1)
            p = seg(f, 12, 14.5)
            ax.axvspan(1.2, 1.2 + 4 * p, ymin=0, ymax=1, color=BLUE, alpha=.15)
            ax.axvline(1.2, color=RED, lw=2)
            ax.text(1.25, 18.2, "cue", color=RED, fontsize=11)
            ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_xlim(0, 6)
            for sp in ax.spines.values(): sp.set_visible(False)
        elif s < 23:
            caption(fig, "3 · Decode", "A model turns the 22 × 513 window into one of four commands. We compare two very different models.")
            ax = fig.add_axes([.05, .1, .9, .65]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
            a = seg(f, 17.3, 19)
            for (x0, y0, w, h, c, lab, sub) in [(0.2, 2.2, 1.6, 1.6, MUTED, "EEG window", "22 × 513"),
                                                (3.0, 3.8, 3.2, 1.4, BLUE, "EEGNet", "learned filters → deep features"),
                                                (3.0, 0.8, 3.2, 1.4, ORANGE, "CSP + LDA", "spatial filters → band power → vote"),
                                                (7.6, 2.2, 2.0, 1.6, GREEN, "prediction", "4 probabilities")]:
                ax.add_patch(plt.Rectangle((x0, y0), w, h, color=c, alpha=.12 + .5 * a * (c != MUTED) + .1 * (c == MUTED), lw=0))
                ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=c, lw=2))
                ax.text(x0 + w / 2, y0 + h / 2 + .15, lab, ha="center", fontsize=14, fontweight="bold", color=INK)
                ax.text(x0 + w / 2, y0 + h / 2 - .3, sub, ha="center", fontsize=9, color=MUTED)
            for (xa, ya, xb, yb) in [(1.85, 3.0, 2.95, 4.5), (1.85, 3.0, 2.95, 1.5), (6.25, 4.5, 7.55, 3.2), (6.25, 1.5, 7.55, 2.8)]:
                ax.annotate("", xy=(xa + (xb - xa) * a, ya + (yb - ya) * a), xytext=(xa, ya), arrowprops=dict(arrowstyle="->", lw=2, color=INK))
        elif s < 28:
            caption(fig, "4 · Predict", "Both models output probabilities — here the model says 'left hand' with 71 %.")
            ax = fig.add_axes([.2, .12, .65, .62])
            p = seg(f, 23.3, 25.5)
            ax.barh(cls, probs * p, color=[GREEN if i == 1 else "#b8c2d6" for i in range(4)])
            ax.invert_yaxis(); ax.set_xlim(0, 1)
            for i, v in enumerate(probs):
                ax.text(v * p + .01, i, f"{v * p:.0%}", va="center", fontsize=13)
            ax.tick_params(labelsize=13)
            for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        else:
            caption(fig, "5 · Explain", "But WHY? Which electrodes, which moments and which rhythms pushed the model to 'left hand'?")
            ax = fig.add_axes([.05, .05, .4, .78]); head(ax)
            xx, yy = np.meshgrid(np.linspace(-1, 1, 80), np.linspace(-1, 1, 80))
            p = seg(f, 28.3, 31)
            z = np.exp(-((xx - .5) ** 2 + (yy - .1) ** 2) / .06) - .5 * np.exp(-((xx + .5) ** 2 + (yy - .1) ** 2) / .08)
            z = np.where(xx ** 2 + yy ** 2 <= 1, z * p, np.nan)
            ax.imshow(z, extent=(-1, 1, -1, 1), origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, zorder=0)
            ax.text(.5, .1, "C4", ha="center", fontsize=11, fontweight="bold")
            ax2 = fig.add_axes([.55, .2, .4, .45])
            ax2.bar(["theta", "mu", "beta"], np.array([.1, .9, .5]) * p, color=BLUE)
            ax2.set_ylim(0, 1); ax2.set_title("rhythm importance", fontsize=12)
            for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
            fig.text(.5, .08, "This app answers that — step by step, with the models' real weights.", ha="center", fontsize=13, color=INK, alpha=ease((s - 30) / 1.5))
    return draw, int(33 * FPS)


# ----------------------------------------------------------------------------- 2 ERD
def video_erd():
    fs = 100
    t = np.arange(0, 6, 1 / fs)
    rng = np.random.RandomState(1)
    carrier = np.sin(2 * np.pi * 10.5 * t) + .25 * rng.randn(len(t))
    env_c3 = np.ones_like(t)
    env_c4 = 1 - .65 * ease((t - 2.2) / .6) * (t < 5.2) + .5 * ease((t - 5.2) / .5) * (t >= 5.2) * 0
    env_c4 = np.where(t < 2.2, 1, np.where(t < 5.2, 1 - .65 * ease((t - 2.2) / .6), .35 + .65 * ease((t - 5.2) / .8)))
    x3, x4 = env_c3 * carrier * 1.0, env_c4 * carrier * 1.0

    def power(x):
        k = 50
        return np.convolve(x ** 2, np.ones(k) / k, mode="same")
    p3, p4 = power(x3), power(x4)

    def draw(fig, f):
        s = f / FPS
        cur = min(s / 16.0 * 6.0, 6.0)
        i = int(cur * fs)
        caption(fig, "Motor imagery changes brain rhythms", "Imagining a LEFT-hand movement weakens the 8–13 Hz mu rhythm over the RIGHT motor cortex (electrode C4): ERD.")
        ax = fig.add_axes([.14, .48, .52, .30]); ax2 = fig.add_axes([.14, .17, .52, .24])
        ax.plot(t[:i], x3[:i] + 2.4, color=BLUE, lw=1); ax.plot(t[:i], x4[:i], color=RED, lw=1)
        ax.set_xlim(0, 6); ax.set_ylim(-1.8, 4.2); ax.set_yticks([0, 2.4]); ax.set_yticklabels(["C4 (right)", "C3 (left)"]); ax.set_xticklabels([])
        ax2.plot(t[:i], p3[:i], color=BLUE, lw=2, label="C3 power"); ax2.plot(t[:i], p4[:i], color=RED, lw=2, label="C4 power")
        ax2.set_xlim(0, 6); ax2.set_ylim(0, 1.3); ax2.set_xlabel("time (s)"); ax2.legend(loc="upper right", fontsize=9, frameon=False)
        for a in (ax, ax2):
            a.axvspan(2.2, 5.2, color=GREEN, alpha=.08)
            for sp in ["top", "right"]: a.spines[sp].set_visible(False)
        ax2.text(3.7, 1.18, "imagery", ha="center", color=GREEN, fontsize=10)
        axh = fig.add_axes([.70, .2, .28, .55]); head(axh)
        d = (p3[i] if i < len(p3) else p3[-1], p4[i] if i < len(p4) else p4[-1])
        for (cx, val, lab, col) in [(-.5, d[0], "C3", BLUE), (.5, d[1], "C4", RED)]:
            axh.add_patch(plt.Circle((cx, .1), .1 + .3 * val, color=col, alpha=.35))
            axh.text(cx, .1, lab, ha="center", va="center", fontsize=10, fontweight="bold")
        fig.text(.84, .13, "circle size = rhythm power", ha="center", fontsize=9, color=MUTED)
        if s > 12:
            fig.text(.5, .05, "Right hand → the mirror image over C3. Feet → near the mid-line (Cz).", ha="center", fontsize=11, color=INK, alpha=ease((s - 12) / 1.5))
    return draw, int(18 * FPS)


# ----------------------------------------------------------------------------- 3 CSP
def video_csp():
    rng = np.random.RandomState(3)
    n = 250
    R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    A = (R(0.6) @ np.diag([2.0, .45]) @ rng.randn(2, n)).T
    B = (R(-0.6) @ np.diag([2.0, .45]) @ rng.randn(2, n)).T
    thetas = np.linspace(0, np.pi, 181)
    ratio = []
    for th in thetas:
        w = np.array([np.cos(th), np.sin(th)])
        va, vb = np.var(A @ w), np.var(B @ w)
        ratio.append(va / (va + vb))
    ratio = np.array(ratio)
    best = thetas[np.argmax(ratio)]

    def draw(fig, f):
        s = f / FPS
        caption(fig, "CSP: find the viewing direction that separates classes", "Two electrodes, two classes. A spatial filter is a direction; we keep the one where class A is LOUD and class B is QUIET.")
        ax = fig.add_axes([.08, .17, .40, .58])
        ax.scatter(A[:, 0], A[:, 1], s=8, color=BLUE, alpha=.5, label="class A"); ax.scatter(B[:, 0], B[:, 1], s=8, color=ORANGE, alpha=.5, label="class B")
        p = seg(f, 2, 12)
        th = best if s > 12 else thetas[min(int(p * 180), 180)] if s > 2 else 0.0
        if s > 12:
            th = best
        w = np.array([np.cos(th), np.sin(th)])
        ax.plot([-w[0] * 7, w[0] * 7], [-w[1] * 7, w[1] * 7], color=INK, lw=2.5)
        ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.set_aspect("equal"); ax.set_xlabel("electrode 1"); ax.set_ylabel("electrode 2"); ax.legend(frameon=False, loc="upper left")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        va, vb = np.var(A @ w), np.var(B @ w)
        ax2 = fig.add_axes([.58, .52, .38, .22]); ax2.bar(["class A", "class B"], [va, vb], color=[BLUE, ORANGE]); ax2.set_ylim(0, 9); ax2.set_title("variance (power) along the line", fontsize=11)
        ax3 = fig.add_axes([.58, .19, .38, .22]); i = int(np.clip(np.searchsorted(thetas, th), 0, 180))
        ax3.plot(thetas[:i + 1], ratio[:i + 1], color=INK, lw=2); ax3.set_xlim(0, np.pi); ax3.set_ylim(0, 1); ax3.set_xlabel("line angle"); ax3.set_title("share of power from class A", fontsize=11)
        for a in (ax2, ax3):
            for sp in ["top", "right"]: a.spines[sp].set_visible(False)
        if s > 12.5:
            fig.text(.5, .05, "Best direction found → its log-variance becomes a feature. CSP+LDA then takes a weighted vote of 8 such features.", ha="center", fontsize=11, color=INK, alpha=ease((s - 12.5) / 1.2))
    return draw, int(16 * FPS)


# ----------------------------------------------------------------------------- 4 explainability
def video_xai():
    fs = 100
    t = np.arange(0, 4, 1 / fs)
    rng = np.random.RandomState(5)
    burst = np.exp(-((t - 2.2) ** 2) / .25) * np.sin(2 * np.pi * 10.5 * t)
    x = .35 * rng.randn(len(t)) + burst
    bands = ["theta", "mu", "beta"]
    drops = [.03, .62, .21]
    wins = np.linspace(0, 4, 9)
    tf = np.outer([.02, 1.0, .3], np.exp(-((wins[:-1] + .25 - 2.2) ** 2) / .6))

    def draw(fig, f):
        s = f / FPS
        caption(fig, "How we explain a decision", "Idea: remove one rhythm in one time window and see how much the model's confidence falls. A big fall = the model relied on it.")
        ax = fig.add_axes([.06, .5, .56, .27])
        ax.plot(t, x, color="#2a3a56", lw=1)
        k = int(seg(f, 2, 12) * 7.999)
        if s > 2:
            ax.axvspan(wins[k], wins[k + 1], color=RED, alpha=.18)
            ax.text(wins[k] + .25, 1.35, "removed", ha="center", color=RED, fontsize=9)
        ax.set_ylim(-1.5, 1.6); ax.set_yticks([])
        for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
        axp = fig.add_axes([.7, .5, .26, .27])
        base = .71
        cur = base - (tf[1, k] * .6 if s > 2 else 0)
        axp.bar(["confidence"], [cur], color=GREEN if cur > .5 else RED); axp.set_ylim(0, 1); axp.set_title("P(left hand)", fontsize=11)
        for sp in ["top", "right"]: axp.spines[sp].set_visible(False)
        ax2 = fig.add_axes([.1, .19, .5, .22])
        z = np.full_like(tf, np.nan); z[:, :k + 1] = tf[:, :k + 1] if s > 2 else np.nan
        ax2.imshow(z, aspect="auto", cmap="magma", vmin=0, vmax=1, extent=(0, 4, -.5, 2.5), origin="lower")
        ax2.set_yticks([0, 1, 2]); ax2.set_yticklabels(bands); ax2.set_xlabel("time window (s)"); ax2.set_title("importance map builds up window by window", fontsize=11)
        if s > 12.5:
            fig.text(.8, .25, "Result:\nthe 10 Hz (mu)\nrhythm around 2 s\nmattered most", ha="center", fontsize=13, color=INK, fontweight="bold", alpha=ease((s - 12.5) / 1.2))
            fig.text(.5, .05, "Together with Integrated Gradients (electrode-by-time evidence) this tells us WHERE, WHEN and WHICH RHYTHM.", ha="center", fontsize=11, color=INK, alpha=ease((s - 13.5) / 1.2))
    return draw, int(17 * FPS)


VIDEOS = {"01_bci_pipeline": video_pipeline, "02_erd_motor_imagery": video_erd, "03_csp_spatial_filtering": video_csp, "04_explainability": video_xai}

if __name__ == "__main__":
    which = sys.argv[1:] or list(VIDEOS)
    for k in which:
        draw, n = VIDEOS[k]()
        render(k, n, draw)
