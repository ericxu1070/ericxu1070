"""Render assets/lorenz-{light,dark}.gif: five Lorenz-63 trajectories from nearly identical starts."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp


def lorenz(t, s, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


T, N, TAIL, FRAMES = 30.0, 6000, 400, 150
t = np.linspace(0, T, N)
starts = [np.array([1.0, 1.0, 1.0]) + 1e-5 * k for k in range(5)]
paths = [solve_ivp(lorenz, (0, T), s0, t_eval=t, rtol=1e-10, atol=1e-12).y for s0 in starts]
colors = ["#e4572e", "#f3a712", "#29bf12", "#4ea5d9", "#a663cc"]
steps = np.linspace(TAIL, N - 1, FRAMES).astype(int)

for theme, bg, ghost in [("light", "#ffffff", "#d0d0d0"), ("dark", "#0d1117", "#30363d")]:
    fig, ax = plt.subplots(figsize=(4, 1.8), dpi=100)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    ax.plot(paths[0][0], paths[0][2], color=ghost, lw=0.4)
    ax.set_xlim(-22, 22)
    ax.set_ylim(0, 52)
    lines = [ax.plot([], [], color=c, lw=1.4)[0] for c in colors]
    heads = [ax.plot([], [], "o", color=c, ms=4)[0] for c in colors]

    def draw(i):
        k = steps[i]
        for p, ln, hd in zip(paths, lines, heads):
            ln.set_data(p[0][k - TAIL:k], p[2][k - TAIL:k])
            hd.set_data([p[0][k]], [p[2][k]])
        return lines + heads

    FuncAnimation(fig, draw, frames=FRAMES, blit=True).save(
        f"assets/lorenz-{theme}.gif", writer=PillowWriter(fps=20), savefig_kwargs={"facecolor": bg})
    plt.close(fig)
