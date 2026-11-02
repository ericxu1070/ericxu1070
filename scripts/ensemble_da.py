"""Generate the animated ensemble + data assimilation banner for the profile README.

A 20-member ensemble of Lorenz-63 runs starts from nearly identical states and
fans out (chaos). At t = T_DA, noisy observations of the truth start arriving
and a stochastic ensemble Kalman filter (EnKF) pulls the ensemble back onto
the truth. Writes assets/ensemble-da-{light,dark}.svg as SMIL-animated SVGs.

Run: uv run --with numpy scripts/ensemble_da.py
"""

from pathlib import Path

import numpy as np

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.01
T_DA = 8.0  # time when observations start
T_END = 14.0
OBS_EVERY = 0.25  # time between observations
OBS_STD = 1.5
N_ENS = 20
INIT_STD = 0.02
SEED = 56

W, H = 960, 300
PAD_L, PAD_R, PAD_T, PAD_B = 46, 20, 56, 40
DURATION = 16.0  # seconds per animation loop

THEMES = {
    "light": dict(ens="#3b82c4", truth="#1f2328", obs="#d9480f", text="#57606a",
                  axis="#d0d7de", band="#f0883e"),
    "dark": dict(ens="#58a6ff", truth="#e6edf3", obs="#ff8a4c", text="#8b949e",
                 axis="#30363d", band="#f0883e"),
}


def lorenz(s):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z], axis=-1)


def rk4(s):
    k1 = lorenz(s)
    k2 = lorenz(s + 0.5 * DT * k1)
    k3 = lorenz(s + 0.5 * DT * k2)
    k4 = lorenz(s + DT * k3)
    return s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(seed):
    rng = np.random.default_rng(seed)
    truth = np.array([1.0, 1.0, 1.0]) + rng.standard_normal(3)
    for _ in range(1500):  # spin up onto the attractor
        truth = rk4(truth)
    ens = truth + INIT_STD * rng.standard_normal((N_ENS, 3))

    n_steps = round(T_END / DT)
    obs_stride = round(OBS_EVERY / DT)
    da_start = round(T_DA / DT)
    truth_hist, ens_hist, obs = [truth.copy()], [ens.copy()], []
    for k in range(1, n_steps + 1):
        truth = rk4(truth)
        ens = rk4(ens)
        if k >= da_start and (k - da_start) % obs_stride == 0:
            # Stochastic EnKF, full state observed (H = I), perturbed observations.
            y = truth + OBS_STD * rng.standard_normal(3)
            P = np.cov(ens.T) * 1.05  # light multiplicative inflation
            K = P @ np.linalg.inv(P + OBS_STD**2 * np.eye(3))
            y_pert = y + OBS_STD * rng.standard_normal((N_ENS, 3))
            ens = ens + (y_pert - ens) @ K.T
            obs.append((k * DT, y[0]))
        truth_hist.append(truth.copy())
        ens_hist.append(ens.copy())
    t = np.arange(n_steps + 1) * DT
    return t, np.array(truth_hist), np.array(ens_hist), obs


def pick_seed():
    """Choose a seed whose free forecast spreads across both wings by T_DA."""
    best = None
    for seed in range(40):
        _, _, ens, _ = simulate(seed)
        k = round(T_DA / DT) - 1  # last step before the first analysis
        spread = ens[k, :, 0].std()
        both_wings = (ens[k, :, 0] > 0).mean()
        score = spread * min(both_wings, 1 - both_wings)
        if best is None or score > best[0]:
            best = (score, seed)
    return best[1]


def to_px(t, x):
    px = PAD_L + (t / T_END) * (W - PAD_L - PAD_R)
    py = PAD_T + (1 - (x + 22) / 44) * (H - PAD_T - PAD_B)
    return px, py


def path_d(t, x, stride=2):
    px, py = to_px(t[::stride], x[::stride])
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
    return f"M{pts.replace(' ', ' L', 1)}"


def render(theme, t, truth, ens, obs):
    c = THEMES[theme]
    font = "ui-sans-serif,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
    x_da, _ = to_px(T_DA, 0)
    plot_w = W - PAD_R
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="{font}" role="img" aria-label="Animated ensemble forecast of the Lorenz-63 '
        f'system diverging, then collapsing onto the truth once a Kalman filter assimilates observations">',
        "<defs>",
        f'<clipPath id="reveal"><rect x="0" y="0" height="{H}" width="0">'
        f'<animate attributeName="width" values="0;{plot_w};{plot_w}" keyTimes="0;0.78;1" '
        f'dur="{DURATION}s" repeatCount="indefinite"/></rect></clipPath>',
        "</defs>",
        # shaded assimilation window
        f'<rect x="{x_da:.1f}" y="{PAD_T - 6}" width="{plot_w - x_da:.1f}" '
        f'height="{H - PAD_T - PAD_B + 12}" fill="{c["band"]}" opacity="0.07"/>',
        # axes
        f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{plot_w}" y2="{H - PAD_B}" stroke="{c["axis"]}"/>',
        f'<line x1="{x_da:.1f}" y1="{PAD_T - 6}" x2="{x_da:.1f}" y2="{H - PAD_B}" '
        f'stroke="{c["text"]}" stroke-dasharray="4 4" opacity="0.6"/>',
        # labels
        f'<text x="{PAD_L}" y="22" font-size="15" font-weight="600" fill="{c["text"]}">'
        f"Forecast: errors of 0.02 grow until the ensemble disagrees</text>",
        f'<text x="{PAD_L}" y="40" font-size="12.5" fill="{c["text"]}">'
        f"20 Lorenz-63 runs, nearly identical starts</text>",
        f'<text x="{x_da + 10:.1f}" y="22" font-size="15" font-weight="600" fill="{c["text"]}">'
        f"Data assimilation: observations pull it back</text>",
        f'<text x="{x_da + 10:.1f}" y="40" font-size="12.5" fill="{c["text"]}">'
        f"ensemble Kalman filter, noisy obs every {OBS_EVERY} time units</text>",
        f'<text x="{PAD_L}" y="{H - 14}" font-size="12" fill="{c["text"]}">t = 0</text>',
        f'<text x="{plot_w}" y="{H - 14}" font-size="12" fill="{c["text"]}" text-anchor="end">'
        f"t = {T_END:g}</text>",
        f'<text x="{PAD_L - 10}" y="{(PAD_T + H - PAD_B) / 2:.1f}" font-size="12" '
        f'fill="{c["text"]}" text-anchor="end">x(t)</text>',
        # animated content
        '<g clip-path="url(#reveal)">',
        '<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.93;0.985;1" '
        f'dur="{DURATION}s" repeatCount="indefinite"/>',
        f'<g fill="none" stroke="{c["ens"]}" stroke-width="1.1" stroke-opacity="0.55" '
        'stroke-linejoin="round">',
    ]
    for m in range(ens.shape[1]):
        parts.append(f'<path d="{path_d(t, ens[:, m, 0])}"/>')
    parts.append("</g>")
    parts.append(
        f'<path d="{path_d(t, truth[:, 0])}" fill="none" stroke="{c["truth"]}" '
        'stroke-width="2" stroke-linejoin="round"/>'
    )
    for to, xo in obs:
        px, py = to_px(to, xo)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{c["obs"]}"/>')
    parts.append("</g>")
    # legend
    lx, ly = PAD_L + 150, H - 14
    parts += [
        f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 22}" y2="{ly - 4}" stroke="{c["truth"]}" stroke-width="2"/>',
        f'<text x="{lx + 28}" y="{ly}" font-size="12" fill="{c["text"]}">truth</text>',
        f'<line x1="{lx + 80}" y1="{ly - 4}" x2="{lx + 102}" y2="{ly - 4}" stroke="{c["ens"]}" stroke-width="1.5"/>',
        f'<text x="{lx + 108}" y="{ly}" font-size="12" fill="{c["text"]}">ensemble members</text>',
        f'<circle cx="{lx + 232}" cy="{ly - 4}" r="3.2" fill="{c["obs"]}"/>',
        f'<text x="{lx + 241}" y="{ly}" font-size="12" fill="{c["text"]}">observations</text>',
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def main():
    seed = SEED if SEED is not None else pick_seed()
    t, truth, ens, obs = simulate(seed)
    k = round(T_DA / DT) - 1  # last step before the first analysis
    print(f"seed={seed} spread at T_DA={ens[k, :, 0].std():.2f} "
          f"final spread={ens[-1, :, 0].std():.2f} final err={abs(ens[-1, :, 0].mean() - truth[-1, 0]):.2f}")
    out = Path(__file__).resolve().parent.parent / "assets"
    out.mkdir(exist_ok=True)
    for theme in THEMES:
        p = out / f"ensemble-da-{theme}.svg"
        p.write_text(render(theme, t, truth, ens, obs))
        print(p, f"{p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
