"""Animate an AI+RES (rare-event sampling) forecast of the June 2021 Pacific Northwest heat dome.

64 FourCastNet3 walkers start on June 7, 21 days before the peak. Every 3 days from day 6
to day 15, FourCastNet3 forecasts score each walker on how hot the June 22-28 week looks
from its current state (not how hot it is now). High scorers are cloned and low scorers are
killed (Diffusion
Monte Carlo, after Lancelin et al., arXiv:2510.27066).

Left: 2 m temperature anomaly along the lineage of the walker with the hottest final week.
Right: PNW box index of every walker, with killed branches fading out at each resampling.

The run (tag fcn3res1) lives in my private S2S_ExtremeWeather repo, so this only runs where
that repo is checked out:

    uv run --with xarray --with netcdf4 --with matplotlib --with cartopy \
        scripts/aires_rollout.py ~/Projects/S2S_ExtremeWeather
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

S2S = Path(sys.argv[1]).expanduser()
RUN = S2S / "runs/aires/PNW_HeatDome_2021/res/fcn3res1"
OUT = Path(__file__).resolve().parent.parent / "assets"
MAP_LAT, MAP_LON = (31.0, 50.0), (235.0, 258.0)
SEG_STEPS, N_SEG = 12, 7  # 6 h steps per 3-day segment
FPS, HOLD, END_HOLD = 10, 10, 30  # frames

THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#57606a", grid="#d0d7de", alive="#6e7781",
                  dead="#d0d7de", best="#cf222e", obs="#1f2328", window="#f0883e", clone="#bc4c00",
                  cmap="RdBu_r"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#8b949e", grid="#30363d", alive="#8b949e",
                 dead="#30363d", best="#ff7b72", obs="#e6edf3", window="#f0883e", clone="#f0883e",
                 cmap=LinearSegmentedColormap.from_list(
                     "dark_div", ["#d1e5f0", "#67a9cf", "#2166ac", "#21262d", "#b2182b", "#ef8a62", "#fddbc7"])),
}


def clim_for(times):
    clim = xr.open_dataset(S2S / "runs/models/clim_1990_2019_t2m_conus.nc")["2m_temperature"]
    t = xr.DataArray(times, dims="time")
    return clim.sel(hour=xr.DataArray(t.dt.hour.values, dims="time"),
                    dayofyear=xr.DataArray(t.dt.dayofyear.values, dims="time"))


def box_index(t2m, box_lat, box_lon):
    """cos-lat weighted box-mean anomaly, the pipeline's observable before smoothing."""
    a = t2m - clim_for(t2m.time.values).assign_coords(time=t2m.time).values
    a = a.sel(lat=slice(*box_lat), lon=slice(*box_lon))
    return a.weighted(np.cos(np.deg2rad(a.lat))).mean(["lat", "lon"]).values


def load():
    anim = json.loads((RUN / "anim.json").read_text())
    cfg = json.loads((RUN / "run.json").read_text())
    box_lat, box_lon = cfg["box_lat"], cfg["box_lon"]
    segs = anim["segments"]
    parents = {k: segs[str(k)]["parent"] for k in range(2, N_SEG + 1)}

    # raw 6 h box index for every (slot, segment); frame j of segment k is lead 3(k-1) + (j+1)/4
    raw = np.zeros((64, N_SEG, SEG_STEPS))
    for w in range(64):
        for k in range(1, N_SEG + 1):
            ds = xr.open_dataset(RUN / f"walkers/w{w:02d}/step{k:02d}/diag.nc")
            raw[w, k - 1] = box_index(ds["2m_temperature"].isel(batch=0), box_lat, box_lon)

    def lineage(j):
        slots = [j]
        for k in range(N_SEG, 1, -1):
            slots.append(parents[k][slots[-1]])
        return slots[::-1]  # slot per segment 1..7

    # A_L = mean of the raw index over leads 15..21 d inclusive (25 frames), as the pipeline
    # computes it; lead 15.0 is the last frame of segment 5
    a_l = np.array([np.concatenate([raw[s[4], 4, -1:], raw[s[5], 5], raw[s[6], 6]]).mean()
                    for s in map(lineage, range(64))])
    obs = anim["era5"]["a_l"]

    # direct sampling: the plain 24-member FCN3 ensemble from the same init
    cube = xr.open_dataset(S2S / "runs/fcn3/week3/cache/PNW_HeatDome_2021_cube.nc")
    lead = (cube.time - cube.time[0]).values / np.timedelta64(1, "D")
    win = (lead >= 15) & (lead <= 21)
    direct = np.array([box_index(cube["2m_temperature"].isel(member=m), box_lat, box_lon)[win].mean()
                       for m in range(cube.sizes["member"])])

    best = int(a_l.argmax())
    best_line = lineage(best)
    fields = []
    for k, s in enumerate(best_line, start=1):
        ds = xr.open_dataset(RUN / f"walkers/w{s:02d}/step{k:02d}/diag.nc")
        t2 = ds["2m_temperature"].isel(batch=0)
        a = t2 - clim_for(t2.time.values).assign_coords(time=t2.time).values
        fields.append(a.sel(lat=slice(*MAP_LAT), lon=slice(*MAP_LON)).load())
    fields = xr.concat(fields, "time")
    return anim, parents, best_line, fields, a_l, obs, direct


def render(theme, anim, parents, best_line, fields, a_l, obs, direct):
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": c["fg"],
                         "axes.labelcolor": c["muted"], "xtick.color": c["muted"],
                         "ytick.color": c["muted"]})
    fig = plt.figure(figsize=(9.6, 4.0), dpi=100, facecolor=c["bg"])
    proj = ccrs.LambertConformal(central_longitude=-112, central_latitude=40)
    ax = fig.add_axes([0.0, 0.17, 0.44, 0.66], projection=proj)
    ax.set_extent([MAP_LON[0] - 360, MAP_LON[1] - 360, MAP_LAT[0] + 1, MAP_LAT[1] + 0.6], ccrs.PlateCarree())
    ax.set_facecolor(c["bg"])
    ax.spines["geo"].set_visible(False)
    mesh = ax.pcolormesh(fields.lon.values - 360, fields.lat.values, fields.isel(time=0).values,
                         cmap=c["cmap"], vmin=-16, vmax=16, transform=ccrs.PlateCarree(), shading="auto")
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=c["fg"], linewidth=0.35, alpha=0.6)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=c["fg"], linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=c["fg"], linewidth=0.6)
    ax.add_patch(Rectangle((236 - 360, 44), 6, 6, fill=False, edgecolor=c["fg"], linewidth=1.2,
                           linestyle="--", transform=ccrs.PlateCarree()))
    cax = fig.add_axes([0.07, 0.09, 0.30, 0.022])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-16, -8, 0, 8, 16])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    cb.set_label("2 m temperature anomaly (°C), hottest lineage", fontsize=8)

    fig.text(0.02, 0.93, "2021 Pacific Northwest heat dome, 3 weeks out, with rare-event sampling",
             fontsize=13, fontweight="bold")
    stamp = fig.text(0.02, 0.86, "", fontsize=9.5, color=c["muted"])
    event = fig.text(0.97, 0.80, "", fontsize=8.5, color=c["fg"], ha="right")

    ts = fig.add_axes([0.52, 0.17, 0.45, 0.60], facecolor=c["bg"])
    ts.set_xlim(0, 21)
    ts.set_ylim(-9, 19)
    ts.set_xticks([0, 6, 9, 12, 15, 21], ["Jun 7\ninit", "6", "9", "12", "15", "Jun 28\npeak"], fontsize=8)
    ts.tick_params(axis="both", labelsize=8, length=0)
    ts.set_ylabel("PNW box anomaly (°C)", fontsize=8.5)
    for s in ts.spines.values():
        s.set_visible(False)
    ts.grid(axis="y", color=c["grid"], linewidth=0.5)
    ts.axvspan(15, 21, color=c["window"], alpha=0.08, linewidth=0)
    for b in (6, 9, 12, 15):
        ts.axvline(b, color=c["grid"], linewidth=0.8, linestyle=":")

    segs = anim["segments"]
    lines = {}
    for k in range(1, N_SEG + 1):
        for w in range(64):
            lines[k, w] = ts.plot([], [], color=c["alive"], linewidth=0.7, alpha=0.6)[0]
    best_artist = ts.plot([], [], color=c["best"], linewidth=2, zorder=5)[0]
    obs_artist = ts.plot([], [], color=c["obs"], linewidth=2.2, zorder=6)[0]

    era_lead = np.array(anim["era5"]["lead"])
    era_y = np.array(anim["era5"]["box"])
    seg_xy = {k: (np.array(segs[str(k)]["lead"][segs[str(k)]["seam"]:]),
                  np.array(segs[str(k)]["y"])[:, segs[str(k)]["seam"]:]) for k in range(1, N_SEG + 1)}
    best_xy = (np.concatenate([seg_xy[k][0] for k in range(1, N_SEG + 1)]),
               np.concatenate([seg_xy[k][1][best_line[k - 1]] for k in range(1, N_SEG + 1)]))
    legs = {leg["lead_end"]: leg for leg in anim["legs"]}

    # timeline: one frame per 6 h step, plus a hold at each resampling barrier
    timeline = []
    for n in range(1, SEG_STEPS * N_SEG + 1):
        timeline.append(n)
        if n % SEG_STEPS == 0 and n // SEG_STEPS in (2, 3, 4, 5):
            timeline += [n] * HOLD
    timeline += [timeline[-1]] * END_HOLD

    dead_from = {}  # (segment, slot) -> killed when this segment ends
    for k in range(2, 6):
        children = set(parents[k + 1])
        for w in range(64):
            if w not in children:
                dead_from[k, w] = True

    # resampling markers: x where a walker is killed, a dot sized by copies where one is cloned
    marks = {}
    for k in range(2, 6):
        x, y = seg_xy[k]
        dead = [w for w in range(64) if dead_from.get((k, w))]
        counts = np.bincount(parents[k + 1], minlength=64)
        cl = np.flatnonzero(counts >= 2)
        km = ts.scatter(np.full(len(dead), x[-1]), y[dead, -1], marker="x", s=14, linewidths=0.9,
                        color=c["muted"], zorder=7, visible=False)
        cm = ts.scatter(np.full(len(cl), x[-1]), y[cl, -1], s=8 * counts[cl], color=c["clone"],
                        edgecolors="none", alpha=0.9, zorder=7, visible=False)
        marks[k] = (km, cm)
    clone_h = Line2D([], [], marker="o", linestyle="", markersize=5, color=c["clone"])
    kill_h = Line2D([], [], marker="x", linestyle="", markersize=4.5, color=c["muted"])
    ts.legend([obs_artist, best_artist, lines[1, 0], clone_h, kill_h],
              ["observed (ERA5)", "hottest lineage (map)", "walkers", "cloned (size = copies)", "killed"],
              loc="upper left", fontsize=7.5, frameon=False, labelcolor=c["muted"])

    n_reach, n_direct = int((a_l >= obs).sum()), int((direct >= obs).sum())

    def frame(n):
        lead_now = n / 4
        k_now = (n - 1) // SEG_STEPS + 1
        mesh.set_array(fields.isel(time=n - 1).values.ravel())
        stamp.set_text(f"64 FourCastNet3 walkers, resampled on FCN3 forecasts of the Jun 22-28 week"
                       f" · lead {lead_now:.2f} d")
        for k in range(1, k_now + 1):
            x, y = seg_xy[k]
            m = x <= lead_now
            for w in range(64):
                ln = lines[k, w]
                ln.set_data(x[m], y[w, m])
                if dead_from.get((k, w)) and lead_now >= 3 * k:
                    ln.set_color(c["dead"])
                    ln.set_alpha(0.5)
        for k, (km, cm) in marks.items():
            km.set_visible(lead_now >= 3 * k)
            cm.set_visible(lead_now >= 3 * k)
        m = best_xy[0] <= lead_now
        best_artist.set_data(best_xy[0][m], best_xy[1][m])
        m = era_lead <= lead_now
        obs_artist.set_data(era_lead[m], era_y[m])
        barrier = 3 * (k_now if n % SEG_STEPS == 0 else k_now - 1)
        if n == SEG_STEPS * N_SEG:
            event.set_text(f"{n_reach} of 64 walkers reach the observed week (+{obs:.1f}°C); "
                           f"the plain 24-member ensemble: {n_direct}")
        elif barrier in (6, 9, 12, 15):
            leg = legs[float(barrier)]
            event.set_text(f"day {barrier}: {leg['killed']} walkers killed, the best-scored cloned")
        else:
            event.set_text("")

    path = OUT / f"aires-{theme}.gif"
    with tempfile.TemporaryDirectory() as tmp:
        for i, n in enumerate(timeline):
            frame(n)
            fig.savefig(f"{tmp}/f{i:03d}.png", facecolor=c["bg"])
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/f%03d.png",
             "-filter_complex",
             "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=none",
             "-loop", "0", str(path)],
            check=True)
    plt.close(fig)
    print(path, f"{path.stat().st_size / 1e6:.1f} MB")


def main():
    anim, parents, best_line, fields, a_l, obs, direct = load()
    print(f"observed A_L {obs:+.2f}; AI+RES {int((a_l >= obs).sum())}/64 reach (max {a_l.max():+.2f}); "
          f"direct {int((direct >= obs).sum())}/{len(direct)} (max {direct.max():+.2f}); "
          f"hottest lineage {best_line}")
    for theme in THEMES:
        render(theme, anim, parents, best_line, fields, a_l, obs, direct)


if __name__ == "__main__":
    main()
