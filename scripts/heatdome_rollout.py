"""Animate a FourCastNet3 week-3 forecast of the June 2021 Pacific Northwest heat dome.

Left: 2 m temperature anomaly from the ensemble member that came closest to the
observed peak, every 6 h from the Jun 7 initialization to the Jun 28 peak.
Right: PNW box-mean anomaly for all 24 members against ERA5.

The forecast cube comes from my (private) S2S_ExtremeWeather repo, so this
script only runs where that repo is checked out:

    uv run --with xarray --with netcdf4 --with matplotlib --with cartopy \
        scripts/heatdome_rollout.py ~/Projects/S2S_ExtremeWeather
"""

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
from matplotlib.patches import Rectangle

S2S = Path(sys.argv[1]).expanduser()
OUT = Path(__file__).resolve().parent.parent / "assets"
EVENT = "PNW_HeatDome_2021"
BOX_LAT, BOX_LON = (44.0, 50.0), (236.0, 242.0)  # same box as the S2S repo's ERA5 series
MAP_LAT, MAP_LON = (31.0, 50.0), (235.0, 258.0)
FPS = 10
HOLD_SECONDS = 2.5

THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#57606a", grid="#d0d7de",
                  member="#8c959f", best="#cf222e", obs="#1f2328", cmap="RdBu_r"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#8b949e", grid="#30363d",
                 member="#6e7681", best="#ff7b72", obs="#e6edf3",
                 cmap=LinearSegmentedColormap.from_list(
                     "dark_div", ["#d1e5f0", "#67a9cf", "#2166ac", "#21262d", "#b2182b", "#ef8a62", "#fddbc7"])),
}


def load():
    cube = xr.open_dataset(S2S / f"runs/fcn3/week3/cache/{EVENT}_cube.nc")
    clim = xr.open_dataset(S2S / "runs/models/clim_1990_2019_t2m_conus.nc")["2m_temperature"]
    t = cube.time
    clim_t = clim.sel(hour=xr.DataArray(t.dt.hour.values, dims="time"),
                      dayofyear=xr.DataArray(t.dt.dayofyear.values, dims="time"))
    anom = cube["2m_temperature"] - clim_t.assign_coords(time=t)
    anom = anom.sel(lat=slice(*MAP_LAT), lon=slice(*MAP_LON)).load()
    box = anom.sel(lat=slice(*BOX_LAT), lon=slice(*BOX_LON)).mean(["lat", "lon"])
    era5 = xr.open_dataset(S2S / f"runs/aires/{EVENT}/era5_series_6h.nc")["box_raw"]
    assert (era5.time.values == t.values).all()
    best = int(box.isel(time=slice(-4, None)).mean("time").argmax("member"))  # warmest final day
    return anom, box, era5, best, cube.attrs


def render(theme, anom, box, era5, best, attrs):
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
    lon, lat = anom.lon.values - 360, anom.lat.values
    mesh = ax.pcolormesh(lon, lat, anom.isel(member=best, time=0).values, cmap=c["cmap"],
                         vmin=-16, vmax=16, transform=ccrs.PlateCarree(), shading="auto")
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=c["fg"], linewidth=0.35, alpha=0.6)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=c["fg"], linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=c["fg"], linewidth=0.6)
    ax.add_patch(Rectangle((BOX_LON[0] - 360, BOX_LAT[0]), 6, 6, fill=False, edgecolor=c["fg"],
                           linewidth=1.2, linestyle="--", transform=ccrs.PlateCarree()))
    cax = fig.add_axes([0.07, 0.09, 0.30, 0.022])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-16, -8, 0, 8, 16])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    cb.set_label("2 m temperature anomaly (°C)", fontsize=8)

    fig.text(0.02, 0.93, "2021 Pacific Northwest heat dome, forecast 3 weeks ahead",
             fontsize=13, fontweight="bold")
    stamp = fig.text(0.02, 0.86, "", fontsize=9.5, color=c["muted"])

    ts = fig.add_axes([0.52, 0.17, 0.45, 0.62], facecolor=c["bg"])
    days = (box.time - box.time[0]).values / np.timedelta64(1, "D")
    ts.set_xlim(0, days[-1])
    ts.set_ylim(-9, 19)
    ts.set_xticks([0, 7, 14, 21], ["Jun 7\ninit", "Jun 14", "Jun 21", "Jun 28\npeak"], fontsize=8)
    ts.tick_params(axis="y", labelsize=8, length=0)
    ts.tick_params(axis="x", length=0)
    ts.set_ylabel("PNW box anomaly (°C)", fontsize=8.5)
    for s in ts.spines.values():
        s.set_visible(False)
    ts.axhline(0, color=c["grid"], linewidth=0.8)
    ts.grid(axis="y", color=c["grid"], linewidth=0.5)
    members = [ts.plot([], [], color=c["member"], linewidth=0.8, alpha=0.55)[0]
               for _ in range(box.sizes["member"])]
    mean_line, = ts.plot([], [], color=c["member"], linewidth=2, linestyle=":")
    best_line, = ts.plot([], [], color=c["best"], linewidth=2)
    obs_line, = ts.plot([], [], color=c["obs"], linewidth=2.2)
    ts.legend([obs_line, best_line, mean_line, members[0]],
              ["observed (ERA5)", f"member {best} (map)", "ensemble mean", "other members"],
              loc="upper left", fontsize=7.5, frameon=False, labelcolor=c["muted"])
    note = fig.text(0.97, 0.80, "", fontsize=8.5, color=c["fg"], ha="right")

    bx, ob, mean = box.values, era5.values, box.mean("member").values
    nt = len(days)

    def frame(i):
        k = min(i, nt - 1)
        mesh.set_array(anom.isel(member=best, time=k).values.ravel())
        stamp.set_text(f"FourCastNet3, 24 members, initialized Jun 7 · "
                       f"{np.datetime_as_string(box.time.values[k], unit='h').replace('T', ' ')}:00 UTC"
                       f" · lead {days[k]:.1f} d")
        for m, ln in enumerate(members):
            ln.set_data(days[:k + 1], bx[m, :k + 1])
        best_line.set_data(days[:k + 1], bx[best, :k + 1])
        mean_line.set_data(days[:k + 1], mean[:k + 1])
        obs_line.set_data(days[:k + 1], ob[:k + 1])
        note.set_text(f"peak: observed {ob[-1]:+.0f}°C, member {best} {bx[best, -1]:+.0f}°C, "
                      f"mean {mean[-1]:+.0f}°C" if k == nt - 1 else "")

    path = OUT / f"heatdome-{theme}.gif"
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(nt):
            frame(i)
            fig.savefig(f"{tmp}/f{i:03d}.png", facecolor=c["bg"])
        # One shared palette keeps the GIF small; tpad holds the last frame.
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/f%03d.png",
             "-filter_complex",
             f"tpad=stop_mode=clone:stop_duration={HOLD_SECONDS},split[a][b];"
             "[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=none",
             "-loop", "0", str(path)],
            check=True)
    plt.close(fig)
    print(path, f"{path.stat().st_size / 1e6:.1f} MB")


def main():
    anom, box, era5, best, attrs = load()
    assert attrs["model"] == "FourCastNet3" and attrs["lead_days"] == 21
    print(f"best member {best}: peak {box.values[best, -1]:+.1f} vs ERA5 {era5.values[-1]:+.1f}, "
          f"mean {box.mean('member').values[-1]:+.1f}")
    OUT.mkdir(exist_ok=True)
    for theme in THEMES:
        render(theme, anom, box, era5, best, attrs)


if __name__ == "__main__":
    main()
