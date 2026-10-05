#!/usr/bin/env python3
# estimate_vs_resonance_like_paper.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from pyproj import Transformer

sys.path.append("/home/lea/Desktop/code/thesis/das_datas/lib/")
import map.geo_functions as gf
import vs_functions as func


# ============================================================
# PARAMS
# ============================================================

SAVE_DIR = "/home/lea/Desktop/code/thesis/das_datas/resonance_model"
os.makedirs(SAVE_DIR, exist_ok=True)

BATHY_PATH = "/home/lea/Desktop/code/thesis/das_datas/bathy_svalbard_subset.nc"
CABLE_CSV = "/home/lea/Desktop/code/thesis/das_datas/map/outer_cable_positions.csv"

DAS_FOLDER = "/media/lea/Expansion/DAS/20250816/dphi"

DX = 4.08
CHAN_START = 0
CHAN_END = 5001

FS_IN = 2000
FS_OUT = 40

FMIN = 0.3
FMAX = 19.0

# fenêtre temporelle type événement
T_START = [4,54,42]
T_END = [5,4,42]

# hypothèse case 1 du papier
CS_AVG = 200.0  # m/s

# picking
SMOOTH_FREQ_SIGMA = 1.0
MIN_PEAK_PROMINENCE = 0.08
MIN_PEAK_DISTANCE_HZ = 0.4

# ============================================================
# LOAD BATHY ALONG CABLE
# ============================================================

ds = xr.open_dataset(BATHY_PATH)
bathy = ds["z"]

Lat, Lon, new_Lat, new_Lon = gf.create_cable_map(
    CABLE_CSV,
    CHAN_END + 1,
    1.02 * 4,
    start_offset_m=0,
)

transformer_bathy = Transformer.from_crs("EPSG:4326", "EPSG:3996", always_xy=True)

cable_lon = np.asarray(new_Lon)
cable_lat = np.asarray(new_Lat)

cable_x, cable_y = transformer_bathy.transform(cable_lon, cable_lat)

bathy_on_cable = bathy.interp(
    x=xr.DataArray(cable_x, dims="channel"),
    y=xr.DataArray(cable_y, dims="channel"),
)

bathy_values = bathy_on_cable.values
dist_bathy = np.arange(len(bathy_values)) * DX


# ============================================================
# COMPUTE F-X POWER SPECTRUM
# ============================================================

""" res_fx = func.das_fx_spectrum_chunked(
    t_start=T_START,
    t_end=T_END,
    path_folder=DAS_FOLDER,
    chan_start_idx=CHAN_START,
    chan_end_idx=CHAN_END,
    chunk_channels=100,
    fs_in=FS_IN,
    fs_out=FS_OUT,
    fmin=FMIN,
    fmax=FMAX,
    nperseg_s=60,
    noverlap_s=30,
    remove_common=False,
    robust_normalize=True,
    plot=True,
)

f, dist_m, Z = func.extract_fx_arrays(res_fx)

# Z attendu : shape = [n_freq, n_x] ou [n_x, n_freq]
if Z.shape[0] != len(f):
    Z = Z.T

# conversion en amplitude normalisée
Z = np.asarray(Z, dtype=float)
Z = np.nan_to_num(Z, nan=np.nanmedian(Z))

from scipy.ndimage import gaussian_filter

Z = gaussian_filter(Z, sigma=(1.0, 20.0))  # freq, espace


# ============================================================
# PICK fs1 / fs2 ALONG CABLE
# ============================================================
x_chan = dist_m / DX

rows = []
# Pick manuel fs1
x_fs1, f_fs1 = func.manual_pick_curve(
    Z, f, x_chan,
    title="Manual picking fs1 - click along first resonance"
)

# Pick manuel fs2
x_fs2, f_fs2 = func.manual_pick_curve(
    Z, f, x_chan,
    title="Manual picking fs2 - click along second resonance"
)

fs1_manual = func.interpolate_manual_picks(
    x_fs1,
    f_fs1,
    x_chan,
    sigma_pts=15
)

fs2_manual = func.interpolate_manual_picks(
    x_fs2,
    f_fs2,
    x_chan,
    sigma_pts=15
)

plt.figure(figsize=(15, 6))

plt.imshow(
    Z,
    aspect="auto",
    origin="lower",
    extent=[x_chan[0], x_chan[-1], f[0], f[-1]]
)

plt.colorbar(label="Relative PSD used for picking")
plt.xlabel("DAS channel index")
plt.ylabel("Frequency [Hz]")
plt.title("Manual resonance picking")

plt.plot(x_fs1, f_fs1, "o", ms=4, label="fs1 manual points")
plt.plot(x_fs2, f_fs2, "o", ms=4, label="fs2 manual points")

plt.plot(x_chan, fs1_manual, lw=2, label="fs1 interpolated")
plt.plot(x_chan, fs2_manual, lw=2, label="fs2 interpolated")

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "manual_resonance_picks_1608.png"), dpi=300)
plt.show()

# ============================================================
# BUILD df_res_raw FROM MANUAL PICKS
# ============================================================

rows = []

for ix, x in enumerate(dist_m):

    fs1 = fs1_manual[ix]
    fs2 = fs2_manual[ix]

    model = func.compute_resonance_model(
        fs1,
        fs2,
        H=None,          # Case 1 : comme papier sans sismique
        cs_avg=CS_AVG,
    )

    if model is None:
        continue

    water_depth = -np.interp(x, dist_bathy, bathy_values)

    rows.append({
        "dist_m": x,
        "dist_km": x / 1000,
        "water_depth_m": water_depth,
        "fs1": fs1,
        "fs2": fs2,
        **model,
    })

np.savez(
    os.path.join(SAVE_DIR, "manual_resonance_picks_points_1608.npz"),
    x_fs1=x_fs1,
    f_fs1=f_fs1,
    x_fs2=x_fs2,
    f_fs2=f_fs2,
) """

""" manual = np.load(os.path.join(SAVE_DIR, "manual_resonance_picks_points.npz"))

x_fs1 = manual["x_fs1"]
f_fs1 = manual["f_fs1"]
x_fs2 = manual["x_fs2"]
f_fs2 = manual["f_fs2"] """

""" df_res_raw = pd.DataFrame(rows)
df_res_raw = df_res_raw.sort_values("dist_m").reset_index(drop=True)


# sauvegarde brut
df_res_raw.to_csv(os.path.join(SAVE_DIR, "resonance_picks_model_case1_raw_1608.csv"), index=False)
df_res_raw.to_pickle(os.path.join(SAVE_DIR, "resonance_picks_model_case1_raw_1608.pkl"))

df_res_raw = pd.read_pickle(os.path.join(SAVE_DIR, "resonance_picks_model_case1_raw_1608.pkl")) """
df_res_raw = pd.read_pickle(os.path.join(SAVE_DIR, "resonance_merged_events_median.pkl"))

# grille régulière
dx_interp = 4.08
x_grid = np.arange(
    df_res_raw["dist_m"].min(),
    df_res_raw["dist_m"].max() + dx_interp,
    dx_interp
)

df_grid = pd.DataFrame({"dist_m": x_grid})
df_grid["dist_km"] = df_grid["dist_m"] / 1000

for col in ["fs1_median", "fs2_median", "water_depth_m"]:
    df_grid[col] = np.interp(
        x_grid,
        df_res_raw["dist_m"].values,
        df_res_raw[col].values
    )

# lissage spatial des fréquences, pas de H
sigma_m = 50
sigma_pts = sigma_m / dx_interp

df_grid["fs1_smooth"] = gaussian_filter1d(df_grid["fs1_median"], sigma_pts)
df_grid["fs2_smooth"] = gaussian_filter1d(df_grid["fs2_median"], sigma_pts)

# recalcul du modèle depuis fs1/fs2 lissés
rows_filled = []

for _, row in df_grid.iterrows():

    model = func.compute_resonance_model(
        row["fs1_smooth"],
        row["fs2_smooth"],
        H=None,
        cs_avg=CS_AVG,
    )

    if model is None:
        continue

    rows_filled.append({
        "dist_m": row["dist_m"],
        "dist_km": row["dist_km"],
        "water_depth_m": row["water_depth_m"],
        **model,
    })

df_res = pd.DataFrame(rows_filled)
df_res = df_res.sort_values("dist_m").reset_index(drop=True)

# sauvegarde final
df_res.to_csv(os.path.join(SAVE_DIR, "resonance_picks_model_case1_1608.csv"), index=False)
df_res.to_pickle(os.path.join(SAVE_DIR, "resonance_picks_model_case1_1608.pkl"))
df_res = pd.read_pickle(os.path.join(SAVE_DIR, "resonance_merged_events_median.pkl"))

print(df_res.head())
print(df_res.describe())


# ============================================================
# BUILD 2D MODEL FOR PLOT
# ============================================================

z_rel = np.linspace(0, np.nanmax(df_res["H_median"]) * 1.05, 80)

X = []
Zabs = []
VS = []

for _, row in df_res.iterrows():

    x = row["dist_m"]
    seafloor = np.interp(x, dist_bathy, bathy_values)

    H = row["H_median"]
    cs0 = row["cs0_median"]
    nu = row["nu_median"]

    z_valid = z_rel[z_rel <= H]

    vs = func.vs_power_law(z_valid, cs0, nu)
    z_abs = seafloor - z_valid

    X.extend([x / 1000] * len(z_valid))
    Zabs.extend(z_abs)
    VS.extend(vs)

X = np.asarray(X)
Zabs = np.asarray(Zabs)
VS = np.asarray(VS)

df_vs_2d = pd.DataFrame({
    "x_km": X,
    "z_m": Zabs,
    "vs_mps": VS,
})

df_vs_2d.to_csv(
    os.path.join(SAVE_DIR, "vs_model_2d_merged.csv"),
    index=False
)

df_vs_2d.to_pickle(
    os.path.join(SAVE_DIR, "vs_model_2d_merged.pkl")
)
# ============================================================
# PLOT RESULT
# ============================================================

fig, ax = plt.subplots(figsize=(15, 6))

sc = ax.scatter(
    X,
    Zabs,
    c=VS,
    s=10,
    cmap="turbo",
    vmin=0,
    vmax=600,
)

ax.plot(
    dist_bathy / 1000,
    bathy_values,
    color="k",
    lw=2,
    label="Bathymetry"
)

# LVL base
base_x = df_res["dist_m"].values / 1000
base_z = np.interp(df_res["dist_m"].values, dist_bathy, bathy_values) - df_res["H_median"].values

ax.plot(
    base_x,
    base_z,
    color="k",
    lw=2,
    ls="--",
    label="Estimated LVL base"
)

ax.set_xlabel("Distance along cable [km]")
ax.set_ylabel("Elevation [m]")
ax.set_title("S-wave resonance model - Taweesintananon-style case 1")
ax.grid(True)
ax.legend()
ax.invert_xaxis()

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Estimated Vs [m/s]")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "resonance_model_case1_merged.png"), dpi=300)
plt.show()


# ============================================================
# PLOT fs1 / fs2
# ============================================================

""" x_chan = dist_m / DX

plt.figure(figsize=(14, 5))

plt.imshow(
    Z,
    aspect="auto",
    origin="lower",
    extent=[x_chan[0], x_chan[-1], f[0], f[-1]]
)

plt.colorbar(label="Relative PSD used for picking")
plt.xlabel("DAS channel index")
plt.ylabel("Frequency [Hz]")
plt.title("DAS f-x spectrum used for picking")

plt.plot(df_res["dist_m"] / DX, df_res["fs1_median"], label="fs1")
plt.plot(df_res["dist_m"] / DX, df_res["fs2_median"], label="fs2")

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "picked_resonance_frequencies_on_Z_merged.png"), dpi=300)
plt.show()

plt.figure(figsize=(14, 5))

plt.imshow(
    Z,
    aspect="auto",
    origin="lower",
    extent=[x_chan[0], x_chan[-1], f[0], f[-1]]
)

plt.colorbar(label="Relative PSD used for picking")
plt.xlabel("DAS channel index")
plt.ylabel("Frequency [Hz]")
plt.title("Raw picks before interpolation/smoothing")

plt.plot(df_res_raw["dist_m"] / DX, df_res_raw["fs1_median"], ".", ms=2, label="fs1 raw")
plt.plot(df_res_raw["dist_m"] / DX, df_res_raw["fs2_median"], ".", ms=2, label="fs2 raw")

plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "raw_resonance_picks_on_Z_merged.png"), dpi=300)
plt.show() """