#!/usr/bin/env python3
"""Calculate DAS spectra, pick fs1/fs2 and export the shear-wave velocity model."""

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vs_functions as func


# =========================== PARAMETERS ============================
SAVE_DIR = "/home/lea/Desktop/code/resonance_model/resonance_model/picking"
DAS_FOLDER = "/media/lea/Expansion/DAS/20250815/dphi"
EVENT_ID = "1508"                    # change for 1208 and 1508
T_START = [4, 54, 42] # 16
T_END = [5, 4, 42]
T_START = [14, 0, 42] # 15
T_END = [14, 15, 42]
""" T_START = [17, 18, 42] # 12
T_END = [17, 28, 42] """
CHAN_START, CHAN_END = 0, 5001
DX = 4.08
FS_IN, FS_OUT = 2000, 40
FMIN, FMAX = 0.3, 19.0              # unchanged calculation band
NPERSEG_S, NOVERLAP_S = 60, 30
CHUNK_CHANNELS = 100
WELCH_AVERAGE = "mean"

NORMALIZE_BAND = (0.5, 9.0)
PREVIEW_BAND = (0.5, 9.0)
FS1_PICK_BAND = (0.5, 4.0)          # display bands; no model constraint is imposed
FS2_PICK_BAND = (2.0, 9.0)
SPATIAL_SIGMA_CHANNELS = 3.0         # 12.24 m standard deviation; approximately 29 m FWHM
FREQ_SIGMA_HZ = 0.0                 # no frequency smoothing
COLOR_PERCENTILES = (5, 95)
SEGMENT_CHANNELS = 1000
MAX_INTERP_GAP_CHANNELS = 500        # retain NaN across long gaps

REUSE_CACHE = True                  # reuse PSDs when calculation settings match
REUSE_PICKS = True                  # reload existing picks for editing
DO_MANUAL_PICKING = True            # False: export previously saved picks without editing
SHOW_PREVIEW = True                 # close the preview to start picking
BUILD_MODEL = True                  # False: run spectrum calculation and picking only
PLOT_MODEL = True                   # export the Vs section and model parameters
SHOW_MODEL = True                   # display model figures after exporting them
MODEL_DEPTH_MAX_M = None            # None: use the largest valid layer thickness
MODEL_DEPTH_SAMPLES = 300
MODEL_REVERSE_DISTANCE = True       # original orientation: higher distances on the left
MODEL_VS_LIMITS = (0, 600)           # original color scale; does not clip model values
CS_AVG = 200.0                      # same assumption as in the original script
BATHY_PATH = "/home/lea/Desktop/code/thesis/das_datas/bathy_svalbard_subset.nc"
CABLE_CSV = "/home/lea/Desktop/code/thesis/das_datas/map/outer_cable_positions.csv"
# ==================================================================


def atomic_npz(path, **arrays):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def calculation_signature():
    config = dict(version=2, event_id=EVENT_ID, path_folder=DAS_FOLDER,
                  t_start=T_START, t_end=T_END, chan_start=CHAN_START,
                  chan_end=CHAN_END, dx=DX, fs_in=FS_IN, fs_out=FS_OUT,
                  fmin=FMIN, fmax=FMAX, nperseg_s=NPERSEG_S,
                  noverlap_s=NOVERLAP_S, average=WELCH_AVERAGE)
    encoded = json.dumps(config, sort_keys=True)
    return config, hashlib.sha256(encoded.encode()).hexdigest()


def load_or_compute_psd(out_dir, signature):
    cache = out_dir / f"das_fx_psd_{EVENT_ID}.npz"
    if REUSE_CACHE and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["signature"].item()) == signature:
                print(f"Loaded cached PSD: {cache}")
                return {key: saved[key].copy() for key in saved.files
                        if key != "signature"}
        print("Calculation settings changed: recomputing PSDs.")
    res = func.das_fx_spectrum_chunked(
        t_start=T_START, t_end=T_END, path_folder=DAS_FOLDER,
        chan_start_idx=CHAN_START, chan_end_idx=CHAN_END,
        chunk_channels=CHUNK_CHANNELS, fs_in=FS_IN, fs_out=FS_OUT,
        fmin=FMIN, fmax=FMAX, nperseg_s=NPERSEG_S,
        noverlap_s=NOVERLAP_S, average=WELCH_AVERAGE)
    atomic_npz(cache, signature=signature, **res)
    return res


def load_bathymetry(dist_m):
    # Bathymetry dependencies are required only when BUILD_MODEL=True.
    import xarray as xr
    from pyproj import Transformer
    import geo_functions as gf
    with xr.open_dataset(BATHY_PATH) as ds:
        _, _, cable_lat, cable_lon = gf.create_cable_map(
            CABLE_CSV, CHAN_END + 1, DX, start_offset_m=0)
        transform = Transformer.from_crs("EPSG:4326", "EPSG:3996", always_xy=True)
        cable_x, cable_y = transform.transform(cable_lon, cable_lat)
        bathy_values = ds["z"].interp(
            x=xr.DataArray(cable_x, dims="channel"),
            y=xr.DataArray(cable_y, dims="channel")).values
    return -np.interp(dist_m, np.arange(len(bathy_values)) * DX, bathy_values)


def main():
    out = Path(SAVE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    config, signature = calculation_signature()
    # Check existing picks BEFORE starting a potentially long calculation.
    picks_path = out / f"manual_resonance_picks_points_{EVENT_ID}.npz"
    picks = {name: np.array([], dtype=float)
             for name in ("x_fs1", "f_fs1", "x_fs2", "f_fs2")}
    if REUSE_PICKS and picks_path.exists():
        with np.load(picks_path, allow_pickle=False) as saved:
            if str(saved["signature"].item()) != signature:
                raise ValueError("Existing picks belong to a different calculation configuration. "
                                 "Change EVENT_ID/SAVE_DIR or set REUSE_PICKS=False.")
            for name in picks:
                picks[name] = saved[name].copy()
    if not DO_MANUAL_PICKING and not any(len(picks[name]) for name in picks):
        raise ValueError("No saved picks to export: enable DO_MANUAL_PICKING.")
    settings = dict(config, normalize_band=NORMALIZE_BAND,
                    preview_band=PREVIEW_BAND, fs1_band=FS1_PICK_BAND,
                    fs2_band=FS2_PICK_BAND, spatial_sigma_channels=SPATIAL_SIGMA_CHANNELS,
                    freq_sigma_hz=FREQ_SIGMA_HZ, color_percentiles=COLOR_PERCENTILES,
                    segment_channels=SEGMENT_CHANNELS,
                    max_interp_gap_channels=MAX_INTERP_GAP_CHANNELS,
                    cs_avg_assumed=CS_AVG, build_model=BUILD_MODEL,
                    plot_model=PLOT_MODEL, model_depth_max_m=MODEL_DEPTH_MAX_M,
                    model_depth_samples=MODEL_DEPTH_SAMPLES,
                    model_reverse_distance=MODEL_REVERSE_DISTANCE,
                    model_vs_limits=MODEL_VS_LIMITS)
    (out / f"settings_{EVENT_ID}.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8")
    res = load_or_compute_psd(out, signature)
    f, dist_m, Z_db = func.extract_fx_arrays(res, dx=DX)
    x_chan = np.asarray(res["chx"], dtype=float)
    raw, smooth = func.prepare_display(
        Z_db, f, NORMALIZE_BAND, SPATIAL_SIGMA_CHANNELS, FREQ_SIGMA_HZ)
    print(f"Frequency spacing: {np.median(np.diff(f)):.5f} Hz; "
          f"loaded duration: {float(res['duration_s']):.1f} s")
    print("The cache stores PSDs before normalization and smoothing.")
    preview = func.plot_comparison(
        raw, smooth, f, x_chan, PREVIEW_BAND, COLOR_PERCENTILES,
        title=f"Event {EVENT_ID} | spatial sigma = {SPATIAL_SIGMA_CHANNELS * DX:.2f} m",
        save_path=out / f"fx_comparison_{EVENT_ID}.png")
    if SHOW_PREVIEW:
        print("Close the comparison figure to continue.")
        plt.show()
    plt.close(preview)

    def save_picks(mode, px, pf):
        picks[f"x_{mode}"] = np.asarray(px)
        picks[f"f_{mode}"] = np.asarray(pf)
        atomic_npz(picks_path, signature=signature, **picks)

    if DO_MANUAL_PICKING:
        for mode, band in (("fs1", FS1_PICK_BAND), ("fs2", FS2_PICK_BAND)):
            px, pf = func.manual_pick_curve(
                raw, smooth, f, x_chan, title=f"{EVENT_ID} / {mode}", band=band,
                segment_channels=SEGMENT_CHANNELS, percentiles=COLOR_PERCENTILES,
                initial_picks=(picks[f"x_{mode}"], picks[f"f_{mode}"]),
                on_update=lambda px, pf, mode=mode: save_picks(mode, px, pf))
            save_picks(mode, px, pf)
    fs1 = func.interpolate_manual_picks(
        picks["x_fs1"], picks["f_fs1"], x_chan, MAX_INTERP_GAP_CHANNELS)
    fs2 = func.interpolate_manual_picks(
        picks["x_fs2"], picks["f_fs2"], x_chan, MAX_INTERP_GAP_CHANNELS)
    df = pd.DataFrame(dict(channel=x_chan, dist_m=dist_m, dist_km=dist_m / 1000,
                           fs1=fs1, fs2=fs2))
    df["valid_pair"] = np.isfinite(fs1) & np.isfinite(fs2) & (fs2 > fs1)
    df.to_csv(out / f"resonance_frequencies_{EVENT_ID}.csv", index=False)

    fig = func.plot_comparison(raw, smooth, f, x_chan, PREVIEW_BAND,
                               COLOR_PERCENTILES, title=f"Picks {EVENT_ID}")
    for ax in fig.axes[:2]:
        ax.plot(x_chan, fs1, color="white", lw=0.8, label="fs1 interpolated")
        ax.plot(x_chan, fs2, color="tab:orange", lw=0.8, label="fs2 interpolated")
        ax.plot(picks["x_fs1"], picks["f_fs1"], "o", color="white", mec="black", ms=3)
        ax.plot(picks["x_fs2"], picks["f_fs2"], "o", color="tab:orange", mec="black", ms=3)
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / f"manual_resonance_picks_{EVENT_ID}.png", dpi=250)
    plt.close(fig)

    if BUILD_MODEL:
        water_depth = load_bathymetry(dist_m)
        rows = []
        for i, (f1, f2) in enumerate(zip(fs1, fs2)):
            model = func.compute_resonance_model(f1, f2, H=None, cs_avg=CS_AVG)
            if model is not None:
                rows.append(dict(dist_m=dist_m[i], dist_km=dist_m[i] / 1000,
                                 water_depth_m=water_depth[i], **model))
        columns = ["dist_m", "dist_km", "water_depth_m", "fs1", "fs2", "nu",
                   "cs0", "H", "cs_avg_model", "cs_avg_assumed"]
        df_model = pd.DataFrame(rows, columns=columns)
        # Filenames remain compatible with the merge script.
        base = out / f"resonance_picks_model_case1_{EVENT_ID}"
        df_model.to_csv(str(base) + ".csv", index=False)
        df_model.to_pickle(str(base) + ".pkl")
        print(f"Exported model: {len(df_model)} / {len(x_chan)} valid channels.")
        if PLOT_MODEL:
            aligned = df_model.set_index("dist_m").reindex(dist_m)
            figures = func.plot_resonance_model(
                dist_m, aligned["cs0"], aligned["nu"], aligned["H"], dx=DX,
                title=f"Event {EVENT_ID} | Vs model",
                depth_max_m=MODEL_DEPTH_MAX_M, depth_samples=MODEL_DEPTH_SAMPLES,
                water_depth_m=water_depth, reverse_distance=MODEL_REVERSE_DISTANCE,
                vs_limits=MODEL_VS_LIMITS)
            if figures is not None:
                for figure, suffix in zip(figures, ("vs_model", "vs_model_parameters")):
                    figure.savefig(out / f"{suffix}_{EVENT_ID}.png", dpi=250)
                if SHOW_MODEL:
                    plt.show()
                for figure in figures:
                    plt.close(figure)
    print(f"Done. Output directory: {out}")


if __name__ == "__main__":
    try:
        main()
    except func.PickingInterrupted as exc:
        print(exc)
