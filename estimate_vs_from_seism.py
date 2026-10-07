#!/usr/bin/env python3
"""Calculate DAS spectra, pick fs1/fs2 and export the shear-wave velocity model."""

import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vs_functions as func


# =========================== PARAMETERS ============================
SAVE_DIR = "/home/lea/Desktop/code/resonance_model/resonance_model/picking"
DAS_FOLDER = "/media/lea/Expansion/DAS/20250820/dphi"
EVENT_ID = "2008"                    # day/event label; time bounds are added automatically
""" T_START = [4, 54, 42] # 16
T_END = [5, 4, 42]
T_START = [14, 0, 42] # 1508
T_END = [14, 15, 42] """
""" T_START = [12, 30, 42] # 1509
T_END = [12, 40, 42] """
T_START = [19, 40, 42] # 1509
T_END = [19, 55, 42]
T_START = [17, 18, 42] # 12
T_END = [17, 28, 42]
T_START = [19, 35, 42] # 12
T_END = [19, 45, 42]
# Fixed physical reference from the original metadata: channels 5000 to 25004.
# These are section coordinates, independent of each recording's channel grid.
CABLE_ORIGIN_M = 5000 * 1.0213001907746815
SECTION_START_M, SECTION_END_M = 0.0, 20004 * 1.0213001907746815
CHAN_START, CHAN_END = None, None   # optional recorded-index restrictions within the section
DX = None                          # read effective spatial spacing from metadata
FS_IN, FS_OUT = None, 40            # metadata input rate; desired processing rate
FMIN, FMAX = 0.3, FS_OUT//2-1              # requested band; FMAX is capped below Nyquist
NPERSEG_S, NOVERLAP_S = 60, 30
CHUNK_CHANNELS = 100
WELCH_AVERAGE = "mean"

NORMALIZE_BAND = (0.5, FMAX//2)
PREVIEW_BAND = (0.5, FMAX//2)
FS1_PICK_BAND = (0.5, 4.0)          # display bands; no model constraint is imposed
FS2_PICK_BAND = (2.0, FMAX//2)
SPATIAL_SIGMA_CHANNELS = 3.0         # physical smoothing width depends on metadata spacing
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


def analysis_identity(event_id=None, t_start=None, t_end=None):
    """Give every same-day time window its own file key and readable title."""
    event_id = str(EVENT_ID if event_id is None else event_id)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", event_id) or event_id in (".", ".."):
        raise ValueError("EVENT_ID must be a nonempty filename-safe day/event label.")

    def clock(values, name):
        values = np.asarray(values, dtype=float)
        if (values.shape != (3,) or not np.all(np.isfinite(values))
                or not np.all(values == np.floor(values))
                or not np.all((values >= 0) & (values <= [23, 59, 59]))):
            raise ValueError(f"{name} must contain valid integer [hour, minute, second] values.")
        return tuple(int(value) for value in values)

    start = clock(T_START if t_start is None else t_start, "T_START")
    end = clock(T_END if t_end is None else t_end, "T_END")
    if end <= start:
        raise ValueError("T_END must be later than T_START for a same-day analysis.")
    start_text, end_text = (":".join(f"{value:02d}" for value in time)
                            for time in (start, end))
    return dict(analysis_id=f"{event_id}_{start_text.replace(':', '')}_{end_text.replace(':', '')}",
                analysis_title=f"Event {event_id} | {start_text}–{end_text}",
                time_start_label=start_text, time_end_label=end_text)


def resolve_acquisition():
    acquisition = func.resolve_das_acquisition(
        T_START, T_END, DAS_FOLDER, CHAN_START, CHAN_END, FS_OUT, FMIN, FMAX,
        distance_origin_m=CABLE_ORIGIN_M,
        section_start_m=SECTION_START_M, section_end_m=SECTION_END_M)
    for name, requested, detected in (("FS_IN", FS_IN, acquisition["fs_in"]),
                                      ("DX", DX, acquisition["dx"])):
        if requested is not None and not np.isclose(requested, detected, rtol=1e-9):
            raise ValueError(f"{name}={requested} disagrees with the DAS metadata ({detected}). "
                             f"Use {name}=None for automatic detection.")
    return acquisition


def calculation_signature(acquisition=None):
    acquisition = acquisition or resolve_acquisition()
    def number(value):
        return int(value) if value == int(value) else float(value)
    config = dict(version=3, event_id=EVENT_ID, path_folder=DAS_FOLDER,
                  t_start=T_START, t_end=T_END, chan_start=acquisition["chan_start"],
                  chan_end=acquisition["chan_end"], dx=float(acquisition["dx"]),
                  fs_in=number(acquisition["fs_in"]), fs_out=number(acquisition["fs_out"]),
                  fmin=acquisition["fmin"], fmax=acquisition["fmax"], nperseg_s=NPERSEG_S,
                  noverlap_s=NOVERLAP_S, average=WELCH_AVERAGE,
                  distance_origin_m=acquisition["distance_origin_m"],
                  pick_coordinate="section_distance_km")
    # Include offsets and missing positions so cached spectra and picks stay aligned.
    if not np.allclose(acquisition["dist_m"], acquisition["chx"] * acquisition["dx"],
                       rtol=0, atol=1e-6):
        config["channel_distances_m"] = acquisition["dist_m"].tolist()
    encoded = json.dumps(config, sort_keys=True)
    return config, hashlib.sha256(encoded.encode()).hexdigest()


def load_or_compute_psd(out_dir, signature, acquisition, analysis_id=None):
    analysis_id = analysis_id or analysis_identity()["analysis_id"]
    cache = out_dir / f"das_fx_psd_{analysis_id}.npz"
    legacy_cache = out_dir / f"das_fx_psd_{EVENT_ID}.npz"
    for candidate in (cache, legacy_cache):
        if not REUSE_CACHE or not candidate.exists():
            continue
        with np.load(candidate, allow_pickle=False) as saved:
            if str(saved["signature"].item()) == signature:
                print(f"Loaded cached PSD: {candidate}")
                result = {key: saved[key].copy() for key in saved.files if key != "signature"}
                result.update(dist_m=acquisition["dist_m"], dx=acquisition["dx"],
                              fs_in=acquisition["fs_in"],
                              absolute_channels=acquisition["absolute_channels"],
                              metadata_dist_m=acquisition["metadata_dist_m"],
                              distance_origin_m=acquisition["distance_origin_m"])
                if candidate != cache:
                    atomic_npz(cache, signature=signature, **result)
                return result
        if candidate == cache:
            print("Calculation settings changed: recomputing PSDs.")
    res = func.das_fx_spectrum_chunked(
        t_start=T_START, t_end=T_END, path_folder=DAS_FOLDER,
        chan_start_idx=acquisition["chan_start"], chan_end_idx=acquisition["chan_end"],
        chunk_channels=CHUNK_CHANNELS, fs_in=FS_IN, fs_out=FS_OUT,
        fmin=FMIN, fmax=FMAX, nperseg_s=NPERSEG_S,
        noverlap_s=NOVERLAP_S, average=WELCH_AVERAGE, acquisition=acquisition)
    atomic_npz(cache, signature=signature, **res)
    return res


def load_bathymetry(dist_m):
    # Bathymetry dependencies are required only when BUILD_MODEL=True.
    import xarray as xr
    from pyproj import Transformer
    import geo_functions as gf
    with xr.open_dataset(BATHY_PATH) as ds:
        cable_lat, cable_lon = gf.interpolate_cable_coordinates(
            CABLE_CSV, dist_m, allow_outside=True)
        outside = ~np.isfinite(cable_lat) | ~np.isfinite(cable_lon)
        if np.any(outside):
            print(f"Cable geometry does not cover {np.count_nonzero(outside)} selected channels; "
                  "their bathymetry is left missing.")
        transform = Transformer.from_crs("EPSG:4326", "EPSG:3996", always_xy=True)
        water_depth = np.full(len(dist_m), np.nan)
        inside = ~outside
        if np.any(inside):
            cable_x, cable_y = transform.transform(cable_lon[inside], cable_lat[inside])
            bathy_values = ds["z"].interp(
                x=xr.DataArray(cable_x, dims="channel"),
                y=xr.DataArray(cable_y, dims="channel")).values
            water_depth[inside] = -bathy_values
    return water_depth


def main():
    identity = analysis_identity()
    analysis_id, title = identity["analysis_id"], identity["analysis_title"]
    out = Path(SAVE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    acquisition = resolve_acquisition()
    dx = acquisition["dx"]
    config, signature = calculation_signature(acquisition)
    # Check existing picks BEFORE starting a potentially long calculation.
    picks_path = out / f"manual_resonance_picks_points_{analysis_id}.npz"
    legacy_picks = out / f"manual_resonance_picks_points_{EVENT_ID}.npz"
    picks_source = picks_path if picks_path.exists() else legacy_picks
    picks = {name: np.array([], dtype=float)
             for name in ("x_fs1", "f_fs1", "x_fs2", "f_fs2")}
    if REUSE_PICKS and picks_source.exists():
        with np.load(picks_source, allow_pickle=False) as saved:
            if str(saved["signature"].item()) != signature:
                if picks_source == picks_path:
                    raise ValueError("Existing picks belong to a different calculation configuration. "
                                     "Change EVENT_ID/SAVE_DIR or set REUSE_PICKS=False.")
                print("Legacy picks belong to another window or configuration; starting a separate analysis.")
            else:
                for name in picks:
                    picks[name] = saved[name].copy()
                if picks_source != picks_path:
                    atomic_npz(picks_path, signature=signature, **picks)
                    print(f"Copied compatible legacy picks to {picks_path}")
    if not DO_MANUAL_PICKING and not any(len(picks[name]) for name in picks):
        raise ValueError("No saved picks to export: enable DO_MANUAL_PICKING.")
    settings = dict(config, **identity, normalize_band=NORMALIZE_BAND,
                    preview_band=PREVIEW_BAND, fs1_band=FS1_PICK_BAND,
                    fs2_band=FS2_PICK_BAND, spatial_sigma_channels=SPATIAL_SIGMA_CHANNELS,
                    freq_sigma_hz=FREQ_SIGMA_HZ, color_percentiles=COLOR_PERCENTILES,
                    segment_channels=SEGMENT_CHANNELS,
                    max_interp_gap_channels=MAX_INTERP_GAP_CHANNELS,
                    cs_avg_assumed=CS_AVG, build_model=BUILD_MODEL,
                    plot_model=PLOT_MODEL, model_depth_max_m=MODEL_DEPTH_MAX_M,
                    model_depth_samples=MODEL_DEPTH_SAMPLES,
                    model_reverse_distance=MODEL_REVERSE_DISTANCE,
                    model_vs_limits=MODEL_VS_LIMITS,
                    metadata_channel_count=acquisition["n_channels"],
                    metadata_header_dx=acquisition["header_dx"],
                    section_start_m=SECTION_START_M, section_end_m=SECTION_END_M,
                    selected_metadata_channel_start=int(acquisition["absolute_channels"][0]),
                    selected_metadata_channel_end=int(acquisition["absolute_channels"][-1]))
    (out / f"settings_{analysis_id}.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8")
    res = load_or_compute_psd(out, signature, acquisition, analysis_id)
    f, dist_m, Z_db = func.extract_fx_arrays(res, dx=dx)
    normalize_band = func.clip_frequency_band(NORMALIZE_BAND, f)
    preview_band = func.clip_frequency_band(PREVIEW_BAND, f)
    pick_bands = {"fs1": func.clip_frequency_band(FS1_PICK_BAND, f),
                  "fs2": func.clip_frequency_band(FS2_PICK_BAND, f)}
    x_km = dist_m / 1000
    x_label = "Distance along section [km]"
    raw, smooth = func.prepare_display(
        Z_db, f, normalize_band, SPATIAL_SIGMA_CHANNELS, FREQ_SIGMA_HZ)
    print(f"Frequency spacing: {np.median(np.diff(f)):.5f} Hz; "
          f"loaded duration: {float(res['duration_s']):.1f} s")
    print("The cache stores PSDs before normalization and smoothing.")
    preview = func.plot_comparison(
        raw, smooth, f, x_km, preview_band, COLOR_PERCENTILES,
        title=f"{title} | spatial sigma = {SPATIAL_SIGMA_CHANNELS * dx:.2f} m",
        save_path=out / f"fx_comparison_{analysis_id}.png", x_label=x_label)
    if SHOW_PREVIEW:
        print("Close the comparison figure to continue.")
        plt.show()
    plt.close(preview)

    def save_picks(mode, px, pf):
        picks[f"x_{mode}"] = np.asarray(px)
        picks[f"f_{mode}"] = np.asarray(pf)
        atomic_npz(picks_path, signature=signature, **picks)

    if DO_MANUAL_PICKING:
        for mode, band in pick_bands.items():
            px, pf = func.manual_pick_curve(
                raw, smooth, f, x_km, title=f"{title} | {mode}", band=band,
                segment_channels=SEGMENT_CHANNELS * dx / 1000, percentiles=COLOR_PERCENTILES,
                initial_picks=(picks[f"x_{mode}"], picks[f"f_{mode}"]),
                on_update=lambda px, pf, mode=mode: save_picks(mode, px, pf),
                x_label=x_label, position_label="distance [km]")
            save_picks(mode, px, pf)
    fs1 = func.interpolate_manual_picks(
        picks["x_fs1"], picks["f_fs1"], x_km,
        None if MAX_INTERP_GAP_CHANNELS is None else MAX_INTERP_GAP_CHANNELS * dx / 1000)
    fs2 = func.interpolate_manual_picks(
        picks["x_fs2"], picks["f_fs2"], x_km,
        None if MAX_INTERP_GAP_CHANNELS is None else MAX_INTERP_GAP_CHANNELS * dx / 1000)
    df = pd.DataFrame(dict(channel=np.arange(len(x_km)), recorded_channel=res["chx"],
                           absolute_channel=acquisition["absolute_channels"],
                           metadata_dist_m=acquisition["metadata_dist_m"],
                           distance_origin_m=acquisition["distance_origin_m"],
                           dist_m=dist_m, dist_km=dist_m / 1000, dx_m=dx,
                           fs1=fs1, fs2=fs2))
    df["valid_pair"] = np.isfinite(fs1) & np.isfinite(fs2) & (fs2 > fs1)
    df["event_id"] = EVENT_ID
    for name in ("analysis_id", "time_start_label", "time_end_label"):
        df[name] = identity[name]
    df.to_csv(out / f"resonance_frequencies_{analysis_id}.csv", index=False)

    fig = func.plot_comparison(raw, smooth, f, x_km, preview_band,
                               COLOR_PERCENTILES, title=f"{title} | picks", x_label=x_label)
    for ax in fig.axes[:2]:
        ax.plot(x_km, fs1, color="white", lw=0.8, label="fs1 interpolated")
        ax.plot(x_km, fs2, color="tab:orange", lw=0.8, label="fs2 interpolated")
        ax.plot(picks["x_fs1"], picks["f_fs1"], "o", color="white", mec="black", ms=3)
        ax.plot(picks["x_fs2"], picks["f_fs2"], "o", color="tab:orange", mec="black", ms=3)
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / f"manual_resonance_picks_{analysis_id}.png", dpi=250)
    plt.close(fig)

    if BUILD_MODEL:
        water_depth = load_bathymetry(dist_m)
        rows = []
        for i, (f1, f2) in enumerate(zip(fs1, fs2)):
            model = func.compute_resonance_model(f1, f2, H=None, cs_avg=CS_AVG)
            if model is not None:
                rows.append(dict(dist_m=dist_m[i], dist_km=dist_m[i] / 1000,
                                 water_depth_m=water_depth[i],
                                 distance_origin_m=acquisition["distance_origin_m"], **model))
        columns = ["dist_m", "dist_km", "water_depth_m", "distance_origin_m", "fs1", "fs2", "nu",
                   "cs0", "H", "cs_avg_model", "cs_avg_assumed"]
        df_model = pd.DataFrame(rows, columns=columns)
        df_model["event_id"] = EVENT_ID
        for name in ("analysis_id", "time_start_label", "time_end_label"):
            df_model[name] = identity[name]
        base = out / f"resonance_picks_model_case1_{analysis_id}"
        df_model.to_csv(str(base) + ".csv", index=False)
        df_model.to_pickle(str(base) + ".pkl")
        print(f"Exported model: {len(df_model)} / {len(x_km)} valid channels.")
        if PLOT_MODEL:
            aligned = df_model.set_index("dist_m").reindex(dist_m)
            figures = func.plot_resonance_model(
                dist_m, aligned["cs0"], aligned["nu"], aligned["H"], dx=dx,
                title=f"{title} | Vs model",
                depth_max_m=MODEL_DEPTH_MAX_M, depth_samples=MODEL_DEPTH_SAMPLES,
                water_depth_m=water_depth, reverse_distance=MODEL_REVERSE_DISTANCE,
                vs_limits=MODEL_VS_LIMITS)
            if figures is not None:
                for figure, suffix in zip(figures, ("vs_model", "vs_model_parameters")):
                    figure.savefig(out / f"{suffix}_{analysis_id}.png", dpi=250)
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
