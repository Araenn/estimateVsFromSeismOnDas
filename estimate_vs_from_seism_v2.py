#!/usr/bin/env python3
"""Lancer ce fichier dans un terminal Python avec une interface graphique.

Placer vs_functions_v2.py dans le même dossier. Adapter PARAMÈTRES ci-dessous.
Les sorties sont séparées de celles de la version originale.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vs_functions_v2 as func


# =========================== PARAMÈTRES ============================
PROJECT_LIB = "/home/lea/Desktop/code/thesis/das_datas/lib/"
SAVE_DIR = "/home/lea/Desktop/code/thesis/das_datas/resonance_model/picking_v2"
DAS_FOLDER = "/media/lea/Expansion/DAS/20250816/dphi"
EVENT_ID = "1608"                    # modifier pour 1208 et 1508
T_START = [4, 54, 42] #16
T_END = [5, 4, 42]
""" T_START = [14, 0, 42] #15
T_END = [14, 15, 42] """
""" T_START = [17, 18, 42] #12
T_END = [17, 28, 42] """
CHAN_START, CHAN_END = 0, 5001
DX = 4.08
FS_IN, FS_OUT = 2000, 40
FMIN, FMAX = 0.3, 19.0              # bande de calcul, inchangée
NPERSEG_S, NOVERLAP_S = 60, 30
CHUNK_CHANNELS = 100
WELCH_AVERAGE = "mean"              # garder mean pour le premier essai

NORMALIZE_BAND = (0.5, 9.0)
PREVIEW_BAND = (0.5, 9.0)
FS1_PICK_BAND = (0.5, 4.0)          # bandes de visualisation, pas de modèle imposé
FS2_PICK_BAND = (2.0, 9.0)
SPATIAL_SIGMA_CHANNELS = 3.0         # 12,24 m sigma, ~29 m à mi-hauteur
FREQ_SIGMA_HZ = 0.0                 # pas de lissage fréquentiel
COLOR_PERCENTILES = (5, 95)
SEGMENT_CHANNELS = 1000
MAX_INTERP_GAP_CHANNELS = 500        # longs trous laissés à NaN

REUSE_CACHE = True                  # relire la PSD si config de calcul identique
REUSE_PICKS = True                  # recharger les points pour les corriger
DO_MANUAL_PICKING = True            # False: réexporter les points déjà sauvegardés
SHOW_PREVIEW = True                 # fermer l'aperçu pour commencer le picking
BUILD_MODEL = True                  # False: essayer uniquement PSD et picking
CS_AVG = 200.0                      # même hypothèse que dans le script original
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
    cache = out_dir / f"das_fx_psd_{EVENT_ID}_v2.npz"
    if REUSE_CACHE and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["signature"].item()) == signature:
                print(f"PSD rechargée: {cache}")
                return {key: saved[key].copy() for key in saved.files
                        if key != "signature"}
        print("Configuration modifiée: nouveau calcul PSD.")
    res = func.das_fx_spectrum_chunked(
        t_start=T_START, t_end=T_END, path_folder=DAS_FOLDER,
        chan_start_idx=CHAN_START, chan_end_idx=CHAN_END,
        chunk_channels=CHUNK_CHANNELS, fs_in=FS_IN, fs_out=FS_OUT,
        fmin=FMIN, fmax=FMAX, nperseg_s=NPERSEG_S,
        noverlap_s=NOVERLAP_S, average=WELCH_AVERAGE)
    atomic_npz(cache, signature=signature, **res)
    return res


def load_bathymetry(dist_m):
    # Les dépendances bathymétriques ne sont requises que si BUILD_MODEL=True.
    import xarray as xr
    from pyproj import Transformer
    import map.geo_functions as gf
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
    sys.path.append(PROJECT_LIB)
    out = Path(SAVE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    config, signature = calculation_signature()
    # Vérifier les éventuels picks AVANT un nouveau calcul long.
    picks_path = out / f"manual_resonance_picks_points_{EVENT_ID}_v2.npz"
    picks = {name: np.array([], dtype=float)
             for name in ("x_fs1", "f_fs1", "x_fs2", "f_fs2")}
    if REUSE_PICKS and picks_path.exists():
        with np.load(picks_path, allow_pickle=False) as saved:
            if str(saved["signature"].item()) != signature:
                raise ValueError("Les picks existants viennent d'une autre configuration. "
                                 "Changer EVENT_ID/SAVE_DIR ou mettre REUSE_PICKS=False.")
            for name in picks:
                picks[name] = saved[name].copy()
    if not DO_MANUAL_PICKING and not any(len(picks[name]) for name in picks):
        raise ValueError("Pas de picks à réexporter: activer DO_MANUAL_PICKING.")
    settings = dict(config, normalize_band=NORMALIZE_BAND,
                    preview_band=PREVIEW_BAND, fs1_band=FS1_PICK_BAND,
                    fs2_band=FS2_PICK_BAND, spatial_sigma_channels=SPATIAL_SIGMA_CHANNELS,
                    freq_sigma_hz=FREQ_SIGMA_HZ, color_percentiles=COLOR_PERCENTILES,
                    segment_channels=SEGMENT_CHANNELS,
                    max_interp_gap_channels=MAX_INTERP_GAP_CHANNELS,
                    cs_avg_assumed=CS_AVG, build_model=BUILD_MODEL)
    (out / f"settings_{EVENT_ID}_v2.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8")
    res = load_or_compute_psd(out, signature)
    f, dist_m, Z_db = func.extract_fx_arrays(res, dx=DX)
    x_chan = np.asarray(res["chx"], dtype=float)
    raw, smooth = func.prepare_display(
        Z_db, f, NORMALIZE_BAND, SPATIAL_SIGMA_CHANNELS, FREQ_SIGMA_HZ)
    print(f"Pas fréquentiel: {np.median(np.diff(f)):.5f} Hz; "
          f"durée chargée: {float(res['duration_s']):.1f} s")
    print("Le cache contient les PSD avant normalisation et avant lissage.")
    preview = func.plot_comparison(
        raw, smooth, f, x_chan, PREVIEW_BAND, COLOR_PERCENTILES,
        title=f"Événement {EVENT_ID} | σ spatial = {SPATIAL_SIGMA_CHANNELS * DX:.2f} m",
        save_path=out / f"fx_comparison_{EVENT_ID}_v2.png")
    if SHOW_PREVIEW:
        print("Fermer la figure de comparaison pour poursuivre.")
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
    df.to_csv(out / f"resonance_frequencies_{EVENT_ID}_v2.csv", index=False)

    fig = func.plot_comparison(raw, smooth, f, x_chan, PREVIEW_BAND,
                               COLOR_PERCENTILES, title=f"Picks {EVENT_ID}")
    for ax in fig.axes[:2]:
        ax.plot(x_chan, fs1, color="white", lw=0.8, label="fs1 interpolée")
        ax.plot(x_chan, fs2, color="tab:orange", lw=0.8, label="fs2 interpolée")
        ax.plot(picks["x_fs1"], picks["f_fs1"], "o", color="white", mec="black", ms=3)
        ax.plot(picks["x_fs2"], picks["f_fs2"], "o", color="tab:orange", mec="black", ms=3)
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / f"manual_resonance_picks_{EVENT_ID}_v2.png", dpi=250)
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
        # Noms compatibles avec le merge; dossier V2 distinct de l'original.
        base = out / f"resonance_picks_model_case1_{EVENT_ID}"
        df_model.to_csv(str(base) + ".csv", index=False)
        df_model.to_pickle(str(base) + ".pkl")
        print(f"Modèle exporté: {len(df_model)} / {len(x_chan)} canaux valides.")
    print(f"Terminé. Sorties: {out}")


if __name__ == "__main__":
    try:
        main()
    except func.PickingInterrupted as exc:
        print(exc)
