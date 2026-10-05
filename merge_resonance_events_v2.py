#!/usr/bin/env python3
"""Fusion des essais V2 sur les canaux communs, sans remplir les longs trous.

Les fréquences viennent des CSV de picking. Les paramètres physiques viennent
des PKL du même événement si BUILD_MODEL était activé. Les noms du résultat
restent compatibles avec les figures de la version originale.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAVE_DIR = "/home/lea/Desktop/code/thesis/das_datas/resonance_model/picking_v2"
EVENT_IDS = ["1208", "1508", "1608"]
DX = 4.08


def merge_events(folder, event_ids, dx=4.08):
    folder = Path(folder)
    events = []
    for event_id in event_ids:
        path = folder / f"resonance_frequencies_{event_id}_v2.csv"
        if not path.exists():
            print(f"{event_id}: pas encore traité, ignoré.")
            continue
        df = pd.read_csv(path)
        df["channel"] = df["channel"].round().astype(int)
        if df["channel"].duplicated().any():
            raise ValueError(f"Canaux dupliqués pour {event_id}.")
        if not np.allclose(df["dist_m"], df["channel"] * dx):
            raise ValueError(f"Espacement DX incompatible pour {event_id}.")
        df = df.set_index("channel")
        # Ne pas relier les zones rejetées par le modèle via np.interp.
        model_path = folder / f"resonance_picks_model_case1_{event_id}.pkl"
        model_cols = ["water_depth_m", "H", "cs0", "nu"]
        for col in model_cols:
            df[col] = np.nan
        settings_path = folder / f"settings_{event_id}_v2.json"
        use_model = True
        if settings_path.exists():
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            use_model = settings.get("build_model", True)
        if use_model and model_path.exists():
            model = pd.read_pickle(model_path)
            model["channel"] = np.rint(model["dist_m"] / dx).astype(int)
            model = model.set_index("channel")
            aligned = model.reindex(df.index)
            agrees = (np.isclose(df["fs1"], aligned["fs1"], rtol=1e-8, atol=1e-8)
                      & np.isclose(df["fs2"], aligned["fs2"], rtol=1e-8, atol=1e-8))
            for col in model_cols:
                df[col] = aligned[col].where(agrees)
        df["event_id"] = event_id
        df["valid_fs1"] = np.isfinite(df["fs1"])
        df["valid_fs2"] = np.isfinite(df["fs2"])
        df["valid_model"] = np.isfinite(df["H"])
        events.append(df)
    if not events:
        raise FileNotFoundError("Aucun CSV de fréquences V2 dans SAVE_DIR.")
    lo = max(df.index.min() for df in events)
    hi = min(df.index.max() for df in events)
    if hi < lo:
        raise ValueError("Pas de canaux communs entre les événements.")
    all_events = pd.concat([df.loc[(df.index >= lo) & (df.index <= hi)]
                            for df in events]).reset_index()
    aggregations = dict(water_depth_m=("water_depth_m", "median"),
                        n_events=("valid_model", "sum"),
                        n_events_fs1=("valid_fs1", "sum"),
                        n_events_fs2=("valid_fs2", "sum"))
    for col in ("fs1", "fs2", "H", "cs0", "nu"):
        aggregations[f"{col}_median"] = (col, "median")
        aggregations[f"{col}_std"] = (col, "std")
        if col in ("fs1", "fs2", "H"):
            aggregations[f"{col}_mean"] = (col, "mean")
    merged = all_events.groupby("channel").agg(**aggregations).reset_index()
    merged["dist_m"] = merged["channel"] * dx
    merged["dist_km"] = merged["dist_m"] / 1000
    merged["fs2_fs1_ratio"] = merged["fs2_median"] / merged["fs1_median"]
    return all_events, merged


def main():
    folder = Path(SAVE_DIR)
    all_events, merged = merge_events(folder, EVENT_IDS, DX)
    all_events.to_csv(folder / "resonance_all_events_interpolated.csv", index=False)
    merged.to_csv(folder / "resonance_merged_events_median.csv", index=False)
    merged.to_pickle(folder / "resonance_merged_events_median.pkl")
    for mode in ("fs1", "fs2"):
        fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
        for event_id, df in all_events.groupby("event_id"):
            ax.plot(df["channel"], df[mode], lw=0.8, alpha=0.6, label=event_id)
        ax.plot(merged["channel"], merged[f"{mode}_median"], "k", lw=1.5,
                label="Médiane")
        # Écart-type entre événements, pas une incertitude de chaque clic.
        center, std = merged[f"{mode}_median"], merged[f"{mode}_std"]
        ax.fill_between(merged["channel"], center - std, center + std,
                        color="0.5", alpha=0.2, label="± écart-type entre événements")
        ax.set(xlabel="Canal DAS", ylabel="Fréquence [Hz]", title=mode)
        ax.legend(fontsize=8)
        fig.savefig(folder / f"merged_{mode}_events.png", dpi=250)
        plt.close(fig)
    print(f"Fusion exportée dans {folder}")


if __name__ == "__main__":
    main()
