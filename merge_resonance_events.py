#!/usr/bin/env python3
"""Merge trials on their common channels without filling long gaps.

Frequencies are read from picking CSVs. Physical parameters are read
from the same event PKL when BUILD_MODEL was enabled. Output filenames
remain compatible with the original plotting scripts.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from estimate_vs_from_seism import (SAVE_DIR, DX, PLOT_MODEL, SHOW_MODEL,
                                   MODEL_DEPTH_MAX_M, MODEL_DEPTH_SAMPLES,
                                   MODEL_REVERSE_DISTANCE, MODEL_VS_LIMITS)
import vs_functions as func

EVENT_IDS = ["1208", "1508", "1608"]


def merge_events(folder, event_ids, dx=4.08):
    folder = Path(folder)
    events = []
    for event_id in event_ids:
        path = folder / f"resonance_frequencies_{event_id}.csv"
        if not path.exists():
            print(f"{event_id}: no frequency CSV available; skipping.")
            continue
        df = pd.read_csv(path)
        df["channel"] = df["channel"].round().astype(int)
        if df["channel"].duplicated().any():
            raise ValueError(f"Duplicate channels for {event_id}.")
        if not np.allclose(df["dist_m"], df["channel"] * dx):
            raise ValueError(f"Channel spacing DX does not match event {event_id}.")
        df = df.set_index("channel")
        # Do not bridge model-rejected regions using np.interp.
        model_path = folder / f"resonance_picks_model_case1_{event_id}.pkl"
        model_cols = ["water_depth_m", "H", "cs0", "nu"]
        for col in model_cols:
            df[col] = np.nan
        settings_path = folder / f"settings_{event_id}.json"
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
        raise FileNotFoundError("No frequency CSVs found in SAVE_DIR.")
    lo = max(df.index.min() for df in events)
    hi = min(df.index.max() for df in events)
    if hi < lo:
        raise ValueError("The events have no channels in common.")
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
                label="Median")
        # Standard deviation across events, not the uncertainty of individual clicks.
        center, std = merged[f"{mode}_median"], merged[f"{mode}_std"]
        ax.fill_between(merged["channel"], center - std, center + std,
                        color="0.5", alpha=0.2, label="± standard deviation across events")
        ax.set(xlabel="DAS channel", ylabel="Frequency [Hz]", title=mode)
        ax.legend(fontsize=8)
        fig.savefig(folder / f"merged_{mode}_events.png", dpi=250)
        plt.close(fig)


    folder = Path("/home/lea/Desktop/code/resonance_model/resonance_model/picking")
    model = pd.read_pickle(folder / "resonance_merged_events_median.pkl")

    rows = []
    for r in model.itertuples(index=False):
        values = (r.dist_km, r.water_depth_m, r.H_median,
                r.cs0_median, r.nu_median)
        if not np.all(np.isfinite(values)) or r.H_median <= 0:
            continue

        for depth in np.linspace(0, r.H_median, 80):
            rows.append({
                "x_km": r.dist_km,
                "z_m": -r.water_depth_m - depth,
                "vs_mps": r.cs0_median * max(depth, 0.1) ** r.nu_median,
            })

    vs2d = pd.DataFrame(rows)
    vs2d.to_pickle(folder / "vs_model_2d.pkl")
    vs2d.to_csv(folder / "vs_model_2d_merged.csv", index=False)
    if PLOT_MODEL:
        channels = merged["channel"].to_numpy()
        # Match the original merged plot: evaluate Vs from the median parameters.
        parameters = [merged[f"{name}_median"].to_numpy() for name in ("cs0", "nu", "H")]
        figures = func.plot_resonance_model(
            channels * DX, *parameters, dx=DX,
            title="Merged events | Vs model from median parameters",
            depth_max_m=MODEL_DEPTH_MAX_M, depth_samples=MODEL_DEPTH_SAMPLES,
            water_depth_m=merged["water_depth_m"].to_numpy(),
            reverse_distance=MODEL_REVERSE_DISTANCE, vs_limits=MODEL_VS_LIMITS)
        if figures is not None:
            for figure, suffix in zip(figures, ("vs_model", "vs_model_parameters")):
                figure.savefig(folder / f"merged_{suffix}.png", dpi=250)
            if SHOW_MODEL:
                plt.show()
            for figure in figures:
                plt.close(figure)
    print(f"Merged results exported to {folder}")


if __name__ == "__main__":
    main()
