#!/usr/bin/env python3
"""Merge trials on their common channels without filling long gaps.

Frequencies are read from picking CSVs. Physical parameters are read
from the same event PKL when BUILD_MODEL was enabled. Output filenames
remain compatible with the original plotting scripts.
Comparison figures show event curves, a thin dashed median and a standard
deviation band, including H, against kilometres decreasing from left to right.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from estimate_vs_from_seism import (SAVE_DIR, PLOT_MODEL, SHOW_MODEL,
                                   MODEL_DEPTH_MAX_M, MODEL_DEPTH_SAMPLES,
                                   MODEL_REVERSE_DISTANCE, MODEL_VS_LIMITS, analysis_identity)
import vs_functions as func

EVENT_IDS = None  # all analyses; a list can select day labels or complete timed analysis IDs


def discover_analyses(folder, event_ids=None):
    """Select windows independently and avoid counting a renamed legacy export twice."""
    folder = Path(folder)
    prefix = "resonance_frequencies_"
    available = {path.name[len(prefix):-4] for path in folder.glob(f"{prefix}*.csv")}
    if event_ids is None:
        selected = available
    else:
        selectors = [event_ids] if isinstance(event_ids, str) else event_ids
        selected = set()
        for selector in selectors:
            selector = str(selector)
            matches = {key for key in available if key == selector or key.startswith(selector + "_")}
            if not matches:
                print(f"{selector}: no frequency CSV available; skipping.")
            selected.update(matches)
    candidates = []
    for key in sorted(selected):
        settings_path = folder / f"settings_{key}.json"
        settings = (json.loads(settings_path.read_text(encoding="utf-8"))
                    if settings_path.exists() else {})
        title, identity, timed_export = key, None, False
        if "t_start" in settings and "t_end" in settings:
            label = settings.get("event_id", key)
            info = analysis_identity(label, settings["t_start"], settings["t_end"])
            title = info["analysis_title"]
            identity = (settings.get("path_folder"), info["analysis_id"])
            timed_export = key == info["analysis_id"]
        elif settings.get("analysis_title"):
            title = settings["analysis_title"]
        else:
            parts = key.rsplit("_", 2)
            if len(parts) == 3 and all(len(value) == 6 and value.isdigit() for value in parts[1:]):
                times = [[int(value[i:i + 2]) for i in (0, 2, 4)] for value in parts[1:]]
                title = analysis_identity(parts[0], *times)["analysis_title"]
        candidates.append((not timed_export, key, settings, title, identity))
    resolved, seen = [], set()
    for _, key, settings, title, identity in sorted(candidates, key=lambda item: (item[0], item[1])):
        if identity is not None and identity in seen:
            print(f"{key}: the same window already has a timed export; skipping the duplicate.")
            continue
        if identity is not None:
            seen.add(identity)
        resolved.append((key, settings, title))
    return sorted(resolved, key=lambda item: item[0])


def merge_events(folder, event_ids=None, dx=None):
    """Align events by physical distance, preserving unmeasured channels.

    Different spacings may be merged when their sensor positions align with
    the finest available grid. No frequency or model interpolation is used.
    """
    folder = Path(folder)
    loaded, spacings, distance_origins = [], [], []
    for event_id, settings, title in discover_analyses(folder, event_ids):
        path = folder / f"resonance_frequencies_{event_id}.csv"
        if not path.exists():
            print(f"{event_id}: no frequency CSV available; skipping.")
            continue
        df = pd.read_csv(path).sort_values("dist_m")
        if df.empty:
            continue
        if (not np.all(np.isfinite(df["dist_m"])) or df["dist_m"].duplicated().any()
                or df["channel"].duplicated().any()):
            raise ValueError(f"Invalid or duplicate channel positions for {event_id}.")
        distance_origin = settings.get("distance_origin_m", 0.0)
        if "distance_origin_m" in df:
            values = df["distance_origin_m"].to_numpy(dtype=float)
            if (not np.all(np.isfinite(values))
                    or not np.allclose(values, values[0], rtol=0, atol=1e-6)):
                raise ValueError(f"Inconsistent cable distance origin for {event_id}.")
            if "distance_origin_m" in settings and not np.isclose(
                    distance_origin, values[0], rtol=0, atol=1e-6):
                raise ValueError(f"The settings and CSV use different cable origins for {event_id}.")
            distance_origin = float(values[0])
        if not np.isfinite(distance_origin) or distance_origin < 0:
            raise ValueError(f"Invalid cable distance origin for {event_id}.")
        distance_origins.append(float(distance_origin))
        spacing = settings.get("dx")
        if spacing is None and "dx_m" in df:
            spacing = float(df["dx_m"].iloc[0])
        if spacing is None and len(df) > 1:
            spacing = float(np.min(np.diff(df["dist_m"])))
        if spacing is not None:
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError(f"Invalid channel spacing for {event_id}.")
            spacings.append(float(spacing))
        df["analysis_title"] = title
        loaded.append((event_id, df, settings))
    if not loaded:
        raise FileNotFoundError("No frequency CSVs found in SAVE_DIR.")
    if not np.allclose(distance_origins, distance_origins[0], rtol=0, atol=1e-6):
        raise ValueError("The events use different cable distance origins. "
                         "Export them with the same CABLE_ORIGIN_M before merging.")
    if dx is None:
        if not spacings:
            raise ValueError("Channel spacing is missing from the event settings and tables.")
        dx = min(spacings)
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("The merge spacing must be positive and finite.")
    positions = np.concatenate([df["dist_m"].to_numpy() for _, df, _ in loaded])
    origin = (0.0 if np.allclose(positions, np.rint(positions / dx) * dx, rtol=0, atol=1e-6)
              else float(np.min(positions)))

    def grid_indices(distances, event_id):
        distances = np.asarray(distances, dtype=float)
        indices = np.rint((distances - origin) / dx).astype(int)
        if not np.allclose(distances, origin + indices * dx, rtol=0, atol=1e-6):
            raise ValueError(f"{event_id}: sensor positions do not align with the common "
                             "physical grid. Process incompatible geometries separately.")
        return indices

    events = []
    for event_id, df, settings in loaded:
        df["source_channel"] = df["channel"]
        df["channel"] = grid_indices(df["dist_m"], event_id)
        df["dist_km"] = df["dist_m"] / 1000
        df = df.set_index("channel")
        model_path = folder / f"resonance_picks_model_case1_{event_id}.pkl"
        model_cols = ["water_depth_m", "H", "cs0", "nu"]
        for col in model_cols:
            df[col] = np.nan
        if settings.get("build_model", True) and model_path.exists():
            model = pd.read_pickle(model_path)
            if "distance_origin_m" in df and "distance_origin_m" not in model:
                raise ValueError(f"{event_id}: the model has no cable origin. "
                                 "Re-export the model with the current picking code.")
            if "distance_origin_m" in model and not np.allclose(
                    model["distance_origin_m"], distance_origins[0], rtol=0, atol=1e-6):
                raise ValueError(f"{event_id}: the model and frequencies use different cable origins.")
            model["channel"] = grid_indices(model["dist_m"], event_id)
            if model["channel"].duplicated().any():
                raise ValueError(f"Duplicate model positions for {event_id}.")
            aligned = model.set_index("channel").reindex(df.index)
            agrees = (np.isclose(df["fs1"], aligned["fs1"], rtol=1e-8, atol=1e-8)
                      & np.isclose(df["fs2"], aligned["fs2"], rtol=1e-8, atol=1e-8))
            for col in model_cols:
                df[col] = aligned[col].where(agrees)
        df["event_id"] = event_id
        df["analysis_id"] = event_id
        df["valid_fs1"] = np.isfinite(df["fs1"])
        df["valid_fs2"] = np.isfinite(df["fs2"])
        df["valid_model"] = np.isfinite(df["H"])
        events.append(df)
    lo = max(df.index.min() for df in events)
    hi = min(df.index.max() for df in events)
    if hi < lo:
        raise ValueError("The events have no distances in common.")
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
    merged = all_events.groupby("channel").agg(**aggregations).reindex(range(lo, hi + 1))
    for col in ("n_events", "n_events_fs1", "n_events_fs2"):
        merged[col] = merged[col].fillna(0).astype(int)
    merged = merged.reset_index()
    merged["dist_m"] = origin + merged["channel"] * dx
    merged["dist_km"] = merged["dist_m"] / 1000
    merged["dx_m"] = dx
    merged["distance_origin_m"] = distance_origins[0]
    merged["fs2_fs1_ratio"] = merged["fs2_median"] / merged["fs1_median"]
    merged.attrs.update(dx=float(dx), distance_origin_m=origin,
                        cable_origin_m=distance_origins[0])
    return all_events, merged


def plot_merged_quantity(ax, all_events, merged, quantity, dx, ylabel):
    """Draw the original event comparison with median ± sample standard deviation.

    The distance axis is in kilometres, with higher distances on the left.
    The band describes variation across analyses; it is undefined
    where fewer than two analyses contribute, and gaps remain empty.
    """
    for _, df in all_events.groupby("event_id"):
        ax.plot(df["dist_m"] / 1000, df[quantity], lw=1, alpha=0.4,
                label=f"{quantity} {df['analysis_title'].iloc[0]}")
    x = merged["dist_m"].to_numpy(dtype=float) / 1000
    center = merged[f"{quantity}_median"].to_numpy(dtype=float)
    std = merged[f"{quantity}_std"].to_numpy(dtype=float)
    ax.plot(x, center, color="0.25", ls="--", lw=1,
            label=f"{quantity} median")
    ax.fill_between(x, center - std, center + std, color="0.5", alpha=0.2,
                    label=f"{quantity} ± std across analyses")
    # Keep the full measured section rather than clipping its final sensors.
    left, right = x[-1], x[0]
    if left == right:
        left, right = left + dx / 2000, right - dx / 2000
    ax.set(xlabel="Distance along cable [km]", ylabel=ylabel, xlim=(left, right))
    ax.grid()
    ax.legend(fontsize=8)


def main():
    folder = Path(SAVE_DIR)
    all_events, merged = merge_events(folder, EVENT_IDS)
    plot_dx = merged.attrs["dx"]
    all_events.to_csv(folder / "resonance_all_events_interpolated.csv", index=False)
    merged.to_csv(folder / "resonance_merged_events_median.csv", index=False)
    merged.to_pickle(folder / "resonance_merged_events_median.pkl")
    figures_to_show = []
    comparisons = [("fs1", "Frequency [Hz]", "fs1"),
                   ("fs2", "Frequency [Hz]", "fs2")]
    if np.isfinite(merged["H_median"]).any():
        comparisons.append(("H", "Estimated LVL thickness H [m]", "LVL thickness"))
    for quantity, ylabel, label in comparisons:
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_merged_quantity(ax, all_events, merged, quantity, plot_dx, ylabel)
        ax.set_title(f"Merged {label} from multiple analyses")
        fig.tight_layout()
        fig.savefig(folder / f"merged_{quantity}_events.png", dpi=300)
        figures_to_show.append(fig)
    if PLOT_MODEL:
        # Match the original merged plot: evaluate Vs from the median parameters.
        parameters = [merged[f"{name}_median"].to_numpy() for name in ("cs0", "nu", "H")]
        figures = func.plot_resonance_model(
            merged["dist_m"].to_numpy(), *parameters, dx=plot_dx,
            title="Merged events | Vs model from median parameters",
            depth_max_m=MODEL_DEPTH_MAX_M, depth_samples=MODEL_DEPTH_SAMPLES,
            water_depth_m=merged["water_depth_m"].to_numpy(),
            reverse_distance=MODEL_REVERSE_DISTANCE, vs_limits=MODEL_VS_LIMITS)
        if figures is not None:
            # Restore event curves and spread on the parameter comparison too.
            # The Vs section above still uses the unchanged median parameters.
            for ax, quantity, ylabel in zip(
                    figures[1].axes, ("H", "cs0", "nu"),
                    ("Layer thickness H [m]", "Vs at 1 m [m/s]", "Exponent nu")):
                ax.clear()
                plot_merged_quantity(ax, all_events, merged, quantity, plot_dx, ylabel)
            figures[1].axes[-1].set_ylim(-0.02, 1.02)
            for figure, suffix in zip(figures, ("vs_model", "vs_model_parameters")):
                figure.savefig(folder / f"merged_{suffix}.png", dpi=300)
            figures_to_show.extend(figures)
    if SHOW_MODEL:
        plt.show()
    for figure in figures_to_show:
        plt.close(figure)
    print(f"Merged results exported to {folder}")


if __name__ == "__main__":
    main()
