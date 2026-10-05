import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAVE_DIR = "/home/lea/Desktop/code/thesis/das_datas/resonance_model"
DX = 4.08

event_ids = ["1208", "1508", "1608"]

dfs = []

for event_id in event_ids:
    path = os.path.join(
        SAVE_DIR,
        f"resonance_picks_model_case1_{event_id}.pkl"
    )

    df = pd.read_pickle(path)
    df["event_id"] = event_id
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

dx_interp = DX

x_min = max(df.groupby("event_id")["dist_m"].min())
x_max = min(df.groupby("event_id")["dist_m"].max())

x_grid = np.arange(x_min, x_max + dx_interp, dx_interp)

rows = []

for event_id, df_ev in df_all.groupby("event_id"):

    df_ev = df_ev.sort_values("dist_m")

    out = pd.DataFrame({
        "event_id": event_id,
        "dist_m": x_grid,
        "dist_km": x_grid / 1000,
    })

    for col in ["fs1", "fs2", "water_depth_m", "H", "cs0", "nu"]:
        if col in df_ev.columns:
            out[col] = np.interp(
                x_grid,
                df_ev["dist_m"].values,
                df_ev[col].values
            )

    rows.append(out)

df_grid_all = pd.concat(rows, ignore_index=True)

df_merged = (
    df_grid_all
    .groupby("dist_m")
    .agg(
        dist_km=("dist_km", "first"),
        water_depth_m=("water_depth_m", "median"),

        fs1_median=("fs1", "median"),
        fs1_mean=("fs1", "mean"),
        fs1_std=("fs1", "std"),

        fs2_median=("fs2", "median"),
        fs2_mean=("fs2", "mean"),
        fs2_std=("fs2", "std"),

        H_median=("H", "median"),
        H_mean=("H", "mean"),
        H_std=("H", "std"),

        cs0_median=("cs0", "median"),
        cs0_std=("cs0", "std"),

        nu_median=("nu", "median"),
        nu_std=("nu", "std"),

        n_events=("event_id", "nunique"),
    )
    .reset_index()
)

df_merged["fs2_fs1_ratio"] = (
    df_merged["fs2_median"] / df_merged["fs1_median"]
)

df_grid_all.to_csv(
    os.path.join(SAVE_DIR, "resonance_all_events_interpolated.csv"),
    index=False
)

df_merged.to_csv(
    os.path.join(SAVE_DIR, "resonance_merged_events_median.csv"),
    index=False
)

df_merged.to_pickle(
    os.path.join(SAVE_DIR, "resonance_merged_events_median.pkl")
)

x_chan = df_merged["dist_m"] / DX

plt.figure(figsize=(15, 5))

for event_id, df_ev in df_grid_all.groupby("event_id"):
    plt.plot(
        df_ev["dist_m"] / DX,
        df_ev["fs1"],
        lw=1,
        alpha=0.4,
        label=f"fs1 {event_id}"
    )

plt.plot(
    x_chan,
    df_merged["fs1_median"],
    "k",
    lw=2.5,
    label="fs1 median"
)

plt.fill_between(
    x_chan,
    df_merged["fs1_median"] - df_merged["fs1_std"],
    df_merged["fs1_median"] + df_merged["fs1_std"],
    alpha=0.2,
    label="fs1 ± std"
)

plt.xlabel("DAS channel index")
plt.ylabel("Frequency [Hz]")
plt.title("Merged fs1 from multiple earthquakes")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "merged_fs1_events.png"), dpi=300)
plt.show()

plt.figure(figsize=(15, 5))

for event_id, df_ev in df_grid_all.groupby("event_id"):
    plt.plot(
        df_ev["dist_m"] / DX,
        df_ev["fs2"],
        lw=1,
        alpha=0.4,
        label=f"fs2 {event_id}"
    )

plt.plot(
    x_chan,
    df_merged["fs2_median"],
    "k",
    lw=2.5,
    label="fs2 median"
)

plt.fill_between(
    x_chan,
    df_merged["fs2_median"] - df_merged["fs2_std"],
    df_merged["fs2_median"] + df_merged["fs2_std"],
    alpha=0.2,
    label="fs2 ± std"
)

plt.xlabel("DAS channel index")
plt.ylabel("Frequency [Hz]")
plt.title("Merged fs2 from multiple earthquakes")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "merged_fs2_events.png"), dpi=300)
plt.show()

plt.figure(figsize=(15, 5))

for event_id, df_ev in df_grid_all.groupby("event_id"):
    plt.plot(
        df_ev["dist_m"] / DX,
        df_ev["H"],
        lw=1,
        alpha=0.4,
        label=f"H {event_id}"
    )

plt.plot(
    x_chan,
    df_merged["H_median"],
    "k",
    lw=2.5,
    label="H median"
)

plt.fill_between(
    x_chan,
    df_merged["H_median"] - df_merged["H_std"],
    df_merged["H_median"] + df_merged["H_std"],
    alpha=0.2,
    label="H ± std"
)

plt.xlabel("DAS channel index")
plt.ylabel("Estimated LVL thickness H [m]")
plt.title("Merged LVL thickness from multiple earthquakes")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "merged_H_events.png"), dpi=300)
plt.show()