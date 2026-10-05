# DAS Resonance Picking and Vs Modelling

Compute OptoDAS frequency–distance spectra, manually pick `fs1`/`fs2`, estimate a shear-wave velocity model, and merge multiple events. Based on Taweesintananon et al (Taweesintananon, Kittinat, Rørstadbotnen, Robin André, Landrø, Martin, Johansen, Ståle Emil, Arntsen, Børge, Forwick, Matthias, Hanssen, Alfred (2024) Near-surface characterization using shear-wave resonances: A case study from offshore Svalbard, Norway. GEOPHYSICS, 89 (4). doi:10.1190/geo2023-0530.1)

Earthquakes times are picked based on the https://www.jordskjelv.no/ database.

## Setup

Keep the scripts together, using the filenames below without download suffixes such as `(2)`.

| File | Purpose |
| --- | --- |
| `estimate_vs_from_seism.py` | Process and pick one event. |
| `merge_resonance_events.py` | Merge events and plot median results. |
| `vs_functions.py` | Processing, picking and model functions. |
| `simpleDASreader4.py`, `sensor_io_func.py`, `geo_functions.py` | Reading, file selection and cable geometry. |

Install the Python dependencies:

```bash
python -m pip install numpy scipy pandas matplotlib xarray pyproj netCDF4 h5py PySide6
```

Also provide **ASN's `h5pydict` module** on Python's import path. `espg3996tomercator.py` is an optional raster utility requiring `rasterio`; the pipeline does not use it.

Picking requires a graphical display and an interactive Matplotlib backend (QtAgg or TkAgg).

## Process an event

Edit the parameter block in `estimate_vs_from_seism.py`:

| Parameters | Set to |
| --- | --- |
| `DAS_FOLDER`, `SAVE_DIR` | Input recording directory and output directory. |
| `EVENT_ID`, `T_START`, `T_END` | Event label and same-day `[hour, minute, second]` window. The last active assignment applies. |
| `CHAN_START`, `CHAN_END`, `DX` | Channel indices (both bounds included) and spacing in metres. |
| `FS_IN`, `FS_OUT` | Recording and processing sampling rates in Hz. |
| `BATHY_PATH`, `CABLE_CSV` | Bathymetry and cable geometry for the Vs model. |

Inputs: `HHMMSS.hdf5` DAS files; NetCDF bathymetry with `z(x,y)` in EPSG:3996 and negative seabed elevations; a cable CSV with a header and semicolon-separated latitude/longitude columns in EPSG:4326.

```bash
python estimate_vs_from_seism.py
```

Close the preview, then pick `fs1` followed by `fs2` in successive channel segments:

| Action | Control |
| --- | --- |
| Add a pick | Left click |
| Inspect the local spectrum | Shift + left click |
| Remove the nearest visible pick | Right click |
| Advance to the next segment | Enter |
| Interrupt picking | Close the picking window |

Clicks are saved immediately. Use the contrast slider and raw/smoothed view selector. Smoothing affects display only. Picks are interpolated without extrapolation or bridging gaps beyond `MAX_INTERP_GAP_CHANNELS`.

`REUSE_CACHE=True` reuses compatible spectra; `REUSE_PICKS=True` reloads compatible picks. `DO_MANUAL_PICKING=False` exports saved picks without editing. For a different calculation, use a new event label/output directory or set `REUSE_PICKS=False`.

## Model and merge

`BUILD_MODEL=True` estimates thickness `H` and `Vs(z) = cs0 * max(z, 0.1)**nu`, using the assumed mean velocity `CS_AVG`. Depth is in metres below the seabed; `cs0` is Vs at **1 m**, not the whole-layer velocity range. Invalid resonance pairs are excluded.

`PLOT_MODEL` exports model figures; `SHOW_MODEL` displays them. `MODEL_REVERSE_DISTANCE=True` places higher cable distances on the left. `MODEL_VS_LIMITS` controls the color scale.

Process each event with the same output directory and channel spacing. Set `EVENT_IDS` in `merge_resonance_events.py`, then run:

```bash
python merge_resonance_events.py
```

The merge uses the main script's output directory and plotting settings, preserves missing values over the common channel range, and plots median `cs0`, `nu` and `H`. Frequency shading is standard deviation across events.

## Outputs

`SAVE_DIR` contains PSD/click caches (`.npz`), settings (`.json`), frequency/model tables (`.csv`/`.pkl`) and figures (`.png`). Merged outputs use `resonance_merged_events_median` and `merged_*` filenames. The supplied `.gitignore` excludes data and generated outputs.
