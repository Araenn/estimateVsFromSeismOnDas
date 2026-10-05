"""DAS spectra, manual resonance picking and the shear-wave velocity model.

DAS reader dependencies are imported only when loading data.
Matrices returned by das_fx_spectrum_chunked have shape [channel, frequency].
Display and picking functions use shape [frequency, channel].
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, resample_poly, sosfiltfilt, welch


class PickingInterrupted(RuntimeError):
    """User-requested picking interruption; saved clicks remain available."""


def _safe_welch_params(n_samples, fs, nperseg_s, noverlap_s):
    if fs <= 0 or nperseg_s <= 0 or not 0 <= noverlap_s < nperseg_s:
        raise ValueError("Welch requires positive fs and duration, with 0 <= overlap < duration.")
    nperseg = min(int(round(nperseg_s * fs)), n_samples)
    if nperseg < 8:
        raise ValueError(f"Signal too short for Welch: {n_samples} samples")
    # Preserve the overlap fraction when the signal is shorter than requested.
    noverlap = min(int(round(nperseg * noverlap_s / nperseg_s)), nperseg - 1)
    return nperseg, noverlap


def preprocess_das_matrix(data, chx, fs_in=2000, fs_out=40,
                          fmin=0.3, fmax=19.0):
    """Use the same resampling and Butterworth filter as the original version."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("The DAS matrix must have two dimensions.")
    if data.shape[0] == len(chx):
        data_das = data
    elif data.shape[1] == len(chx):
        data_das = data.T
    else:
        raise ValueError(f"Cannot identify DAS axes: {data.shape}, {len(chx)} channels")
    if not 0 < fmin < fmax < fs_out / 2:
        raise ValueError("The filter requires 0 < fmin < fmax < fs_out/2.")
    if int(fs_in) != fs_in or int(fs_out) != fs_out:
        raise ValueError("fs_in and fs_out must be integers for resample_poly.")
    data_dec = resample_poly(data_das, int(fs_out), int(fs_in), axis=1)
    sos = butter(4, [fmin, fmax], btype="bandpass", fs=fs_out, output="sos")
    return sosfiltfilt(sos, data_dec, axis=1)


def spectra_from_matrix(data_filt, fs, fmin, fmax, nperseg_s=60,
                        noverlap_s=30, average="mean"):
    """Compute Welch PSD in linear power units; no spatial normalization is applied."""
    nperseg, noverlap = _safe_welch_params(
        data_filt.shape[1], fs, nperseg_s, noverlap_s)
    f, psd = welch(data_filt, fs=fs, window="hann", nperseg=nperseg,
                   noverlap=noverlap, detrend="constant", scaling="density",
                   average=average, axis=1)
    keep = (f >= fmin) & (f <= fmax)
    return f[keep], psd[:, keep], nperseg, noverlap


def das_fx_spectrum_chunked(t_start, t_end, chan_start_idx, chan_end_idx,
                            path_folder, chunk_channels=100, fs_in=2000,
                            fs_out=40, fmin=0.3, fmax=19.0,
                            nperseg_s=60, noverlap_s=30, average="mean"):
    """Load data as in the original project, then compute unnormalized PSDs.

    File selection is delegated to sensor_io_func.
    No common-mode removal or time-domain signal stacking is applied.
    """
    from sensor_io_func import get_das_files_for_window
    from simpleDASreader4 import load_multiple_DAS_files

    if chunk_channels < 1 or chan_end_idx < chan_start_idx:
        raise ValueError("Invalid channel range or channel chunk size.")
    folder, idfiles = get_das_files_for_window(t_start, t_end, path_folder)
    parts, channels = [], []
    f_ref, duration_ref, params_ref = None, None, None
    for ch0 in range(chan_start_idx, chan_end_idx + 1, chunk_channels):
        chx = np.arange(ch0, min(ch0 + chunk_channels, chan_end_idx + 1))
        print(f"Loading channels {chx[0]}–{chx[-1]}", flush=True)
        data, meta = load_multiple_DAS_files(folder, idfiles, chIndex=chx)
        filtered = preprocess_das_matrix(data, chx, fs_in, fs_out, fmin, fmax)
        f, psd, nperseg, noverlap = spectra_from_matrix(
            filtered, fs_out, fmin, fmax, nperseg_s, noverlap_s, average)
        duration = filtered.shape[1] / fs_out
        if f_ref is None:
            f_ref, duration_ref, params_ref = f, duration, (nperseg, noverlap)
            print(f"Loaded duration: {duration:.1f} s; Welch: "
                  f"{nperseg / fs_out:.1f} s; frequency spacing: {fs_out / nperseg:.5f} Hz")
        elif (f.shape != f_ref.shape or not np.allclose(f, f_ref)
              or duration != duration_ref or (nperseg, noverlap) != params_ref):
            raise ValueError("Frequency axes or durations differ between channel chunks.")
        parts.append(psd)
        channels.append(chx)
        del data, filtered
    return dict(chx=np.concatenate(channels), freqs=f_ref,
                psd=np.vstack(parts), fs=fs_out, duration_s=duration_ref,
                nperseg=params_ref[0], noverlap=params_ref[1])


def extract_fx_arrays(res_fx, dx=4.08):
    """Return frequency, distance and raw PSD in dB with shape [frequency, channel]."""
    f = np.asarray(res_fx["freqs"], dtype=float)
    chx = np.asarray(res_fx["chx"], dtype=float)
    psd = np.asarray(res_fx["psd"], dtype=float)
    if psd.shape != (len(chx), len(f)):
        raise ValueError("Cached PSD dimensions do not match the frequency and channel axes.")
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = 10 * np.log10(np.maximum(psd, 1e-30)).T
    return f, chx * dx, Z


def prepare_display(Z_db, f, normalize_band=(0.5, 9.0),
                    spatial_sigma_channels=3.0, freq_sigma_hz=0.0):
    """Normalize each channel, then apply smoothing for display only.

    The caller retains the raw PSD. Smoothing standard deviations are explicit:
    channels for space and Hz for frequency. Missing values remain missing.
    """
    Z = np.array(Z_db, dtype=float, copy=True)
    if Z.ndim != 2 or Z.shape[0] != len(f):
        raise ValueError("Z must have shape [frequency, channel].")
    band = (f >= normalize_band[0]) & (f <= normalize_band[1])
    if not np.any(band):
        raise ValueError("The normalization band is absent from the frequency axis.")
    finite_columns = np.any(np.isfinite(Z[band]), axis=0)
    offset = np.full(Z.shape[1], np.nan)
    offset[finite_columns] = np.nanmedian(Z[band][:, finite_columns], axis=0)
    raw = Z - offset[None, :]
    if spatial_sigma_channels < 0 or freq_sigma_hz < 0:
        raise ValueError("Smoothing standard deviations must be nonnegative.")
    df = float(np.median(np.diff(f)))
    sigma = (freq_sigma_hz / df, spatial_sigma_channels)
    valid = np.isfinite(raw)
    weights = gaussian_filter(valid.astype(float), sigma=sigma)
    values = gaussian_filter(np.where(valid, raw, 0.0), sigma=sigma)
    smooth = np.full_like(raw, np.nan)
    np.divide(values, weights, out=smooth, where=weights > 1e-12)
    smooth[~valid] = np.nan
    return raw, smooth


def color_limits(Z, f, band, percentiles=(5, 95)):
    selected = Z[(f >= band[0]) & (f <= band[1])]
    finite = selected[np.isfinite(selected)]
    if not finite.size:
        raise ValueError(f"No finite data are available in band {band}.")
    if not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("Percentiles must satisfy 0 <= pmin < pmax <= 100.")
    lo, hi = np.percentile(finite, percentiles)
    return (float(lo), float(hi)) if hi > lo else (float(lo - 1), float(hi + 1))


def plot_comparison(raw, smooth, f, x_chan, band=(0.5, 9.0),
                    percentiles=(5, 95), title="", save_path=None):
    """Show two views with identical color limits and no pick overlays."""
    lo, hi = color_limits(raw, f, band, percentiles)
    keep = (f >= band[0]) & (f <= band[1])
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, Z, label in zip(axes, (raw, smooth), ("No smoothing", "Light smoothing")):
        im = ax.pcolormesh(x_chan, f[keep], np.ma.masked_invalid(Z[keep]),
                          shading="nearest", cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(label)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_ylim(*band)
        ax.set_xlim(x_chan[0], x_chan[-1])
    axes[-1].set_xlabel("DAS channel")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, label="Relative PSD [dB / channel median]")
    if save_path:
        fig.savefig(save_path, dpi=250)
    return fig


def _sorted_picks(x, f):
    """Sort picks and merge duplicate positions to define interpolation uniquely."""
    x, f = np.asarray(x, dtype=float), np.asarray(f, dtype=float)
    valid = np.isfinite(x) & np.isfinite(f)
    x, f = x[valid], f[valid]
    unique = np.unique(x)
    return unique, np.array([np.median(f[x == xx]) for xx in unique])


def interpolate_manual_picks(x_picks, f_picks, x_target,
                             max_gap_channels=500, sigma_pts=0):
    """Linearly interpolate within picked regions; retain NaN across long gaps.

    No post-picking smoothing is applied: sigma_pts must remain zero.
    Set max_gap_channels to None to interpolate across all internal intervals.
    """
    if sigma_pts != 0:
        raise ValueError("Keep sigma_pts=0 for picks.")
    if max_gap_channels is not None and max_gap_channels <= 0:
        raise ValueError("max_gap_channels must be positive or None.")
    x, f = _sorted_picks(x_picks, f_picks)
    target = np.asarray(x_target, dtype=float)
    out = np.full(target.shape, np.nan)
    if not len(x):
        return out
    inside = (target >= x[0]) & (target <= x[-1])
    if len(x) > 1:
        out[inside] = np.interp(target[inside], x, f)
        if max_gap_channels is not None:
            for a, b in zip(x[:-1], x[1:]):
                if b - a > max_gap_channels:
                    out[(target > a) & (target < b)] = np.nan
    for xx, ff in zip(x, f):
        out[np.isclose(target, xx, rtol=0, atol=1e-9)] = ff
    return out


def manual_pick_curve(raw, smooth, f, x_chan, title="fs1", band=(0.5, 4.0),
                       segment_channels=1000, percentiles=(5, 95),
                       initial_picks=None, on_update=None):
    """Matplotlib picker with adjustable contrast, local spectra and immediate saving.

    Left click: add. Shift+left click: inspect only. Right click: remove the
    nearest visible pick. Enter: next segment. Close the window: interrupt.
    Save callbacks receive all picks, including those outside the current view.
    """
    from matplotlib.widgets import RangeSlider, RadioButtons, Button

    if "agg" == plt.get_backend().lower():
        raise RuntimeError("Picking requires an interactive backend (QtAgg/TkAgg).")
    if segment_channels <= 0:
        raise ValueError("segment_channels must be positive.")
    if raw.shape != smooth.shape or raw.shape != (len(f), len(x_chan)):
        raise ValueError("Picking map dimensions do not match.")
    keep = (f >= band[0]) & (f <= band[1])
    if np.count_nonzero(keep) < 2:
        raise ValueError("The picking band contains too few frequency samples.")
    lo, hi = color_limits(raw, f, band, percentiles)
    all_values = np.concatenate((raw[keep].ravel(), smooth[keep].ravel()))
    finite = all_values[np.isfinite(all_values)]
    slider_lo, slider_hi = float(finite.min()) - 1, float(finite.max()) + 1
    if initial_picks is None:
        picks = []
    else:
        px, pf = _sorted_picks(*initial_picks)
        picks = list(zip(px, pf))

    for start in np.arange(x_chan[0], x_chan[-1], segment_channels):
        end = min(start + segment_channels, x_chan[-1])
        # A one-channel margin keeps cell boundaries consistent between segments.
        dx = float(np.median(np.diff(x_chan)))
        spatial = (x_chan >= start - dx) & (x_chan <= end + dx)
        if np.count_nonzero(spatial) < 2:
            continue
        current = {"Z": smooth, "accepted": False}
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_axes([0.07, 0.26, 0.65, 0.63])
        spec_ax = fig.add_axes([0.79, 0.26, 0.18, 0.63], sharey=ax)
        im = ax.pcolormesh(x_chan[spatial], f[keep],
                          np.ma.masked_invalid(smooth[np.ix_(keep, spatial)]),
                          shading="nearest", cmap="viridis", vmin=lo, vmax=hi)
        ax.set(xlim=(start, end), ylim=band, xlabel="DAS channel",
               ylabel="Frequency [Hz]")
        points, = ax.plot([], [], "o", color="white", mec="black", ms=5)
        spec_ax.set(xlabel="Relative PSD [dB]", title="Click a channel")
        fig.suptitle(f"{title} | channels {start:g}–{end:g}")
        fig.text(0.07, 0.02,
                 "Left: add | Shift+left: inspect spectrum | "
                 "Right: remove | Enter: next | Close: interrupt",
                 fontsize=10)
        status = fig.text(0.07, 0.06, "")
        slider = RangeSlider(fig.add_axes([0.12, 0.16, 0.54, 0.03]),
                             "dB", slider_lo, slider_hi, valinit=(lo, hi))
        radio = RadioButtons(fig.add_axes([0.77, 0.07, 0.20, 0.12]),
                             ["No smoothing", "Light smoothing"], active=1)
        reset = Button(fig.add_axes([0.50, 0.07, 0.17, 0.05]), "Reset contrast")

        def redraw(save=False):
            if picks:
                px, pf = _sorted_picks(*zip(*picks))
            else:
                px, pf = np.array([]), np.array([])
            points.set_data(px, pf)
            status.set_text(f"{len(px)} picks for {title}; "
                            "saved after every change")
            if save and on_update is not None:
                on_update(px, pf)
            fig.canvas.draw_idle()

        def show_spectrum(ix):
            spec_ax.clear()
            spec_ax.plot(raw[keep, ix], f[keep], color="0.45", lw=1,
                         label="No smoothing")
            spec_ax.plot(smooth[keep, ix], f[keep], color="tab:blue", lw=1,
                         label="Light smoothing")
            spec_ax.set(xlabel="Relative PSD [dB]", title=f"Channel {x_chan[ix]:g}")
            spec_ax.set_ylim(*band)
            spec_ax.legend(fontsize=8)
            spec_ax.grid(alpha=0.15)

        def click(event):
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                return
            toolbar = getattr(fig.canvas.manager, "toolbar", None)
            if toolbar is not None and getattr(toolbar, "mode", ""):
                return
            ix = int(np.argmin(abs(x_chan - event.xdata)))
            if event.button == 1:
                show_spectrum(ix)
                if not np.any(np.isfinite(raw[keep, ix])):
                    status.set_text("No valid PSD for this channel: no pick added.")
                    fig.canvas.draw_idle()
                    return
                if event.key != "shift":
                    picks.append((float(event.xdata), float(event.ydata)))
                    redraw(save=True)
                else:
                    fig.canvas.draw_idle()
            elif event.button == 3:
                candidates = [i for i, (x, _) in enumerate(picks) if start <= x <= end]
                if candidates:
                    mouse = np.array([event.x, event.y])
                    distances = [np.linalg.norm(ax.transData.transform(picks[i]) - mouse)
                                 for i in candidates]
                    picks.pop(candidates[int(np.argmin(distances))])
                    redraw(save=True)

        def key(event):
            if event.key == "enter":
                current["accepted"] = True
                plt.close(fig)

        def change_view(label):
            current["Z"] = raw if label == "No smoothing" else smooth
            im.set_array(np.ma.masked_invalid(current["Z"][np.ix_(keep, spatial)]))
            fig.canvas.draw_idle()

        def change_clim(values):
            if values[1] > values[0]:
                im.set_clim(*values)
                fig.canvas.draw_idle()

        radio.on_clicked(change_view)
        slider.on_changed(change_clim)
        reset.on_clicked(lambda event: slider.reset())
        fig.canvas.mpl_connect("button_press_event", click)
        fig.canvas.mpl_connect("key_press_event", key)
        redraw()
        plt.show(block=True)
        if not current["accepted"]:
            raise PickingInterrupted("Picking interrupted. Existing clicks have been saved.")
    return _sorted_picks(*zip(*picks)) if picks else (np.array([]), np.array([]))


def compute_resonance_model(fs1, fs2, H=None, cs_avg=200.0):
    """
    Method from Taweesintananon et al.:
    nu = (6 fs1 - 2 fs2) / (5 fs1 - fs2)

    Case 1: H is unknown and estimated using cs_avg.
    Case 2: H is supplied and cs0 is calculated directly.
    """
    if not np.isfinite(fs1) or not np.isfinite(fs2):
        return None

    if fs1 <= 0 or fs2 <= fs1:
        return None

    if fs2 > 3 * fs1:
        return None

    denom = 5 * fs1 - fs2
    if np.isclose(denom, 0):
        return None

    nu = (6 * fs1 - 2 * fs2) / denom

    if not np.isfinite(nu) or nu < 0 or nu > 1:
        return None

    # Case 1: unknown H
    if H is None:
        # Equation 8 with n=1
        coeff = ((2 * 1 - 1) * (1 - nu**2) + 0.5 * nu * (1 + nu))
        H = coeff * cs_avg / (4 * fs1)

    if H <= 0 or not np.isfinite(H):
        return None

    # Equation 7 with n=1
    cs0 = 4 * (H ** (1 - nu)) * fs1 / (1 - 0.5 * nu)

    # Equation 5
    cs_avg_model = cs0 * (H ** nu) / (1 + nu)

    return {
        "fs1": fs1,
        "fs2": fs2,
        "nu": nu,
        "cs0": cs0,
        "H": H,
        "cs_avg_model": cs_avg_model,
        "cs_avg_assumed": cs_avg,
    }


def vs_power_law(z, cs0, nu):
    z = np.asarray(z)
    z_safe = np.maximum(z, 0.1)
    return cs0 * z_safe**nu


def plot_resonance_model(dist_m, cs0, nu, H, dx=4.08, title="",
                          depth_max_m=None, depth_samples=300,
                          water_depth_m=None, reverse_distance=True,
                          vs_limits=(0, 600)):
    """Plot one Vs model and its parameters without changing the model values.

    All inputs have shape [channel]. For the merged model, pass the saved
    cs0_median, nu_median and H_median, matching the original plot calculation.
    Vs is limited to H and missing channels remain blank. The existing 0.1 m
    depth floor in vs_power_law is retained. With bathymetry, plot elevation
    relative to sea level; otherwise plot depth below the seabed. The default
    cable orientation and color scale match the original model figure.
    Return two figures, or None when there is no valid section to plot.
    """
    dist_m = np.asarray(dist_m, dtype=float)
    parameters = [np.asarray(value, dtype=float) for value in (cs0, nu, H)]
    if (dist_m.ndim != 1 or not len(dist_m) or dx <= 0
            or not np.all(np.isfinite(dist_m))
            or not np.all(np.diff(dist_m) > 0)):
        raise ValueError("Model distances must be finite, increasing and nonempty; dx must be positive.")
    if any(value.shape != dist_m.shape for value in parameters):
        raise ValueError("cs0, nu and H must have shape [channel], matching dist_m.")
    if depth_samples < 2 or int(depth_samples) != depth_samples:
        raise ValueError("depth_samples must be an integer of at least two.")
    if len(vs_limits) != 2 or not np.all(np.isfinite(vs_limits)) or vs_limits[1] <= vs_limits[0]:
        raise ValueError("vs_limits must contain two increasing finite velocities.")
    if water_depth_m is not None:
        water_depth_m = np.asarray(water_depth_m, dtype=float)
        if water_depth_m.shape != dist_m.shape:
            raise ValueError("water_depth_m must have shape [channel], matching dist_m.")
        parameters.append(water_depth_m)
    channels = np.rint(dist_m / dx).astype(int)
    if not np.allclose(dist_m, channels * dx, rtol=0, atol=1e-6):
        raise ValueError("Model distances must align with the DAS channel spacing.")
    n_channels = channels[-1] - channels[0] + 1
    full_parameters = []
    for value in parameters:
        full = np.full(n_channels, np.nan)
        full[channels - channels[0]] = value
        full_parameters.append(full)
    cs0, nu, H = full_parameters[:3]
    valid = (np.isfinite(cs0) & np.isfinite(nu) & np.isfinite(H)
             & (cs0 > 0) & (nu >= 0) & (nu <= 1) & (H > 0))
    if not np.any(valid):
        print(f"{title}: no valid Vs model to plot.")
        return None
    if depth_max_m is None:
        depth_max_m = float(np.max(H[valid]))
    if not np.isfinite(depth_max_m) or depth_max_m <= 0:
        raise ValueError("depth_max_m must be positive and finite, or None.")
    z = np.linspace(0, depth_max_m, int(depth_samples))
    section = vs_power_law(z[:, None], cs0[None, :], nu[None, :])
    section = np.where(valid[None, :] & (z[:, None] <= H[None, :]), section, np.nan)
    centers = [np.where(valid, value, np.nan) for value in (H, cs0, nu)]
    x_km = np.arange(channels[0], channels[-1] + 1) * dx / 1000
    x_edges = (np.arange(channels[0], channels[-1] + 2) - 0.5) * dx / 1000
    z_edges = np.r_[0, (z[:-1] + z[1:]) / 2, depth_max_m]
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    if water_depth_m is None:
        y_edges = z_edges
        base = centers[0]
        ylabel, ylim = "Depth below seabed [m]", (depth_max_m, 0)
    else:
        water = full_parameters[3]
        located = valid & np.isfinite(water)
        if not np.any(located):
            print(f"{title}: no bathymetry available to locate the Vs model.")
            return None
        section[:, ~np.isfinite(water)] = np.nan
        # Only mesh coordinates are interpolated; missing Vs values stay masked.
        finite_water = np.isfinite(water)
        mesh_water = np.interp(x_km, x_km[finite_water], water[finite_water])
        water_edges = np.r_[mesh_water[0], (mesh_water[:-1] + mesh_water[1:]) / 2,
                            mesh_water[-1]]
        y_edges = -water_edges[None, :] - z_edges[:, None]
        base = -water - centers[0]
        ylabel = "Elevation relative to sea level [m]"
        bottom = np.min(-water[located] - np.minimum(H[located], depth_max_m))
        top = np.max(-water[located])
        margin = max((top - bottom) * 0.03, 0.1)
        ylim = (bottom - margin, top + margin)
    section = np.ma.masked_invalid(section)
    print(f"{title}: plotted Vs range = {section.min():.2f}–{section.max():.2f} m/s; "
          f"Vs at 1 m = {np.min(cs0[valid]):.2f}–{np.max(cs0[valid]):.2f} m/s.")
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    mesh_x = x_edges if water_depth_m is None else np.broadcast_to(x_edges, y_edges.shape)
    im = ax.pcolormesh(mesh_x, y_edges, section, shading="flat", cmap=cmap,
                       vmin=vs_limits[0], vmax=vs_limits[1], rasterized=True)
    if water_depth_m is not None:
        ax.plot(x_km, -water, "k", lw=1.2, label="Seabed")
    ax.plot(x_km, base, "k--", lw=1.2, label="Estimated layer base")
    ax.set(xlabel="Distance along cable [km]", ylabel=ylabel,
           title=title, xlim=(x_edges[0], x_edges[-1]), ylim=ylim)
    if reverse_distance:
        ax.invert_xaxis()
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, label="Estimated Vs [m/s]")

    parameter_fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True,
                                       constrained_layout=True)
    for ax, center, ylabel in zip(axes, centers,
                                  ("Layer thickness H [m]", "Vs at 1 m [m/s]", "Exponent nu")):
        ax.plot(x_km, center, color="tab:blue", lw=1.3)
        ax.set_ylabel(ylabel)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Distance along cable [km]")
    axes[-1].set_ylim(-0.02, 1.02)
    if reverse_distance:
        axes[-1].invert_xaxis()
    parameter_fig.suptitle(title + " | model parameters")
    return fig, parameter_fig
