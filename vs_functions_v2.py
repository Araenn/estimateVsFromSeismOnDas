"""Calcul DAS et picking manuel de fs1/fs2, version 2.

Les dépendances spécifiques au projet DAS sont importées à la lecture seulement.
Les matrices retournées par das_fx_spectrum_chunked sont [canal, fréquence].
Les fonctions d'affichage et de picking utilisent [fréquence, canal].
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, resample_poly, sosfiltfilt, welch


class PickingInterrupted(RuntimeError):
    """Arrêt volontaire du picking; les clics sauvegardés restent disponibles."""


def _safe_welch_params(n_samples, fs, nperseg_s, noverlap_s):
    if fs <= 0 or nperseg_s <= 0 or not 0 <= noverlap_s < nperseg_s:
        raise ValueError("Welch: fs et durée positifs, 0 <= recouvrement < durée.")
    nperseg = min(int(round(nperseg_s * fs)), n_samples)
    if nperseg < 8:
        raise ValueError(f"Signal trop court pour Welch: {n_samples} échantillons")
    # Conserver la fraction de recouvrement si le signal est trop court.
    noverlap = min(int(round(nperseg * noverlap_s / nperseg_s)), nperseg - 1)
    return nperseg, noverlap


def preprocess_das_matrix(data, chx, fs_in=2000, fs_out=40,
                          fmin=0.3, fmax=19.0):
    """Même rééchantillonnage et filtre Butterworth que la version originale."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("La matrice DAS doit avoir deux dimensions.")
    if data.shape[0] == len(chx):
        data_das = data
    elif data.shape[1] == len(chx):
        data_das = data.T
    else:
        raise ValueError(f"Axes DAS indéterminés: {data.shape}, {len(chx)} canaux")
    if not 0 < fmin < fmax < fs_out / 2:
        raise ValueError("Le filtre doit respecter 0 < fmin < fmax < fs_out/2.")
    if int(fs_in) != fs_in or int(fs_out) != fs_out:
        raise ValueError("fs_in et fs_out doivent être entiers pour resample_poly.")
    data_dec = resample_poly(data_das, int(fs_out), int(fs_in), axis=1)
    sos = butter(4, [fmin, fmax], btype="bandpass", fs=fs_out, output="sos")
    return sosfiltfilt(sos, data_dec, axis=1)


def spectra_from_matrix(data_filt, fs, fmin, fmax, nperseg_s=60,
                        noverlap_s=30, average="mean"):
    """Welch en puissance linéaire; aucune normalisation spatiale ici."""
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
    """Lecture identique au projet original, puis PSD non normalisées.

    La sélection précise des fichiers reste celle de sensor_io_func.
    Aucun retrait de mode commun, aucun stack de signaux temporels.
    """
    from functions_modules.sensor_io_func import get_das_files_for_window
    from simpleDASreader4 import load_multiple_DAS_files

    if chunk_channels < 1 or chan_end_idx < chan_start_idx:
        raise ValueError("Plage de canaux ou taille des chunks invalide.")
    folder, idfiles = get_das_files_for_window(t_start, t_end, path_folder)
    parts, channels = [], []
    f_ref, duration_ref, params_ref = None, None, None
    for ch0 in range(chan_start_idx, chan_end_idx + 1, chunk_channels):
        chx = np.arange(ch0, min(ch0 + chunk_channels, chan_end_idx + 1))
        print(f"Lecture canaux {chx[0]}–{chx[-1]}", flush=True)
        data, meta = load_multiple_DAS_files(folder, idfiles, chIndex=chx)
        filtered = preprocess_das_matrix(data, chx, fs_in, fs_out, fmin, fmax)
        f, psd, nperseg, noverlap = spectra_from_matrix(
            filtered, fs_out, fmin, fmax, nperseg_s, noverlap_s, average)
        duration = filtered.shape[1] / fs_out
        if f_ref is None:
            f_ref, duration_ref, params_ref = f, duration, (nperseg, noverlap)
            print(f"Durée chargée: {duration:.1f} s; Welch: "
                  f"{nperseg / fs_out:.1f} s; pas: {fs_out / nperseg:.5f} Hz")
        elif (f.shape != f_ref.shape or not np.allclose(f, f_ref)
              or duration != duration_ref or (nperseg, noverlap) != params_ref):
            raise ValueError("Axes fréquentiels ou durées différents entre chunks.")
        parts.append(psd)
        channels.append(chx)
        del data, filtered
    return dict(chx=np.concatenate(channels), freqs=f_ref,
                psd=np.vstack(parts), fs=fs_out, duration_s=duration_ref,
                nperseg=params_ref[0], noverlap=params_ref[1])


def extract_fx_arrays(res_fx, dx=4.08):
    """Retourne fréquence, distance et PSD brute en dB [fréquence, canal]."""
    f = np.asarray(res_fx["freqs"], dtype=float)
    chx = np.asarray(res_fx["chx"], dtype=float)
    psd = np.asarray(res_fx["psd"], dtype=float)
    if psd.shape != (len(chx), len(f)):
        raise ValueError("Dimensions du cache PSD incompatibles avec ses axes.")
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = 10 * np.log10(np.maximum(psd, 1e-30)).T
    return f, chx * dx, Z


def prepare_display(Z_db, f, normalize_band=(0.5, 9.0),
                    spatial_sigma_channels=3.0, freq_sigma_hz=0.0):
    """Normalisation par canal, puis lissage d'affichage seulement.

    La PSD brute est conservée par l'appelant. Les sigma sont explicites:
    canaux pour l'espace, Hz pour la fréquence. Pas de remplissage des trous.
    """
    Z = np.array(Z_db, dtype=float, copy=True)
    if Z.ndim != 2 or Z.shape[0] != len(f):
        raise ValueError("Z doit être [fréquence, canal].")
    band = (f >= normalize_band[0]) & (f <= normalize_band[1])
    if not np.any(band):
        raise ValueError("Bande de normalisation absente de l'axe fréquentiel.")
    finite_columns = np.any(np.isfinite(Z[band]), axis=0)
    offset = np.full(Z.shape[1], np.nan)
    offset[finite_columns] = np.nanmedian(Z[band][:, finite_columns], axis=0)
    raw = Z - offset[None, :]
    if spatial_sigma_channels < 0 or freq_sigma_hz < 0:
        raise ValueError("Les sigma de lissage doivent être positifs ou nuls.")
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
        raise ValueError(f"Aucune donnée finie dans la bande {band}.")
    if not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("Percentiles attendus: 0 <= pmin < pmax <= 100.")
    lo, hi = np.percentile(finite, percentiles)
    return (float(lo), float(hi)) if hi > lo else (float(lo - 1), float(hi + 1))


def plot_comparison(raw, smooth, f, x_chan, band=(0.5, 9.0),
                    percentiles=(5, 95), title="", save_path=None):
    """Deux vues de même contraste, sans surimpression des picks."""
    lo, hi = color_limits(raw, f, band, percentiles)
    keep = (f >= band[0]) & (f <= band[1])
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, Z, label in zip(axes, (raw, smooth), ("Sans lissage", "Lissage léger")):
        im = ax.pcolormesh(x_chan, f[keep], np.ma.masked_invalid(Z[keep]),
                          shading="nearest", cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(label)
        ax.set_ylabel("Fréquence [Hz]")
        ax.set_ylim(*band)
        ax.set_xlim(x_chan[0], x_chan[-1])
    axes[-1].set_xlabel("Canal DAS")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, label="PSD relative [dB / médiane du canal]")
    if save_path:
        fig.savefig(save_path, dpi=250)
    return fig


def _sorted_picks(x, f):
    """Tri et fusion de clics à la même abscisse pour une interpolation définie."""
    x, f = np.asarray(x, dtype=float), np.asarray(f, dtype=float)
    valid = np.isfinite(x) & np.isfinite(f)
    x, f = x[valid], f[valid]
    unique = np.unique(x)
    return unique, np.array([np.median(f[x == xx]) for xx in unique])


def interpolate_manual_picks(x_picks, f_picks, x_target,
                             max_gap_channels=500, sigma_pts=0):
    """Interpolation linéaire limitée aux zones pickées; NaN dans les longs trous.

    Aucun lissage après picking dans cette version: sigma_pts doit rester nul.
    max_gap_channels peut être None pour interpoler tous les intervalles internes.
    """
    if sigma_pts != 0:
        raise ValueError("Conserver sigma_pts=0 pour les picks V2.")
    if max_gap_channels is not None and max_gap_channels <= 0:
        raise ValueError("max_gap_channels doit être positif ou None.")
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
    """Picker Matplotlib: contraste réglable, spectre local, sauvegarde immédiate.

    Gauche: ajouter. Shift+gauche: examiner sans ajouter. Droite: retirer le
    point visible le plus proche. Entrée: tronçon suivant. Fermer: interrompre.
    Les callbacks de sauvegarde reçoivent tous les points, y compris hors vue.
    """
    from matplotlib.widgets import RangeSlider, RadioButtons, Button

    if "agg" == plt.get_backend().lower():
        raise RuntimeError("Picking: backend interactif requis (QtAgg/TkAgg).")
    if segment_channels <= 0:
        raise ValueError("segment_channels doit être positif.")
    if raw.shape != smooth.shape or raw.shape != (len(f), len(x_chan)):
        raise ValueError("Dimensions des cartes de picking incompatibles.")
    keep = (f >= band[0]) & (f <= band[1])
    if np.count_nonzero(keep) < 2:
        raise ValueError("La bande de picking contient trop peu de fréquences.")
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
        # Un canal de marge rend les bords cohérents entre tronçons.
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
        ax.set(xlim=(start, end), ylim=band, xlabel="Canal DAS",
               ylabel="Fréquence [Hz]")
        points, = ax.plot([], [], "o", color="white", mec="black", ms=5)
        spec_ax.set(xlabel="PSD relative [dB]", title="Cliquer un canal")
        fig.suptitle(f"{title} | canaux {start:g}–{end:g}")
        fig.text(0.07, 0.02,
                 "Gauche : ajouter | Shift+gauche : spectre seul | "
                 "Droite : retirer | Entrée : suite | fermer : interrompre",
                 fontsize=10)
        status = fig.text(0.07, 0.06, "")
        slider = RangeSlider(fig.add_axes([0.12, 0.16, 0.54, 0.03]),
                             "dB", slider_lo, slider_hi, valinit=(lo, hi))
        radio = RadioButtons(fig.add_axes([0.77, 0.07, 0.20, 0.12]),
                             ["Sans lissage", "Lissage léger"], active=1)
        reset = Button(fig.add_axes([0.50, 0.07, 0.17, 0.05]), "Contraste initial")

        def redraw(save=False):
            if picks:
                px, pf = _sorted_picks(*zip(*picks))
            else:
                px, pf = np.array([]), np.array([])
            points.set_data(px, pf)
            status.set_text(f"{len(px)} points pour {title}; "
                            "sauvegarde à chaque modification")
            if save and on_update is not None:
                on_update(px, pf)
            fig.canvas.draw_idle()

        def show_spectrum(ix):
            spec_ax.clear()
            spec_ax.plot(raw[keep, ix], f[keep], color="0.45", lw=1,
                         label="Sans lissage")
            spec_ax.plot(smooth[keep, ix], f[keep], color="tab:blue", lw=1,
                         label="Lissage léger")
            spec_ax.set(xlabel="PSD relative [dB]", title=f"Canal {x_chan[ix]:g}")
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
                    status.set_text("Canal sans PSD valide: aucun point ajouté.")
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
            current["Z"] = raw if label == "Sans lissage" else smooth
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
            raise PickingInterrupted("Picking interrompu. Les clics déjà effectués sont sauvegardés.")
    return _sorted_picks(*zip(*picks)) if picks else (np.array([]), np.array([]))


def compute_resonance_model(fs1, fs2, H=None, cs_avg=200.0):
    """
    Méthode Taweesintananon et al. :
    nu = (6 fs1 - 2 fs2) / (5 fs1 - fs2)

    Case 1 : H inconnu -> estimé avec cs_avg.
    Case 2 : H fourni -> calcul direct cs0.
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

    # Case 1 : H inconnu
    if H is None:
        # équation 8 avec n=1
        coeff = ((2 * 1 - 1) * (1 - nu**2) + 0.5 * nu * (1 + nu))
        H = coeff * cs_avg / (4 * fs1)

    if H <= 0 or not np.isfinite(H):
        return None

    # équation 7 avec n=1
    cs0 = 4 * (H ** (1 - nu)) * fs1 / (1 - 0.5 * nu)

    # équation 5
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
