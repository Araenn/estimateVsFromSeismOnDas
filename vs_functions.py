
import os
import numpy as np
import matplotlib.pyplot as plt
from functions_modules.sensor_io_func import get_das_files_for_window
from simpleDASreader4 import load_DAS_file, load_multiple_DAS_files
import obs_data.obs_functions as obs_func
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from scipy.signal import (
    resample_poly,
    butter,
    sosfiltfilt,
    welch,
    spectrogram,
)


def _safe_welch_params(n_samples, fs, nperseg_s, noverlap_s):
    nperseg = int(nperseg_s * fs)
    noverlap = int(noverlap_s * fs)

    nperseg = min(nperseg, n_samples)

    if nperseg < 8:
        raise ValueError(f"Signal trop court pour Welch: n_samples={n_samples}")

    noverlap = min(noverlap, nperseg - 1)

    return nperseg, noverlap


def preprocess_das_matrix(
    data,
    chx,
    fs_in=2000,
    fs_out=50,
    fmin=0.1,
    fmax=20.0,
    remove_common=False,
):
    """
    Entrée attendue possible:
        data shape = (n_samples, n_channels)
        ou (n_channels, n_samples)

    Sortie:
        data_filt shape = (n_channels, n_samples_decimated)
    """

    print("raw data shape:", data.shape)

    if data.shape[0] == len(chx):
        data_das = data
    elif data.shape[1] == len(chx):
        data_das = data.T
    else:
        raise ValueError(
            f"Impossible de déterminer axes channel/time. "
            f"data.shape={data.shape}, len(chx)={len(chx)}"
        )

    data_das = np.asarray(data_das, dtype=np.float64)

    print("resample...")
    data_dec = resample_poly(data_das, fs_out, fs_in, axis=1)
    print("resampled shape:", data_dec.shape)

    if remove_common:
        print("remove common mode...")
        common = np.nanmedian(data_dec, axis=0)
        data_dec = data_dec - common[None, :]

    sos = butter(
        4,
        [fmin, fmax],
        btype="bandpass",
        fs=fs_out,
        output="sos",
    )

    data_filt = sosfiltfilt(sos, data_dec, axis=1)

    return data_filt




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, butter, sosfiltfilt, welch
from scipy.ndimage import median_filter

def das_fx_spectrum_chunked(
    t_start,
    t_end,
    chan_start_idx,
    chan_end_idx,
    chunk_channels=250,
    first_channel=5000,
    path_folder="/media/lea/Expansion/DAS/20250813/dphi",
    fs_in=2000,
    fs_out=20,
    fmin=0.2,
    fmax=8.0,
    nperseg_s=60,
    noverlap_s=30,
    remove_common=False,
    robust_normalize=True,
    plot=True,
):
    folder, idfiles = get_das_files_for_window(
        t_start,
        t_end,
        path_folder,
    )

    psd_list = []
    ch_list = []
    freqs_ref = None

    for ch0 in range(chan_start_idx, chan_end_idx + 1, chunk_channels):

        ch1 = min(ch0 + chunk_channels - 1, chan_end_idx)
        chx = np.arange(ch0, ch1 + 1, dtype=int)

        print(f"Chunk {chx[0]}-{chx[-1]}")

        data, meta = load_multiple_DAS_files(
            folder,
            idfiles,
            chIndex=chx,
        )

        # EXACTEMENT comme original
        data_filt = preprocess_das_matrix(
            data=data,
            chx=chx,
            fs_in=fs_in,
            fs_out=fs_out,
            fmin=fmin,
            fmax=fmax,
            remove_common=remove_common,
        )

        n_samples = data_filt.shape[1]
        nperseg, noverlap = _safe_welch_params(
            n_samples=n_samples,
            fs=fs_out,
            nperseg_s=nperseg_s,
            noverlap_s=noverlap_s,
        )

        freqs, psd = welch(
            data_filt,
            fs=fs_out,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            axis=1,
        )

        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs2 = freqs[mask]
        psd2 = psd[:, mask]

        psd_db = 10 * np.log10(psd2 + 1e-30)

        if robust_normalize:
            med = np.nanmedian(psd_db, axis=1, keepdims=True)
            psd_plot = psd_db - med
        else:
            psd_plot = psd_db

        if freqs_ref is None:
            freqs_ref = freqs2
        else:
            if not np.allclose(freqs2, freqs_ref):
                raise ValueError("Frequency axis differs between chunks")

        psd_list.append(psd_plot)
        ch_list.append(chx)

        del data, data_filt, psd, psd2, psd_db, psd_plot

    ch_all = np.concatenate(ch_list)
    psd_plot_all = np.vstack(psd_list)

    if plot:
        plt.figure(figsize=(15, 6))
        plt.imshow(
            psd_plot_all.T,
            aspect="auto",
            origin="lower",
            extent=[ch_all[0], ch_all[-1], freqs_ref[0], freqs_ref[-1]],
        )
        plt.colorbar(label="Relative PSD [dB/channel median]")
        plt.xlabel("DAS channel index")
        plt.ylabel("Frequency [Hz]")
        plt.title(
            f"DAS f-x spectrum chunked-like-original | "
            f"nperseg={nperseg / fs_out:.1f}s"
        )
        plt.tight_layout()
        plt.show()

    return {
        "chx": ch_all,
        "freqs": freqs_ref,
        "psd_plot": psd_plot_all,
        "fs": fs_out,
    }

def manual_pick_curve(Z, f, x_chan, title="Manual picking"):
    """
    Clique des points sur l'image.
    Clic gauche = ajouter un point.
    Entrée = terminer.
    Retourne x_picks, f_picks.
    """

    fig, ax = plt.subplots(figsize=(15, 6))

    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[x_chan[0], x_chan[-1], f[0], f[-1]]
    )

    plt.colorbar(im, ax=ax, label="Relative PSD used for picking")

    ax.set_xlabel("DAS channel index")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    ax.grid(True)

    print("\nClique les points de la courbe.")
    print("Appuie sur Entrée quand tu as fini.\n")

    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    if len(pts) < 2:
        raise ValueError("Il faut au moins deux points pour interpoler.")

    pts = np.asarray(pts)
    x_picks = pts[:, 0]
    f_picks = pts[:, 1]

    # tri spatial
    order = np.argsort(x_picks)
    x_picks = x_picks[order]
    f_picks = f_picks[order]

    return x_picks, f_picks

def interpolate_manual_picks(x_picks, f_picks, x_target, sigma_pts=20):
    """
    Interpole les picks manuels sur la grille x_target.
    Puis applique un lissage optionnel.
    """

    interp_fun = interp1d(
        x_picks,
        f_picks,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    f_interp = interp_fun(x_target)

    if sigma_pts is not None and sigma_pts > 0:
        f_interp = gaussian_filter1d(f_interp, sigma=sigma_pts)

    return f_interp

def extract_fx_arrays(res_fx, dx=4.08):
    f = np.asarray(res_fx["freqs"])
    x = np.asarray(res_fx["chx"])
    Z = np.asarray(res_fx["psd_plot"])

    # chx = indices de canaux -> distance m
    dist_m = x * dx

    # Z doit être [n_freq, n_x]
    if Z.shape[0] != len(f):
        Z = Z.T

    return f, dist_m, Z


def pick_resonance_frequencies(f, spectrum):
    s = np.asarray(spectrum, dtype=float)
    s = np.nan_to_num(s, nan=np.nanmedian(s))

    s = s - np.nanmin(s)
    if np.nanmax(s) > 0:
        s = s / np.nanmax(s)

    s_smooth = gaussian_filter1d(s, 2.0)

    # bandes à ajuster
    band1 = (f >= 1.3) & (f <= 2.6)
    band2 = (f >= 2.6) & (f <= 4.2)

    if band1.sum() < 3 or band2.sum() < 3:
        return np.nan, np.nan

    fs1 = f[band1][np.argmax(s_smooth[band1])]
    fs2 = f[band2][np.argmax(s_smooth[band2])]

    if fs1 <= 1.3 + 0.05 or fs1 >= 2.6 - 0.05:
        return np.nan, np.nan

    if fs2 <= 2.6 + 0.05 or fs2 >= 4.2 - 0.05:
        return np.nan, np.nan
    if fs2 <= fs1:
        return fs1, np.nan

    if fs2 > 3 * fs1:
        return fs1, np.nan

    return fs1, fs2


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
