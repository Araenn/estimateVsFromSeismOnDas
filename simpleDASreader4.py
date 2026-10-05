"""OptoDAS reader used by the resonance picking pipeline.

Requires ASN's h5pydict module for the OptoDAS HDF5 format. Signals have
shape [time, channel]. Integration is continuous across concatenated files.
"""

import os
import warnings
import numpy as np


def unwrap(phi, wrapStep=2 * np.pi, axis=-1):
    """Unwrap phase along an axis using the recorded wrapping interval."""
    scale = 2 * np.pi / wrapStep
    return (np.unwrap(phi * scale, axis=axis) / scale).astype(phi.dtype)


def _absolute_channels(meta):
    """Recover absolute channel numbers from legacy ROI metadata."""
    demod = meta["demodSpec"]
    rois = zip(demod["roiStart"], demod["roiEnd"], demod["roiDec"])
    return np.sort(np.unique(np.concatenate([
        np.arange(start, end + 1, step) for start, end, step in rois])))


def _ROI_channels(meta, roiIndex):
    """Return data column indices corresponding to the selected ROIs."""
    if isinstance(roiIndex, (int, np.integer)):
        roiIndex = [roiIndex]
    channels = np.asarray(meta["header"]["channels"])
    demod = meta["demodSpec"]
    selected = []
    for index in roiIndex:
        if not 0 <= index < len(demod["roiStart"]):
            raise ValueError(f"Invalid ROI index: {index}.")
        absolute = np.arange(demod["roiStart"][index],
                             demod["roiEnd"][index] + 1,
                             demod["roiDec"][index])
        selected.append(np.flatnonzero(np.isin(channels, absolute)))
    if not selected:
        raise ValueError("At least one ROI must be selected.")
    return np.concatenate(selected)


def _fix_meta(meta):
    """Supply version 7 interpretation fields for older OptoDAS files."""
    if meta["fileVersion"] >= 7:
        return
    c = 299792458
    cable = meta.setdefault("cableSpec", {
        "fiberOverLength": 1.0, "refractiveIndex": 1.4677, "zeta": 0.78})
    header, demod = meta["header"], meta["demodSpec"]
    dx_fiber = demod["dTau"] * c / (2 * cable["refractiveIndex"])
    overlength = cable.get("fiberOverLength", cable.get("fiberOverlength", 1.0))
    header["dx"] = dx_fiber / overlength
    header.setdefault("gaugeLength", demod["nDiffTau"] * dx_fiber)
    scale = header["dt"] * header["gaugeLength"]
    header["dataScale"] = header.get("dataScale", np.pi / 2**29) / scale
    header["spatialUnwrRange"] = header.get("spatialUnwrRange", 8 * np.pi) / scale
    header["unit"] = "rad/m/s"
    header["sensitivityUnit"] = "rad/m/ε"
    header["channels"] = _absolute_channels(meta)
    cable["sensorDistances"] = header["channels"] * header["dx"]
    itu = int(meta["monitoring"]["Laser"]["itu"])
    wavelength = c / (190e12 + itu * 1e11)
    header["sensitivity"] = 4 * np.pi * cable["zeta"] * cable["refractiveIndex"] / wavelength


def _check_phase_options(meta, integrate, unwr, spikeThr):
    """Require time-differentiated phase for unwrapping and integration."""
    if integrate or unwr or spikeThr is not None:
        if meta["header"]["dataType"] < 3 or meta["demodSpec"]["nDiffTau"] == 0:
            raise ValueError("Unwrapping, spike removal and integration require "
                             "time-differentiated phase data.")


def _integrated_unit(unit):
    """Remove the time denominator when integrating a recorded rate."""
    return unit[:-2] if unit.endswith("/s") else f"({unit})*s"


def load_DAS_file(filename, chIndex=None, roiIndex=None, samples=None,
                  integrate=True, unwr=True, metaDetail=1, useSensitivity=True,
                  spikeThr=None):
    """Read selected DAS channels, apply scaling and optionally integrate.

    chIndex selects data columns; roiIndex selects recorded regions instead.
    samples is a count or a range. Decimate only after integration or filtering.
    metaDetail=1 keeps interpretation metadata; other values keep all metadata.
    With sensitivity conversion, the output unit is strain/s or strain.
    """
    import h5pydict

    with h5pydict.DictFile(filename, "r") as stream:
        meta = stream.load_dict(skipFields=["data"])
        _fix_meta(meta)
        if metaDetail == 1:
            demod, monitor = meta["demodSpec"], meta["monitoring"]
            meta = {key: meta[key] for key in
                    ("fileVersion", "header", "timing", "cableSpec")}
            meta["monitoring"] = {"Gps": monitor["Gps"],
                                  "Laser": {"itu": monitor["Laser"]["itu"]}}
            meta["demodSpec"] = {key: demod[key] for key in
                                  ("roiStart", "roiEnd", "roiDec",
                                   "nDiffTau", "nAvgTau", "dTau")}
        if isinstance(samples, (int, np.integer)):
            samples = range(samples)
        elif samples is None:
            samples = slice(None)
        if isinstance(samples, range) and samples.step > 1 and integrate:
            warnings.warn("Time decimation before integration is not recommended.",
                          stacklevel=2)
        if roiIndex is not None:
            if chIndex is not None:
                raise ValueError("chIndex must be None when using roiIndex.")
            chIndex = _ROI_channels(meta, roiIndex)
        elif chIndex is None:
            chIndex = slice(None)
        signal = np.asarray(stream["data"][samples, chIndex], dtype=np.float32)
        signal = signal * meta["header"]["dataScale"]

    header = meta["header"]
    _check_phase_options(meta, integrate, unwr, spikeThr)
    if unwr and header["spatialUnwrRange"]:
        signal = unwrap(signal, header["spatialUnwrRange"],
                        axis=1 if signal.ndim > 1 else 0)
    if spikeThr is not None:
        signal[np.abs(signal) > spikeThr] = 0
    unit = header.get("unit", "rad/m/s")
    if useSensitivity:
        sensitivity = header.get("sensitivity", header.get("sensitivities"))
        if sensitivity is None:
            raise KeyError("No sensitivity found in the DAS header.")
        if np.ndim(sensitivity):
            sensitivity = np.asarray(sensitivity)
            if sensitivity.size == len(header["channels"]):
                sensitivity = sensitivity[chIndex]
        signal = signal / sensitivity
        unit = "strain/s"
    if integrate:
        signal = np.cumsum(signal, axis=0) * header["dt"]
        unit = _integrated_unit(unit)
    meta["appended"] = {
        "dataOffs": np.asarray(header["phiOffs"])[chIndex],
        "unit": unit,
        "channels": np.asarray(header["channels"])[chIndex]}
    return signal, meta


def load_multiple_DAS_files(path, fileIds, chIndex=None, roiIndex=None,
                            integrate=True, unwr=True, metaDetail=1,
                            useSensitivity=True, spikeThr=None):
    """Concatenate contiguous files before spatial unwrapping and integration.

    fileIds are numeric HHMMSS filenames without the .hdf5 extension.
    """
    parts, meta = [], None
    for file_id in fileIds:
        filename = os.path.join(path, str(file_id).zfill(6) + ".hdf5")
        signal, current = load_DAS_file(
            filename, chIndex=chIndex, roiIndex=roiIndex, integrate=False,
            unwr=False, metaDetail=metaDetail, useSensitivity=useSensitivity)
        if meta is None:
            meta = current
        elif (current["header"]["dt"] != meta["header"]["dt"]
              or not np.array_equal(current["appended"]["channels"],
                                    meta["appended"]["channels"])):
            raise ValueError("DAS files must have matching sample intervals and channels.")
        parts.append(signal)
    if not parts:
        raise ValueError("At least one DAS file must be selected.")
    signal = np.concatenate(parts, axis=0)
    _check_phase_options(meta, integrate, unwr, spikeThr)
    if unwr and meta["header"]["spatialUnwrRange"]:
        signal = unwrap(signal, meta["header"]["spatialUnwrRange"], axis=1)
    if spikeThr is not None:
        signal[np.abs(signal) > spikeThr] = 0
    if integrate:
        signal = np.cumsum(signal, axis=0) * meta["header"]["dt"]
        meta["appended"]["unit"] = _integrated_unit(meta["appended"]["unit"])
    return signal, meta
