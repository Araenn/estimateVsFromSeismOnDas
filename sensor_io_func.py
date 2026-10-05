"""Select the OptoDAS files used to calculate an event spectrum."""

import os
import re


def get_das_files_for_window(t_start, t_end, base_path):
    """Return a folder and sorted HHMMSS file IDs for a same-day time window.

    Times are [hour, minute, second]. The original five-second tolerance is
    retained. This selects whole files; it does not trim individual samples.
    """
    t0 = t_start[0] * 3600 + t_start[1] * 60 + t_start[2]
    t1 = t_end[0] * 3600 + t_end[1] * 60 + t_end[2]
    if t1 <= t0:
        raise ValueError("The end time must follow the start time on the same day.")
    ids = []
    for filename in os.listdir(base_path):
        if not filename.endswith(".hdf5"):
            continue
        match = re.search(r"(\d{6})", filename)
        if match is None:
            continue
        name = match.group(1)
        hh, mm, ss = int(name[:2]), int(name[2:4]), int(name[4:6])
        time_seconds = hh * 3600 + mm * 60 + ss
        if t0 - 5 <= time_seconds <= t1 + 5:
            ids.append(int(name))
    if not ids:
        raise FileNotFoundError(f"No DAS files found for the requested window in {base_path}.")
    return base_path, sorted(ids)
