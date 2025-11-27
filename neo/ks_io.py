# neo_pipeline/ks_io.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _ismember(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Equivalent of MATLAB ismember for 1D arrays, returning indices or None.
    """
    mapping = {val: i for i, val in enumerate(b)}
    return np.array([mapping.get(x, None) for x in a])


def _separate_spikes(
    cluster_ids: np.ndarray,
    spike_times: np.ndarray,
    spike_samples: np.ndarray,
    spike_amps: np.ndarray,
    spike_clusters: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build a dict of clusters -> {Spike Times, Spike Samples, Spike Amps}
    """
    cluster_dict: Dict[str, Dict[str, np.ndarray]] = {}

    for cid in cluster_ids:
        mask = spike_clusters == cid
        cluster_dict[str(cid)] = {
            "Spike Times": spike_times[mask],
            "Spike Samples": spike_samples[mask],
            "Spike Amps": spike_amps[mask],
        }

    return cluster_dict


def _read_cluster_labels(file_location: str, automated: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read KS cluster labels and return arrays of good, mua, noise cluster IDs.
    """
    if automated:
        df = pd.read_csv(f"{file_location}/cluster_KSLabel.tsv", delim_whitespace=True)
        group_col = "KSLabel"
    else:
        df = pd.read_csv(f"{file_location}/cluster_group.tsv", delim_whitespace=True)
        group_col = "group"

    replace = {"mua": 1, "good": 2, "noise": 3}
    df_number = df.replace(replace)

    mua_cluster = np.array(df_number[df_number[group_col] == 1]["cluster_id"])
    good_cluster = np.array(df_number[df_number[group_col] == 2]["cluster_id"])
    noise_cluster = np.array(df_number[df_number[group_col] == 3]["cluster_id"])

    return good_cluster, mua_cluster, noise_cluster


def read_ks_output(
    file_location: str,
    samplerate: int = 30_000,
    automated: bool = False,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """
    Read Kilosort output and return dictionaries for good, MUA, and noise clusters.
    """
    spike_samples = np.load(f"{file_location}/spike_times.npy")
    spike_clusters = np.load(f"{file_location}/spike_clusters.npy")
    spike_amps = np.load(f"{file_location}/amplitudes.npy")

    print(f"cluster has: {spike_samples.size} spikes")

    spike_times = spike_samples / samplerate

    good_cluster_ids, mua_cluster_ids, noise_cluster_ids = _read_cluster_labels(file_location, automated=automated)

    # Good
    good_mask = _ismember(spike_clusters, good_cluster_ids) != None  # noqa: E711 (want None comparison)
    good_cluster_dict = _separate_spikes(
        good_cluster_ids,
        spike_times[good_mask].flatten(),
        spike_samples[good_mask].flatten(),
        spike_amps[good_mask].flatten(),
        spike_clusters[good_mask].flatten(),
    )

    # MUA
    mua_mask = _ismember(spike_clusters, mua_cluster_ids) != None  # noqa: E711
    mua_cluster_dict = _separate_spikes(
        mua_cluster_ids,
        spike_times[mua_mask].flatten(),
        spike_samples[mua_mask].flatten(),
        spike_amps[mua_mask].flatten(),
        spike_clusters[mua_mask].flatten(),
    )

    # Noise
    noise_mask = _ismember(spike_clusters, noise_cluster_ids) != None  # noqa: E711
    noise_cluster_dict = _separate_spikes(
        noise_cluster_ids,
        spike_times[noise_mask].flatten(),
        spike_samples[noise_mask].flatten(),
        spike_amps[noise_mask].flatten(),
        spike_clusters[noise_mask].flatten(),
    )

    return good_cluster_dict, mua_cluster_dict, noise_cluster_dict


def readKS_SaveDF(
    file_location: str,
    save_path: str,
    samplerate: int = 30_000,
    automated: bool = False,
) -> pd.DataFrame:
    """
    Read Kilosort output and save a Pandas DataFrame with nested cluster dictionaries.

    Parameters
    ----------
    file_location : str
        Main result folder from Kilosort.
    save_path : str
        Path where the DataFrame pickle is stored.
    samplerate : int
        Sampling rate in Hz (default: 30000).
    automated : bool
        If True, read 'cluster_KSLabel.tsv'; otherwise 'cluster_group.tsv'.

    Returns
    -------
    DataFrame
        Multi-indexed dict-of-dicts (Good / Mua / Noise).
    """
    good_dict, mua_dict, noise_dict = read_ks_output(
        file_location=file_location, samplerate=samplerate, automated=automated
    )

    result_dict = {"Good": good_dict, "Mua": mua_dict, "Noise": noise_dict}

    result_df = pd.DataFrame.from_dict(
        {(quality, cid): result_dict[quality][cid] for quality in result_dict for cid in result_dict[quality]},
        orient="columns",
    )

    result_df.to_pickle(f"{save_path}/TimesSamplesAmps")
    return result_df
