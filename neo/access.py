# neo_pipeline/access.py

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import neo
import numpy as np
import pandas as pd


def _split_loc(loc: str) -> Tuple[int, int]:
    seg_idx_str, obj_idx_str = loc.split("_")
    return int(seg_idx_str), int(obj_idx_str)


def access_raster_core(block: neo.Block, loc: str) -> np.ndarray:
    """
    Access a raster from block using 'segmentIndex_rasterIndex' format.
    Returns (N_spikes, 2) array: [time, trial_index].
    """
    seg_idx, ras_idx = _split_loc(loc)
    raster = block.segments[seg_idx].irregularlysampledsignals[ras_idx]
    data_trials = raster.as_array().ravel()
    data_spikes = raster.times

    data = np.zeros((data_spikes.size, 2), dtype=float)
    data[:, 0] = data_spikes
    data[:, 1] = data_trials
    return data


def access_events_core(block: neo.Block, loc: str) -> np.ndarray:
    """
    Access event timestamps from block using 'segmentIndex_eventIndex'.
    """
    seg_idx, ev_idx = _split_loc(loc)
    return block.segments[seg_idx].events[ev_idx].as_array().ravel()


def access_aisignal_core(block: neo.Block, loc: str) -> np.ndarray:
    """
    Access analog signal data using 'segmentIndex_aiIndex'.
    """
    seg_idx, ai_idx = _split_loc(loc)
    return block.segments[seg_idx].analogsignals[ai_idx].as_array()


def access_spiketrain_core(block: neo.Block, loc: str) -> np.ndarray:
    """
    Access spike times from a spiketrain using 'segmentIndex_spikeIndex'.
    """
    seg_idx, sp_idx = _split_loc(loc)
    return block.segments[seg_idx].spiketrains[sp_idx].times


def get_all_data(
    block: neo.Block,
    selection: pd.DataFrame,
    raster: bool = False,
    timestamps: bool = False,
    spikes: bool = False,
    aisignal: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Fetch multiple data streams for each row in `selection` DataFrame.

    Expects columns: ['raster_loc', 'event_loc', 'spiketrain_loc', 'ai_loc'].
    """
    raster_list: List[np.ndarray] = []
    timestamps_list: List[np.ndarray] = []
    spikes_list: List[np.ndarray] = []
    aisignal_list: List[np.ndarray] = []

    for _, row in selection.iterrows():
        raster_loc, event_loc, spiketrain_loc, ai_loc = row[
            ["raster_loc", "event_loc", "spiketrain_loc", "ai_loc"]
        ].values

        if raster and isinstance(raster_loc, str):
            raster_list.append(access_raster_core(block, raster_loc))

        if timestamps and isinstance(event_loc, str):
            timestamps_list.append(access_events_core(block, event_loc))

        if spikes and isinstance(spiketrain_loc, str):
            spikes_list.append(access_spiketrain_core(block, spiketrain_loc))

        if aisignal and isinstance(ai_loc, str):
            aisignal_list.append(access_aisignal_core(block, ai_loc))

    return raster_list, timestamps_list, spikes_list, aisignal_list
