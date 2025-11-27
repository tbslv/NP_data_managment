# neo_pipeline/helpers.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import neo
import numpy as np
import pandas as pd
import quantities as pq


@dataclass
class SweepParameters:
    sampling_rate: float
    sweeplength: float
    pre: float
    post: float
    repetitions: int
    exp_id: Any
    pupil_freq: Any
    active_zones: Any
    stim_infos: List[Dict[str, Any]]


def generate_segment_core(
    date: Optional[Any] = None,
    date_of_rec: Optional[Any] = None,
    working_directory: Optional[str] = None,
    exp: Optional[str] = None,
) -> neo.Segment:
    """
    Create a basic neo.Segment with common metadata populated.
    """
    seg = neo.Segment()
    seg.file_datetime = date
    seg.rec_datetime = date_of_rec
    seg.file_origin = working_directory
    seg.name = exp
    return seg


def generate_block_core(
    date: Optional[Any] = None,
    working_directory: Optional[str] = None,
    date_of_rec: Optional[Any] = None,
    recording_location: Optional[str] = None,
    animal_id: Optional[str] = None,
    number_exp: Optional[int] = None,
) -> neo.Block:
    """
    Create a basic neo.Block with common metadata populated.
    """
    block = neo.Block()
    block.file_datetime = date
    block.file_origin = working_directory
    block.rec_datetime = date_of_rec
    block.name = recording_location
    block.annotate(animalID=animal_id, exps_tot=str(number_exp))
    return block


def list_files(working_directory: str) -> Tuple[List[str], List[str]]:
    """
    List metadata and stimdata files inside a folder, based on filename patterns.
    """
    import os

    content = os.listdir(working_directory)

    metadatas: List[str] = []
    stimdatas: List[str] = []

    for entry in content:
        if "sweepParameter" in entry:
            metadatas.append(entry)
        elif "stimData" in entry:
            stimdatas.append(entry)

    return metadatas, stimdatas


def read_sweepparamter(metadata: pd.DataFrame) -> SweepParameters:
    """
    Extract sweep parameters and stimulus information from the metadata DataFrame.

    The function is intentionally tolerant to 'old' formats where certain
    columns (e.g. 'PupilFreq', 'active_zones', ramps) may be missing.
    """
    combi = np.unique(metadata["sweepID"])

    samplingrate = float(metadata["Samplingrate"].iloc[0])
    sweeplength = float(metadata["Sweeplength"].iloc[0])
    pre = float(metadata["Pre Stimulus Time"].iloc[0])
    post = sweeplength - pre
    repetitions = int(metadata["Repititions"].iloc[0])
    exp_id = metadata["ID"].iloc[0]

    pupil_freq = metadata["PupilFreq"].iloc[0] if "PupilFreq" in metadata.columns else "no pupil tracking"
    active_zones = metadata["active_zones"].iloc[0] if "active_zones" in metadata.columns else "old stimulator"

    stim_infos: List[Dict[str, Any]] = []
    for sweep_id in combi:
        selection = metadata[metadata["sweepID"] == sweep_id]
        if "Modalities" in selection.columns:
            stim_infos.append(
                {
                    "sweepID": selection["sweepID"].iloc[0],
                    "modality": selection["Modalities"].iloc[0],
                    "duration": selection["Stimulus Duration"].iloc[0],
                }
            )
        else:
            # Legacy temperature-based stimulator
            base_dict: Dict[str, Any] = {
                "sweepID": selection["sweepID"].iloc[0],
                "duration": selection["Stimulus Duration"].iloc[0],
            }
            if "Stimulus Temp" in selection.columns:
                base_dict["stimtemp"] = selection["Stimulus Temp"].iloc[0]
            if "Stimulus Temp" not in selection.columns and "stimamp" in selection.columns:
                base_dict["stimamp"] = selection["stimamp"].iloc[0]
            if "Baseline Temp" in selection.columns:
                base_dict["basetemp"] = selection["Baseline Temp"].iloc[0]
            if "OnRamp" in selection.columns:
                base_dict["onramp"] = selection["OnRamp"].iloc[0]
            if "OffRamp" in selection.columns:
                base_dict["offramp"] = selection["OffRamp"].iloc[0]
            stim_infos.append(base_dict)

    return SweepParameters(
        sampling_rate=samplingrate,
        sweeplength=sweeplength,
        pre=pre,
        post=post,
        repetitions=repetitions,
        exp_id=exp_id,
        pupil_freq=pupil_freq,
        active_zones=active_zones,
        stim_infos=stim_infos,
    )


def generate_event_core(data_timestamps: np.ndarray, title: str) -> neo.Event:
    """
    Build a neo.Event from time stamps (in seconds) and a label.
    """
    timestamps = np.asarray(data_timestamps, dtype=float)
    ttl = neo.Event(timestamps * pq.s, labels=[title] * timestamps.size)
    ttl.name = title
    return ttl


def generate_spiketrain_core(
    spikes: pd.DataFrame,
    start: Optional[float] = None,
    end: Optional[float] = None,
    quality: str = "Good",
    cluster_id: Optional[str] = None,
) -> neo.SpikeTrain:
    """
    Build a neo.SpikeTrain from the KS spike dict structure.

    Parameters
    ----------
    spikes : dict-like
        Structure indexed by quality -> cluster_id -> 'Spike Times'.
    """
    if cluster_id is None:
        raise ValueError("cluster_id must be provided")

    unit_dict = spikes[quality][cluster_id]
    spike_times = np.asarray(unit_dict["Spike Times"], dtype=float)

    if start is not None and end is not None:
        mask = (spike_times > start) & (spike_times < end)
        spike_times = spike_times[mask]

    train = neo.SpikeTrain(spike_times, units="s", t_start=start, t_stop=end, name=cluster_id)
    train.annotate(quality=quality)
    return train


def raster_build(
    spikes: np.ndarray,
    trials: Iterable[float],
    samplingrate: float = 30_000,
    sweeplength: Optional[float] = None,
) -> np.ndarray:
    """
    Build a spike raster (trials x time_bins) around trial onset times.

    Parameters
    ----------
    spikes : array-like
        Spike times in seconds (absolute).
    trials : iterable of float
        Trial onset times in seconds.
    samplingrate : float
        Sampling rate in Hz.
    sweeplength : float
        Duration of each trial (seconds).
    """
    spikes = np.asarray(spikes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    if sweeplength is None:
        raise ValueError("sweeplength must be provided")

    n_trials = len(trials)
    n_bins = int(sweeplength * samplingrate)
    raster = np.zeros((n_trials, n_bins), dtype=int)

    for i, t0 in enumerate(trials):
        trial_spikes = spikes[(spikes > t0) & (spikes < (t0 + sweeplength))] - t0
        indices = (trial_spikes * samplingrate).astype(int)
        raster[i, indices] = 1

    return raster


def generate_alignedspikes_core(raster: np.ndarray, name: Optional[str] = None) -> neo.IrregularlySampledSignal:
    """
    Convert a binary raster into an IrregularlySampledSignal storing aligned spike times.

    The 'signal' values correspond to trial indices; the times correspond to bin indices.
    """
    trial_idx, time_idx = np.where(raster == 1)
    spikes = neo.IrregularlySampledSignal(
        times=time_idx,
        signal=trial_idx,
        units="s",
        time_units="s",
        name=name,
        description="sweep-aligned spike times // signal values = sweep_id (trial index)",
    )
    return spikes


def generate_analogsignal_core(
    data: np.ndarray,
    units: str = "C",
    sampling_rate: float = 1_000.0,
    index: Optional[int] = None,
) -> neo.AnalogSignal:
    """
    Create a neo.AnalogSignal from numpy data.

    Parameters
    ----------
    data : array-like
        Shape: (time, channels, ...) or similar.
    """
    data = np.asarray(data)
    name = None if index is None else index + 1
    return neo.AnalogSignal(data, units=units, sampling_rate=sampling_rate * pq.Hz, name=name)


def add_metaData(
    obj: Any,
    samplingrate: float,
    sweeplength: float,
    pre: float,
    post: float,
    rep: int,
    exp_id: Any,
    pupil_freq: Any,
    active_zones: Any,
    stim_info: Dict[str, Any],
    mod: Optional[str] = None,
) -> None:
    """
    Add common Neo annotations (sweep parameters, stimulus info) to any neo object.
    """
    # Default values
    sweep_id = stim_info.get("sweepID")
    modality = mod or stim_info.get("modality", "unknown")
    duration = stim_info.get("duration", "unknown")
    basetemp = stim_info.get("basetemp", "no temp")
    stimtemp = stim_info.get("stimtemp", stim_info.get("stimamp", "no temp"))
    onramp = stim_info.get("onramp", "old stim")
    offramp = stim_info.get("offramp", "old stim")

    obj.annotate(
        sweeplength=sweeplength,
        samplingrate=samplingrate,
        basetemp=basetemp,
        sweep_Id=sweep_id,
        stimtemp=stimtemp,
        duration=duration,
        onramp=onramp,
        offramp=offramp,
        pre=pre,
        post=post,
        pupil_freq=pupil_freq,
        active_zones=active_zones,
        trials=rep,
        mod=modality,
        exp_id=exp_id,
    )


def generate_segment(
    data: pd.DataFrame,
    segment: neo.Segment,
    spike_data: pd.DataFrame,
    metadata: pd.DataFrame,
    quality: str = "Good",
) -> neo.Segment:
    """
    Build a fully populated neo.Segment with:
    - Analog signals (feedback channels)
    - Event timestamps (stimulus onsets)
    - Spike trains and aligned spike rasters

    Parameters match structure of original generate_segment() implementation.
    """
    params = read_sweepparamter(metadata)

    # Determine modality
    modality = "touch" if "Modalities" in metadata.columns else "temp"

    # --- Analog signals + metadata ---
    stim_infos_iter = iter(params.stim_infos)
    for row_idx in range(data.shape[0]):
        feedback_3 = data.iloc[row_idx]["Feedback_temp"]
        # feedback_4 = data.iloc[row_idx]['Feedback_4']  # can be added when needed

        feedback = np.zeros((feedback_3.shape[0], 2, feedback_3.shape[2]))
        feedback[:, 0, :] = feedback_3[:, 0, :]

        ai_signal = generate_analogsignal_core(feedback, index=row_idx)
        stim_info = next(stim_infos_iter)
        add_metaData(
            ai_signal,
            samplingrate=params.sampling_rate,
            sweeplength=params.sweeplength,
            pre=params.pre,
            post=params.post,
            rep=params.repetitions,
            exp_id=params.exp_id,
            pupil_freq=params.pupil_freq,
            active_zones=params.active_zones,
            stim_info=stim_info,
            mod=modality,
        )
        segment.analogsignals.append(ai_signal)

    # --- Events (stimulus starts) ---
    data_timestamps = data["Stimstart"].dropna()
    stim_infos_iter = iter(params.stim_infos)
    for idx, (title, timestamps) in enumerate(data_timestamps.items()):
        ttl = generate_event_core(np.asarray(timestamps), str(title))
        stim_info = next(stim_infos_iter)
        add_metaData(
            ttl,
            samplingrate=params.sampling_rate,
            sweeplength=params.sweeplength,
            pre=params.pre,
            post=params.post,
            rep=params.repetitions,
            exp_id=params.exp_id,
            pupil_freq=params.pupil_freq,
            active_zones=params.active_zones,
            stim_info=stim_info,
            mod=modality,
        )
        segment.events.append(ttl)

    # --- Determine segment start/stop based on events ---
    starts = [np.min(ev.times) for ev in segment.events]
    ends = [np.max(ev.times) for ev in segment.events]
    start = float(np.min(starts))
    end = float(np.max(ends) + params.post)

    print(f"start: {start}")
    print(f"end:   {end}")

    # --- Spike trains ---
    cluster_ids = list(spike_data[quality].keys())
    for cluster_id in cluster_ids:
        train = generate_spiketrain_core(
            spikes=spike_data,
            start=start,
            end=end,
            quality=quality,
            cluster_id=cluster_id,
        )
        segment.spiketrains.append(train)

    # --- Raster-aligned spikes ---
    for spk in segment.spiketrains:
        spikes = spk.as_array()
        for ev, stim_info in zip(segment.events, params.stim_infos):
            ttls = ev.as_array() - params.pre
            raster = raster_build(
                spikes,
                ttls,
                samplingrate=params.sampling_rate,
                sweeplength=params.sweeplength,
            )
            aligned_spikes = generate_alignedspikes_core(raster, name=spk.name)
            add_metaData(
                aligned_spikes,
                samplingrate=params.sampling_rate,
                sweeplength=params.sweeplength,
                pre=params.pre,
                post=params.post,
                rep=params.repetitions,
                exp_id=params.exp_id,
                pupil_freq=params.pupil_freq,
                active_zones=params.active_zones,
                stim_info=stim_info,
                mod=modality,
            )
            segment.irregularlysampledsignals.append(aligned_spikes)

    return segment


def generate_df(block: neo.Block) -> pd.DataFrame:
    """
    Build a pandas DataFrame summarizing relationships between rasters,
    events, analog signals and spiketrains for each segment in the block.
    """
    df_list: List[pd.DataFrame] = []

    for seg_idx, seg in enumerate(block.segments):
        seg_df = pd.DataFrame()
        for r_idx, raster in enumerate(seg.irregularlysampledsignals):
            row_key = str(r_idx)
            seg_df.at[row_key, "cluster"] = raster.name
            seg_df.at[row_key, "expID"] = seg.name

            # Copy annotations
            for key, value in raster.annotations.items():
                seg_df.at[row_key, str(key)] = value

            seg_df.at[row_key, "raster_loc"] = f"{seg_idx}_{r_idx}"

            # Find matching event, analog signal, spiketrain
            for e_idx, event in enumerate(seg.events):
                if raster.annotations == event.annotations:
                    seg_df.at[row_key, "event_loc"] = f"{seg_idx}_{e_idx}"

            for ai_idx, ai in enumerate(seg.analogsignals):
                if raster.annotations == ai.annotations:
                    seg_df.at[row_key, "ai_loc"] = f"{seg_idx}_{ai_idx}"

            for sp_idx, sp in enumerate(seg.spiketrains):
                if raster.name == sp.name:
                    seg_df.at[row_key, "spiketrain_loc"] = f"{seg_idx}_{sp_idx}"

            seg_df.at[row_key, "date_rec"] = block.rec_datetime
            seg_df.at[row_key, "animal_id"] = block.annotations.get("animalID")
            seg_df.at[row_key, "rawdata_path"] = block.file_origin
            seg_df.at[row_key, "structure"] = block.name

        df_list.append(seg_df)

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, axis=0)
