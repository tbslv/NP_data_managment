# neo_pipeline/cli.py

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from neo.io import PickleIO

from . import helpers as nh
from . import ks_io as rKS


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate neo.Block and metadata DataFrame from KS + NI data.")
    parser.add_argument("working_directory", type=Path, help="Root recording folder (e.g. Record_Node_101).")
    parser.add_argument("--recording_location", type=str, default="pIC", help="Recording location (e.g., pIC).")
    parser.add_argument("--date_of_rec", type=str, default=None, help="Recording date (e.g. 20250606).")
    parser.add_argument("--animal_id", type=str, default="UNKNOWN", help="Animal ID.")
    parser.add_argument("--depth_of_probe", type=int, default=None, help="Probe depth in µm (optional).")
    parser.add_argument("--probe_index_pIC", type=int, default=0, help="Probe index when recording_location == 'pIC'.")
    parser.add_argument("--samplerate", type=int, default=30_000, help="Sampling rate in Hz.")
    parser.add_argument(
        "--automated",
        action="store_true",
        help="Use automated labels (cluster_KSLabel.tsv) instead of manual cluster_group.tsv.",
    )
    return parser


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        parser = build_argparser()
        args = parser.parse_args()

    working_directory = args.working_directory.resolve()
    recording_location = args.recording_location
    date_of_rec = args.date_of_rec or dt.datetime.now().strftime("%Y%m%d")
    animal_id = args.animal_id

    if recording_location == "pIC":
        flag = "A"
        probe = args.probe_index_pIC
    else:
        flag = "B"
        probe = 1

    date_str = dt.datetime.now().strftime("%y-%m-%d-%H-%M")

    stimdata_folder = working_directory / "NationalInstruments"
    metadatas, stimdatas = nh.list_files(str(stimdata_folder))

    stimdatas = [s for s in stimdatas if str(probe) in s]

    print("Selected stimdatas:", stimdatas)
    print(f"Found {len(stimdatas)} stimdata files.")

    ks_output_folder = (
        working_directory
        / "experiment1"
        / "recording1"
        / "continuous"
        / f"Neuropix-PXI-100.Probe{flag}-AP"
        / "Sorting"
        / "1"
    )
    savefolder = working_directory / f"SpikeSortingResults_Probe_{recording_location}"
    savefolder.mkdir(exist_ok=True)

    spikes_tot = rKS.readKS_SaveDF(
        str(ks_output_folder), str(savefolder), samplerate=args.samplerate, automated=args.automated
    )

    print("SpikeData keys:", spikes_tot.keys())

    block = nh.generate_block_core(
        date=date_str,
        working_directory=str(working_directory),
        date_of_rec=date_of_rec,
        recording_location=recording_location,
        animal_id=animal_id,
        number_exp=len(metadatas),
    )

    for exp_idx, meta_name in enumerate(metadatas):
        stim_name = stimdatas[exp_idx]
        print("Processing:", stimdata_folder / stim_name)

        segment = nh.generate_segment_core(
            date=date_str,
            date_of_rec=date_of_rec,
            working_directory=str(working_directory),
            exp=stim_name[:3],
        )

        data = pd.read_pickle(stimdata_folder / stim_name)
        metadata = pd.read_csv(stimdata_folder / meta_name)

        segment = nh.generate_segment(data, segment, spikes_tot, metadata, quality="Good")
        block.segments.append(segment)

    # Save block
    io = PickleIO(filename=f"{recording_location}_block_data_{date_of_rec}.pkl")
    io.write(block)

    # Save DataFrame
    df = nh.generate_df(block)
    df.to_pickle(savefolder / f"{recording_location}_block_df_{date_of_rec}.pkl")

    print("Done.")
    print("Block saved to:", io.filename)
    print("DataFrame saved to:", savefolder / f"{recording_location}_block_df_{date_of_rec}.pkl")


if __name__ == "__main__":
    main()
