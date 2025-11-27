# NP data managment

Tools to convert Neuropixels / Kilosort output and National Instruments stimulus data
into [`neo`](https://neuralensemble.org/neo/) `Block`/`Segment` structures and
analysis-ready pandas DataFrames.

## Features

- Read Kilosort output (`spike_times`, `spike_clusters`, `amplitudes`, cluster labels)
- Build `neo.Block` and `neo.Segment` objects with:
  - Stimulus events
  - Analog feedback signals
  - Spike trains
  - Raster-aligned spike data
- Generate a summary DataFrame linking rasters, events, analog signals, and units
- Convenience accessors to pull rasters/events/spikes back out of the `Block`
