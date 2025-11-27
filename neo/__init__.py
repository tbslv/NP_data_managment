"""
neo_pipeline

Tools for converting Neuropixels / Kilosort output and National Instruments
stimulus data into neo Block/Segment structures and pandas DataFrames.
"""

# Convenience imports (lightweight ones only)
from .helpers import (
    generate_block_core,
    generate_segment_core,
    generate_segment,
    generate_df,
    generate_spiketrain_core,
    raster_build,
)

from .access import (
    access_raster_core,
    access_events_core,
    access_spiketrain_core,
    access_aisignal_core,
    get_all_data,
)

from .ks_io import (
    read_ks_output,
    readKS_SaveDF,
)
