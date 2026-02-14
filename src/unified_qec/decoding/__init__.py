"""
Decoding Layer

BP+OSD decoder (ASR-MP), Union-Find decoder, Unified Sinter API,
and DEM utilities.
"""

from unified_qec.decoding.dem_utils import (
    dem_to_matrices as dem_to_matrices,
    get_channel_llrs as get_channel_llrs,
)
from unified_qec.decoding.sinter_api import (
    UnifiedQECDecoder as UnifiedQECDecoder,
)
