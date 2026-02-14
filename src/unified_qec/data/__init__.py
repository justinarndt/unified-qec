"""
Open Data: Cloud-native storage for QEC syndrome datasets.

Provides Zarr-based analytical storage and WebDataset-based
streaming archives for ML training pipelines.
"""

from unified_qec.data.webdataset_writer import WebDatasetWriter as WebDatasetWriter
from unified_qec.data.converters import SinterToData as SinterToData

# Zarr import is conditional
try:
    from unified_qec.data.zarr_store import ZarrStore as ZarrStore
except ImportError:
    pass
