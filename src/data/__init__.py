"""Portable dataset contracts used by public training and evaluation tools."""

from src.data.brep_dataset import PortableBrepDataset, validate_brep_record
from src.data.manifest import DatasetManifest, load_dataset_manifest

__all__ = [
    "DatasetManifest",
    "PortableBrepDataset",
    "load_dataset_manifest",
    "validate_brep_record",
]
