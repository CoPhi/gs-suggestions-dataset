"""
Ithaca Module for Ancient Greek Character-Level Text Restoration.
Integrates DeepMind's Ithaca architecture with GreekSchools datasets and API.
"""

from models.ithaca.inference.predict import fill_mask_ithaca

__all__ = ["fill_mask_ithaca"]
