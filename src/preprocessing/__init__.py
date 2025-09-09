"""
Preprocessing module for Deep-Sea eDNA Analysis Pipeline
======================================================

This module contains components for preprocessing raw eDNA sequencing data:
- Sequence quality control and filtering
- Marker gene extraction
- Feature generation for ML models

Author: SIH 2025 Team
Date: 2025-09-09
"""

from .sequence_processor import SequenceProcessor

__all__ = ['SequenceProcessor']
