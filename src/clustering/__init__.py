"""
Clustering module for Deep-Sea eDNA Analysis Pipeline
====================================================

This module contains components for unsupervised learning and novel taxa discovery:
- Sequence encoding to numerical vectors
- Novel taxa discovery using clustering algorithms
- Dimensionality reduction techniques

Author: SIH 2025 Team
Date: 2025-09-09
"""

from .sequence_encoder import SequenceEncoder
from .novel_taxa_discoverer import NovelTaxaDiscoverer

__all__ = ['SequenceEncoder', 'NovelTaxaDiscoverer']
