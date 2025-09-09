"""
Biodiversity module for Deep-Sea eDNA Analysis Pipeline
======================================================

This module contains components for biodiversity assessment and analysis:
- Diversity indices calculation (Shannon, Simpson, Chao1, etc.)
- Community structure analysis
- Beta diversity and ordination analysis
- Statistical testing for biodiversity patterns

Author: SIH 2025 Team
Date: 2025-09-09
"""

from .diversity_analyzer import DiversityAnalyzer

__all__ = ['DiversityAnalyzer']
