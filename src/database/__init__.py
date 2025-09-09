"""
Database integration module for Deep-Sea eDNA Analysis Pipeline
===============================================================

This module provides optional integration with external databases:
- NCBI BLAST database downloads and management
- Validation of novel taxa discoveries
- Reference sequence matching and analysis

The pipeline is designed to minimize database dependency while
providing optional validation capabilities.

Author: SIH 2025 Team
Date: 2025-09-09
"""

from .ncbi_integration import NCBIDatabaseManager

__all__ = ['NCBIDatabaseManager']
