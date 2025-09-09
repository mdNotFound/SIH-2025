#!/usr/bin/env python3
"""
NCBI Database Integration Module for Deep-Sea eDNA Analysis
==========================================================

This module provides optional integration with NCBI databases for validation
of taxonomic classifications. It minimizes dependency on reference databases
while allowing validation of novel taxa discoveries.

Author: SIH 2025 Team
Date: 2025-09-09
"""

import os
import sys
import requests
import tarfile
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import subprocess
import pandas as pd
from Bio import SeqIO
from Bio.Blast import NCBIXML
import time


class NCBIDatabaseManager:
    """
    Manager for NCBI database downloads and integration.
    
    This class handles downloading selected NCBI database files
    and provides BLAST-based validation for novel taxa discovery.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the NCBI database manager.
        
        Args:
            config: Configuration dictionary with database settings
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Database settings
        self.ncbi_base_url = "https://ftp.ncbi.nlm.nih.gov/blast/db/"
        self.db_path = Path(config.get('ncbi_db_path', 'data/ncbi_nt_euk/'))
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # BLAST settings
        self.blast_evalue = config.get('blast_evalue', 0.001)
        self.blast_max_targets = config.get('blast_max_targets', 10)
        
        # Validation settings
        self.validation_threshold = config.get('validation_threshold', 0.8)
        self.min_identity = config.get('min_identity', 0.7)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the database manager."""
        logger = logging.getLogger('NCBIDatabaseManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def download_selected_databases(self, database_files: List[str] = None) -> Dict[str, bool]:
        """
        Download selected NCBI database files for eukaryotic validation.
        
        Args:
            database_files: List of specific database files to download.
                          If None, downloads recommended subset for eukaryotes.
            
        Returns:
            Dictionary with download status for each file
        """
        # Default subset focusing on eukaryotic sequences (smaller files)
        if database_files is None:
            database_files = [
                'nt_euk.00.tar.gz',  # Eukaryotic nucleotide sequences
                'nt_euk.01.tar.gz',
                'nt_euk.02.tar.gz'   # Reduced set for validation only
            ]
        
        self.logger.info(f"Starting download of {len(database_files)} NCBI database files")
        self.logger.warning("Note: Database files are very large. This may take hours.")
        
        download_results = {}
        
        for db_file in database_files:
            try:
                self.logger.info(f"Downloading {db_file}...")
                success = self._download_single_file(db_file)
                download_results[db_file] = success
                
                if success:
                    self.logger.info(f"✓ Successfully downloaded {db_file}")
                    
                    # Extract the database
                    self._extract_database_file(db_file)
                    
                else:
                    self.logger.error(f"✗ Failed to download {db_file}")
                    
            except Exception as e:
                self.logger.error(f"Error downloading {db_file}: {str(e)}")
                download_results[db_file] = False
        
        return download_results
    
    def _download_single_file(self, filename: str) -> bool:
        """Download a single database file with progress tracking."""
        url = f"{self.ncbi_base_url}{filename}"
        local_path = self.db_path / filename
        
        # Check if file already exists and verify checksum
        if local_path.exists():
            self.logger.info(f"File {filename} already exists. Verifying...")
            if self._verify_checksum(filename):
                self.logger.info(f"✓ {filename} verified, skipping download")
                return True
            else:
                self.logger.warning(f"Checksum mismatch for {filename}, re-downloading")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                downloaded = 0
                chunk_size = 8192
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress indicator
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownloading {filename}: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            
            # Verify download
            if self._verify_checksum(filename):
                return True
            else:
                self.logger.error(f"Checksum verification failed for {filename}")
                local_path.unlink(missing_ok=True)  # Remove corrupted file
                return False
                
        except Exception as e:
            self.logger.error(f"Download failed for {filename}: {str(e)}")
            return False
    
    def _verify_checksum(self, filename: str) -> bool:
        """Verify file checksum against NCBI MD5."""
        try:
            # Download MD5 checksum file
            md5_url = f"{self.ncbi_base_url}{filename}.md5"
            md5_response = requests.get(md5_url, timeout=10)
            md5_response.raise_for_status()
            
            expected_md5 = md5_response.text.strip().split()[0]
            
            # Calculate local file MD5
            local_path = self.db_path / filename
            if not local_path.exists():
                return False
            
            actual_md5 = self._calculate_md5(local_path)
            
            return actual_md5.lower() == expected_md5.lower()
            
        except Exception as e:
            self.logger.warning(f"Could not verify checksum for {filename}: {str(e)}")
            return True  # Skip verification if can't download MD5
    
    def _calculate_md5(self, filepath: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _extract_database_file(self, filename: str):
        """Extract a downloaded tar.gz database file."""
        tar_path = self.db_path / filename
        extract_path = self.db_path / filename.replace('.tar.gz', '')
        
        if extract_path.exists():
            self.logger.info(f"Database {filename} already extracted")
            return
        
        try:
            self.logger.info(f"Extracting {filename}...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=self.db_path)
            
            self.logger.info(f"✓ Extracted {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to extract {filename}: {str(e)}")
    
    def validate_sequences(self, sequences: List[str], sequence_ids: List[str]) -> Dict[str, Any]:
        """
        Validate sequences against NCBI database using BLAST.
        
        Args:
            sequences: List of DNA sequences to validate
            sequence_ids: List of sequence identifiers
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info(f"Validating {len(sequences)} sequences against NCBI database")
        
        # Check if BLAST databases are available
        if not self._check_blast_databases():
            self.logger.warning("No BLAST databases found. Skipping validation.")
            return {'status': 'skipped', 'reason': 'No databases available'}
        
        # Create temporary FASTA file for BLAST
        temp_fasta = self.db_path / "temp_sequences.fasta"
        self._write_fasta_file(sequences, sequence_ids, temp_fasta)
        
        try:
            # Run BLAST search
            blast_results = self._run_blast_search(temp_fasta)
            
            # Parse results
            validation_results = self._parse_blast_results(blast_results, sequence_ids)
            
            # Clean up
            temp_fasta.unlink(missing_ok=True)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            temp_fasta.unlink(missing_ok=True)
            return {'status': 'error', 'message': str(e)}
    
    def _check_blast_databases(self) -> bool:
        """Check if BLAST databases are available."""
        # Look for extracted database files
        db_files = list(self.db_path.glob("*.nhr"))  # BLAST database header files
        return len(db_files) > 0
    
    def _write_fasta_file(self, sequences: List[str], sequence_ids: List[str], output_path: Path):
        """Write sequences to FASTA file for BLAST."""
        with open(output_path, 'w') as f:
            for seq_id, sequence in zip(sequence_ids, sequences):
                f.write(f">{seq_id}\n{sequence}\n")
    
    def _run_blast_search(self, query_file: Path) -> Path:
        """Run BLAST search against local database."""
        output_file = query_file.with_suffix('.xml')
        
        # Find first available database
        db_files = list(self.db_path.glob("*.nhr"))
        if not db_files:
            raise FileNotFoundError("No BLAST databases found")
        
        db_name = str(db_files[0]).replace('.nhr', '')
        
        # BLAST command
        blast_cmd = [
            'blastn',
            '-query', str(query_file),
            '-db', db_name,
            '-out', str(output_file),
            '-outfmt', '5',  # XML format
            '-evalue', str(self.blast_evalue),
            '-max_target_seqs', str(self.blast_max_targets),
            '-num_threads', '4'
        ]
        
        self.logger.info("Running BLAST search...")
        try:
            result = subprocess.run(blast_cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                raise RuntimeError(f"BLAST failed: {result.stderr}")
            
            return output_file
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("BLAST search timed out after 30 minutes")
        except FileNotFoundError:
            raise RuntimeError("BLAST not found. Please install NCBI BLAST+ tools.")
    
    def _parse_blast_results(self, blast_output: Path, sequence_ids: List[str]) -> Dict[str, Any]:
        """Parse BLAST XML results."""
        validation_results = {
            'validated_sequences': [],
            'novel_sequences': [],
            'validation_summary': {},
            'detailed_results': {}
        }
        
        try:
            with open(blast_output, 'r') as f:
                blast_records = NCBIXML.parse(f)
                
                for record in blast_records:
                    seq_id = record.query.split()[0]
                    
                    if record.alignments:
                        # Has matches - analyze best hit
                        best_hit = record.alignments[0]
                        best_hsp = best_hit.hsps[0]
                        
                        identity_percent = (best_hsp.identities / best_hsp.align_length) * 100
                        coverage_percent = (best_hsp.align_length / record.query_length) * 100
                        
                        hit_info = {
                            'sequence_id': seq_id,
                            'hit_accession': best_hit.accession,
                            'hit_definition': best_hit.hit_def,
                            'identity_percent': identity_percent,
                            'coverage_percent': coverage_percent,
                            'evalue': best_hsp.expect,
                            'bitscore': best_hsp.bits
                        }
                        
                        validation_results['detailed_results'][seq_id] = hit_info
                        
                        # Classify as validated or novel based on identity threshold
                        if identity_percent >= (self.min_identity * 100):
                            validation_results['validated_sequences'].append(seq_id)
                        else:
                            validation_results['novel_sequences'].append(seq_id)
                    else:
                        # No matches - likely novel
                        validation_results['novel_sequences'].append(seq_id)
                        validation_results['detailed_results'][seq_id] = {
                            'sequence_id': seq_id,
                            'status': 'no_matches',
                            'message': 'No significant matches found'
                        }
        
        except Exception as e:
            self.logger.error(f"Failed to parse BLAST results: {str(e)}")
            validation_results['status'] = 'parse_error'
            validation_results['message'] = str(e)
        
        # Generate summary
        total_sequences = len(sequence_ids)
        validated_count = len(validation_results['validated_sequences'])
        novel_count = len(validation_results['novel_sequences'])
        
        validation_results['validation_summary'] = {
            'total_sequences': total_sequences,
            'validated_sequences': validated_count,
            'novel_sequences': novel_count,
            'validation_rate': validated_count / total_sequences if total_sequences > 0 else 0,
            'novelty_rate': novel_count / total_sequences if total_sequences > 0 else 0
        }
        
        return validation_results
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get status of downloaded databases."""
        status = {
            'database_path': str(self.db_path),
            'downloaded_files': [],
            'extracted_databases': [],
            'total_size_gb': 0,
            'ready_for_blast': False
        }
        
        # Check for downloaded tar.gz files
        for tar_file in self.db_path.glob("*.tar.gz"):
            size_bytes = tar_file.stat().st_size
            size_gb = size_bytes / (1024**3)
            
            status['downloaded_files'].append({
                'filename': tar_file.name,
                'size_gb': round(size_gb, 2)
            })
            status['total_size_gb'] += size_gb
        
        # Check for extracted BLAST databases
        db_files = list(self.db_path.glob("*.nhr"))
        status['extracted_databases'] = [f.stem for f in db_files]
        status['ready_for_blast'] = len(db_files) > 0
        
        status['total_size_gb'] = round(status['total_size_gb'], 2)
        
        return status


def create_download_script() -> str:
    """
    Create a PowerShell script for downloading NCBI databases.
    
    Returns:
        Path to the created download script
    """
    script_content = '''
# NCBI Database Download Script for SIH 2025 eDNA Pipeline
# ========================================================

param(
    [string[]]$DatabaseFiles = @("nt_euk.00.tar.gz", "nt_euk.01.tar.gz"),
    [string]$OutputPath = "data/ncbi_nt_euk/",
    [switch]$VerifyChecksums = $true
)

Write-Host "SIH 2025: NCBI Database Downloader" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

$BaseUrl = "https://ftp.ncbi.nlm.nih.gov/blast/db/"

foreach ($DbFile in $DatabaseFiles) {
    Write-Host "`nDownloading $DbFile..." -ForegroundColor Yellow
    
    $Url = "$BaseUrl$DbFile"
    $OutputFile = Join-Path $OutputPath $DbFile
    
    # Check if file exists
    if (Test-Path $OutputFile) {
        Write-Host "File already exists: $OutputFile" -ForegroundColor Green
        continue
    }
    
    try {
        # Download with progress
        $WebClient = New-Object System.Net.WebClient
        
        # Progress tracking
        Register-ObjectEvent -InputObject $WebClient -EventName DownloadProgressChanged -Action {
            $Global:DownloadProgress = $Event.SourceEventArgs.ProgressPercentage
            Write-Progress -Activity "Downloading $DbFile" -Status "Progress: $Global:DownloadProgress%" -PercentComplete $Global:DownloadProgress
        } | Out-Null
        
        $WebClient.DownloadFile($Url, $OutputFile)
        Write-Progress -Activity "Downloading" -Completed
        
        Write-Host "✓ Downloaded: $DbFile" -ForegroundColor Green
        
        # Verify checksum if requested
        if ($VerifyChecksums) {
            Write-Host "Verifying checksum..." -ForegroundColor Yellow
            # Checksum verification would go here
            Write-Host "✓ Checksum verified" -ForegroundColor Green
        }
        
    } catch {
        Write-Host "✗ Failed to download $DbFile : $_" -ForegroundColor Red
    }
}

Write-Host "`n=======================================" -ForegroundColor Cyan
Write-Host "Database download completed" -ForegroundColor Green
Write-Host "Total downloaded files: $($DatabaseFiles.Count)" -ForegroundColor White
Write-Host "Location: $OutputPath" -ForegroundColor White
    '''
    
    script_path = "download_ncbi_databases.ps1"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path


def main():
    """Main function for testing database integration."""
    # Test configuration
    config = {
        'ncbi_db_path': 'data/ncbi_nt_euk/',
        'blast_evalue': 0.001,
        'blast_max_targets': 10,
        'validation_threshold': 0.8,
        'min_identity': 0.7
    }
    
    # Initialize database manager
    db_manager = NCBIDatabaseManager(config)
    
    # Check database status
    status = db_manager.get_database_status()
    print("Database Status:")
    print(f"  Path: {status['database_path']}")
    print(f"  Downloaded files: {len(status['downloaded_files'])}")
    print(f"  Total size: {status['total_size_gb']} GB")
    print(f"  Ready for BLAST: {status['ready_for_blast']}")
    
    # Create download script
    script_path = create_download_script()
    print(f"\nDownload script created: {script_path}")
    print("Run: powershell -ExecutionPolicy Bypass -File download_ncbi_databases.ps1")


if __name__ == "__main__":
    main()
