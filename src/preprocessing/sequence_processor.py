#!/usr/bin/env python3
"""
Sequence Processing Module for Deep-Sea eDNA Analysis
====================================================

This module handles preprocessing of raw eDNA sequencing reads including:
- Quality control and filtering
- Sequence trimming and cleaning
- Format conversion
- Duplicate removal
- Sequence length filtering

Author: SIH 2025 Team
Date: 2025-09-09
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import multiprocessing as mp
from collections import Counter
import hashlib
import gzip


class SequenceProcessor:
    """
    Main class for processing raw eDNA sequencing reads.
    
    This class provides methods for quality control, filtering, and preprocessing
    of eDNA sequences before machine learning analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SequenceProcessor.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Processing parameters
        self.min_length = config.get('min_sequence_length', 100)
        self.max_length = config.get('max_sequence_length', 2000)
        self.min_quality = config.get('min_quality_score', 20)
        self.quality_window = config.get('quality_window_size', 4)
        self.n_threads = config.get('n_threads', mp.cpu_count())
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the processor."""
        logger = logging.getLogger('SequenceProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def process_fastq_files(self, input_path: str, output_path: str) -> Dict:
        """
        Process FASTQ files from input directory.
        
        Args:
            input_path: Path to directory containing FASTQ files
            output_path: Path to output directory for processed files
            
        Returns:
            Dictionary containing processing statistics
        """
        self.logger.info(f"Starting FASTQ processing from {input_path}")
        
        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Find all FASTQ files
        fastq_files = self._find_fastq_files(input_path)
        self.logger.info(f"Found {len(fastq_files)} FASTQ files")
        
        # Process files
        stats = {}
        for file_path in fastq_files:
            file_stats = self._process_single_fastq(file_path, output_path)
            stats[Path(file_path).name] = file_stats
            
        # Save processing summary
        self._save_processing_summary(stats, output_path)
        
        return stats
    
    def _find_fastq_files(self, directory: str) -> List[str]:
        """Find all FASTQ files in directory."""
        extensions = ['.fastq', '.fq', '.fastq.gz', '.fq.gz']
        fastq_files = []
        
        for ext in extensions:
            pattern = f"*{ext}"
            files = list(Path(directory).glob(pattern))
            fastq_files.extend([str(f) for f in files])
            
        return sorted(fastq_files)
    
    def _process_single_fastq(self, file_path: str, output_path: str) -> Dict:
        """
        Process a single FASTQ file.
        
        Args:
            file_path: Path to input FASTQ file
            output_path: Output directory path
            
        Returns:
            Processing statistics dictionary
        """
        self.logger.info(f"Processing {file_path}")
        
        # Setup file paths
        input_file = Path(file_path)
        output_file = Path(output_path) / f"{input_file.stem}_processed.fasta"
        
        # Initialize statistics
        stats = {
            'total_sequences': 0,
            'passed_quality': 0,
            'passed_length': 0,
            'duplicates_removed': 0,
            'final_sequences': 0
        }
        
        # Process sequences
        processed_sequences = []
        seen_sequences = set()
        
        # Open input file (handle gzipped files)
        if file_path.endswith('.gz'):
            file_handle = gzip.open(file_path, 'rt')
        else:
            file_handle = open(file_path, 'r')
            
        try:
            for record in SeqIO.parse(file_handle, "fastq"):
                stats['total_sequences'] += 1
                
                # Quality filtering
                if not self._passes_quality_filter(record):
                    continue
                stats['passed_quality'] += 1
                
                # Length filtering
                if not self._passes_length_filter(record):
                    continue
                stats['passed_length'] += 1
                
                # Clean sequence
                cleaned_seq = self._clean_sequence(str(record.seq))
                
                # Remove duplicates
                seq_hash = self._get_sequence_hash(cleaned_seq)
                if seq_hash in seen_sequences:
                    stats['duplicates_removed'] += 1
                    continue
                seen_sequences.add(seq_hash)
                
                # Create processed record
                processed_record = SeqRecord(
                    Seq(cleaned_seq),
                    id=record.id,
                    description=f"processed|length={len(cleaned_seq)}"
                )
                processed_sequences.append(processed_record)
                stats['final_sequences'] += 1
                
        finally:
            file_handle.close()
        
        # Write processed sequences
        if processed_sequences:
            SeqIO.write(processed_sequences, output_file, "fasta")
            self.logger.info(f"Wrote {len(processed_sequences)} sequences to {output_file}")
        
        return stats
    
    def _passes_quality_filter(self, record: SeqRecord) -> bool:
        """
        Check if sequence passes quality filtering.
        
        Args:
            record: BioPython SeqRecord object
            
        Returns:
            True if sequence passes quality filter
        """
        if not hasattr(record, 'letter_annotations') or 'phred_quality' not in record.letter_annotations:
            # If no quality scores available, assume it passes
            return True
        
        quality_scores = record.letter_annotations['phred_quality']
        
        # Sliding window quality check
        for i in range(len(quality_scores) - self.quality_window + 1):
            window_scores = quality_scores[i:i + self.quality_window]
            if np.mean(window_scores) < self.min_quality:
                return False
                
        return True
    
    def _passes_length_filter(self, record: SeqRecord) -> bool:
        """
        Check if sequence passes length filtering.
        
        Args:
            record: BioPython SeqRecord object
            
        Returns:
            True if sequence passes length filter
        """
        seq_length = len(record.seq)
        return self.min_length <= seq_length <= self.max_length
    
    def _clean_sequence(self, sequence: str) -> str:
        """
        Clean sequence by removing ambiguous nucleotides and converting to uppercase.
        
        Args:
            sequence: Raw DNA sequence string
            
        Returns:
            Cleaned sequence string
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Remove ambiguous nucleotides (keep only A, T, G, C)
        valid_nucleotides = set('ATGC')
        cleaned_sequence = ''.join([nt for nt in sequence if nt in valid_nucleotides])
        
        return cleaned_sequence
    
    def _get_sequence_hash(self, sequence: str) -> str:
        """Generate hash for sequence deduplication."""
        return hashlib.md5(sequence.encode()).hexdigest()
    
    def _save_processing_summary(self, stats: Dict, output_path: str):
        """Save processing summary to file."""
        summary_file = Path(output_path) / "processing_summary.tsv"
        
        # Convert stats to DataFrame
        df_data = []
        for filename, file_stats in stats.items():
            row = {'filename': filename}
            row.update(file_stats)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(summary_file, sep='\t', index=False)
        
        self.logger.info(f"Processing summary saved to {summary_file}")
    
    def extract_marker_sequences(self, input_path: str, marker_type: str, output_path: str) -> Dict:
        """
        Extract specific marker gene sequences (18S, COI, ITS).
        
        Args:
            input_path: Path to processed FASTA files
            marker_type: Type of marker gene ('18S', 'COI', 'ITS')
            output_path: Output path for extracted sequences
            
        Returns:
            Extraction statistics
        """
        self.logger.info(f"Extracting {marker_type} marker sequences")
        
        # Marker-specific parameters
        marker_params = self._get_marker_parameters(marker_type)
        
        # Find processed FASTA files
        fasta_files = list(Path(input_path).glob("*_processed.fasta"))
        
        stats = {
            'total_files': len(fasta_files),
            'sequences_extracted': 0,
            'files_processed': 0
        }
        
        # Create output directory
        marker_output_path = Path(output_path) / f"{marker_type}_sequences"
        marker_output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for fasta_file in fasta_files:
            file_stats = self._extract_marker_from_file(
                fasta_file, marker_params, marker_output_path
            )
            stats['sequences_extracted'] += file_stats['extracted']
            stats['files_processed'] += 1
        
        return stats
    
    def _get_marker_parameters(self, marker_type: str) -> Dict:
        """Get parameters for specific marker genes."""
        markers = {
            '18S': {
                'min_length': 300,
                'max_length': 1800,
                'primers': ['ACCTGGTTGATCCTGCCAG', 'TGATCCTTCTGCAGGTTCACCTAC']
            },
            'COI': {
                'min_length': 400,
                'max_length': 800,
                'primers': ['GGTCAACAAATCATAAAGATATTGG', 'TAAACTTCAGGGTGACCAAAAAATCA']
            },
            'ITS': {
                'min_length': 200,
                'max_length': 1000,
                'primers': ['TCCGTAGGTGAACCTGCGG', 'TCCTCCGCTTATTGATATGC']
            }
        }
        
        return markers.get(marker_type, markers['18S'])
    
    def _extract_marker_from_file(self, fasta_file: Path, marker_params: Dict, output_path: Path) -> Dict:
        """Extract marker sequences from a single FASTA file."""
        output_file = output_path / f"{fasta_file.stem}_marker.fasta"
        
        extracted_sequences = []
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_str = str(record.seq)
            
            # Check length constraints
            if not (marker_params['min_length'] <= len(seq_str) <= marker_params['max_length']):
                continue
            
            # Simple primer-based detection (can be improved with HMM models)
            if self._contains_marker_signature(seq_str, marker_params['primers']):
                extracted_sequences.append(record)
        
        # Write extracted sequences
        if extracted_sequences:
            SeqIO.write(extracted_sequences, output_file, "fasta")
        
        return {'extracted': len(extracted_sequences)}
    
    def _contains_marker_signature(self, sequence: str, primers: List[str]) -> bool:
        """Check if sequence contains marker-specific signatures."""
        # Simple implementation - check for primer sequences or their reverse complements
        # This can be improved with more sophisticated methods
        
        for primer in primers:
            # Check forward primer
            if primer in sequence:
                return True
            
            # Check reverse complement
            rev_comp = str(Seq(primer).reverse_complement())
            if rev_comp in sequence:
                return True
        
        return False
    
    def generate_sequence_features(self, input_path: str, output_path: str) -> str:
        """
        Generate numerical features from sequences for ML models.
        
        Args:
            input_path: Path to processed sequences
            output_path: Output path for feature files
            
        Returns:
            Path to generated feature file
        """
        self.logger.info("Generating sequence features for ML models")
        
        # Find all processed FASTA files
        fasta_files = list(Path(input_path).glob("*.fasta"))
        
        all_features = []
        sequence_ids = []
        
        for fasta_file in fasta_files:
            for record in SeqIO.parse(fasta_file, "fasta"):
                features = self._extract_sequence_features(str(record.seq))
                all_features.append(features)
                sequence_ids.append(record.id)
        
        # Create feature DataFrame
        feature_columns = [
            'length', 'gc_content', 'at_content',
            'purine_content', 'pyrimidine_content',
            'complexity', 'entropy'
        ]
        
        df_features = pd.DataFrame(all_features, columns=feature_columns)
        df_features.insert(0, 'sequence_id', sequence_ids)
        
        # Save features
        feature_file = Path(output_path) / "sequence_features.tsv"
        df_features.to_csv(feature_file, sep='\t', index=False)
        
        self.logger.info(f"Generated features for {len(all_features)} sequences")
        return str(feature_file)
    
    def _extract_sequence_features(self, sequence: str) -> List[float]:
        """Extract numerical features from a DNA sequence."""
        seq_len = len(sequence)
        
        if seq_len == 0:
            return [0.0] * 7
        
        # Nucleotide counts
        counts = Counter(sequence)
        gc_count = counts.get('G', 0) + counts.get('C', 0)
        at_count = counts.get('A', 0) + counts.get('T', 0)
        
        # Calculate features
        features = [
            float(seq_len),  # Length
            gc_count / seq_len,  # GC content
            at_count / seq_len,  # AT content
            (counts.get('A', 0) + counts.get('G', 0)) / seq_len,  # Purine content
            (counts.get('C', 0) + counts.get('T', 0)) / seq_len,  # Pyrimidine content
            self._calculate_complexity(sequence),  # Linguistic complexity
            self._calculate_entropy(sequence)  # Shannon entropy
        ]
        
        return features
    
    def _calculate_complexity(self, sequence: str) -> float:
        """Calculate linguistic complexity of sequence."""
        if len(sequence) < 4:
            return 0.0
        
        # Count unique k-mers (k=4)
        k = 4
        kmers = set()
        for i in range(len(sequence) - k + 1):
            kmers.add(sequence[i:i+k])
        
        max_possible = min(4**k, len(sequence) - k + 1)
        return len(kmers) / max_possible if max_possible > 0 else 0.0
    
    def _calculate_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of sequence."""
        if not sequence:
            return 0.0
        
        counts = Counter(sequence)
        probabilities = [count / len(sequence) for count in counts.values()]
        
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy


def main():
    """Main function for command-line usage."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Process eDNA sequences")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--input", required=True, help="Input directory path")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--marker", choices=['18S', 'COI', 'ITS'], help="Extract specific marker")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    processor = SequenceProcessor(config['preprocessing'])
    
    # Process sequences
    stats = processor.process_fastq_files(args.input, args.output)
    print(f"Processing completed. Statistics: {stats}")
    
    # Extract marker sequences if specified
    if args.marker:
        marker_stats = processor.extract_marker_sequences(args.output, args.marker, args.output)
        print(f"Marker extraction completed. Statistics: {marker_stats}")
    
    # Generate features
    feature_file = processor.generate_sequence_features(args.output, args.output)
    print(f"Features generated: {feature_file}")


if __name__ == "__main__":
    main()
