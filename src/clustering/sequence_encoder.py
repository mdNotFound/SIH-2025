#!/usr/bin/env python3
"""
Sequence Encoder for Deep-Sea eDNA Analysis
==========================================

This module provides methods to encode DNA sequences into numerical vectors
for machine learning analysis, focusing on k-mer based representations
and other sequence embedding techniques.

Author: SIH 2025 Team
Date: 2025-09-09
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter, defaultdict
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path


class SequenceEncoder:
    """
    Encodes DNA sequences into numerical vectors using various methods.
    
    This class provides multiple encoding strategies:
    - K-mer frequency encoding
    - One-hot encoding
    - Compositional features
    - TF-IDF based encoding
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SequenceEncoder.
        
        Args:
            config: Configuration dictionary with encoding parameters
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Encoding parameters
        self.encoding_method = config.get('encoding_method', 'kmer')
        self.kmer_size = config.get('kmer_size', 6)
        self.max_features = config.get('max_features', 10000)
        
        # Initialize encoders
        self.kmer_vectorizer = None
        self.scaler = None
        self.kmer_vocab = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the encoder."""
        logger = logging.getLogger('SequenceEncoder')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def fit_transform(self, sequences: List[str]) -> np.ndarray:
        """
        Fit encoder and transform sequences to numerical vectors.
        
        Args:
            sequences: List of DNA sequence strings
            
        Returns:
            Numerical array representation of sequences
        """
        self.logger.info(f"Fitting encoder on {len(sequences)} sequences")
        
        if self.encoding_method == 'kmer':
            return self._fit_transform_kmer(sequences)
        elif self.encoding_method == 'onehot':
            return self._fit_transform_onehot(sequences)
        elif self.encoding_method == 'composition':
            return self._fit_transform_composition(sequences)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")
    
    def transform(self, sequences: List[str]) -> np.ndarray:
        """
        Transform sequences using fitted encoder.
        
        Args:
            sequences: List of DNA sequence strings
            
        Returns:
            Numerical array representation of sequences
        """
        if self.encoding_method == 'kmer':
            return self._transform_kmer(sequences)
        elif self.encoding_method == 'onehot':
            return self._transform_onehot(sequences)
        elif self.encoding_method == 'composition':
            return self._transform_composition(sequences)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")
    
    def _fit_transform_kmer(self, sequences: List[str]) -> np.ndarray:
        """Fit and transform sequences using k-mer frequency encoding."""
        self.logger.info(f"Using k-mer encoding with k={self.kmer_size}")
        
        # Generate k-mer sequences from DNA sequences
        kmer_docs = [self._sequence_to_kmers(seq) for seq in sequences]
        
        # Initialize TF-IDF vectorizer for k-mers
        self.kmer_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 1),  # Use individual k-mers
            token_pattern=r'\b\w+\b'
        )
        
        # Fit and transform k-mer documents
        kmer_matrix = self.kmer_vectorizer.fit_transform(kmer_docs)
        
        # Convert to dense array and normalize
        dense_matrix = kmer_matrix.toarray()
        
        # Scale features
        self.scaler = StandardScaler()
        normalized_matrix = self.scaler.fit_transform(dense_matrix)
        
        self.logger.info(f"Generated k-mer matrix: {normalized_matrix.shape}")
        return normalized_matrix
    
    def _transform_kmer(self, sequences: List[str]) -> np.ndarray:
        """Transform sequences using fitted k-mer encoder."""
        if self.kmer_vectorizer is None or self.scaler is None:
            raise ValueError("Encoder not fitted. Call fit_transform first.")
        
        # Generate k-mer sequences
        kmer_docs = [self._sequence_to_kmers(seq) for seq in sequences]
        
        # Transform using fitted vectorizer
        kmer_matrix = self.kmer_vectorizer.transform(kmer_docs)
        dense_matrix = kmer_matrix.toarray()
        
        # Scale using fitted scaler
        normalized_matrix = self.scaler.transform(dense_matrix)
        
        return normalized_matrix
    
    def _sequence_to_kmers(self, sequence: str) -> str:
        """
        Convert DNA sequence to k-mer document string.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Space-separated string of k-mers
        """
        if len(sequence) < self.kmer_size:
            return ""
        
        kmers = []
        for i in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[i:i + self.kmer_size]
            # Only include k-mers with valid nucleotides
            if all(nt in 'ATGC' for nt in kmer):
                kmers.append(kmer)
        
        return ' '.join(kmers)
    
    def _fit_transform_onehot(self, sequences: List[str]) -> np.ndarray:
        """Fit and transform sequences using one-hot encoding."""
        self.logger.info("Using one-hot encoding")
        
        # Determine maximum sequence length
        max_length = max(len(seq) for seq in sequences) if sequences else 0
        
        # Create one-hot encoded matrix
        nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        encoded_sequences = []
        
        for seq in sequences:
            # Pad or truncate sequence to max_length
            padded_seq = seq[:max_length].ljust(max_length, 'N')
            
            # Create one-hot encoding
            one_hot = np.zeros((max_length, 4))
            for i, nt in enumerate(padded_seq):
                if nt in nucleotide_map:
                    one_hot[i, nucleotide_map[nt]] = 1
                # 'N' or unknown nucleotides remain as zero vectors
            
            # Flatten to 1D vector
            encoded_sequences.append(one_hot.flatten())
        
        matrix = np.array(encoded_sequences)
        
        # Scale features
        self.scaler = StandardScaler()
        normalized_matrix = self.scaler.fit_transform(matrix)
        
        self.logger.info(f"Generated one-hot matrix: {normalized_matrix.shape}")
        return normalized_matrix
    
    def _transform_onehot(self, sequences: List[str]) -> np.ndarray:
        """Transform sequences using fitted one-hot encoder."""
        if self.scaler is None:
            raise ValueError("Encoder not fitted. Call fit_transform first.")
        
        # This assumes the same max_length as during fitting
        # In practice, you'd want to store this parameter
        max_length = self.scaler.n_features_in_ // 4
        
        nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        encoded_sequences = []
        
        for seq in sequences:
            padded_seq = seq[:max_length].ljust(max_length, 'N')
            
            one_hot = np.zeros((max_length, 4))
            for i, nt in enumerate(padded_seq):
                if nt in nucleotide_map:
                    one_hot[i, nucleotide_map[nt]] = 1
            
            encoded_sequences.append(one_hot.flatten())
        
        matrix = np.array(encoded_sequences)
        normalized_matrix = self.scaler.transform(matrix)
        
        return normalized_matrix
    
    def _fit_transform_composition(self, sequences: List[str]) -> np.ndarray:
        """Fit and transform sequences using compositional features."""
        self.logger.info("Using compositional feature encoding")
        
        feature_matrix = []
        
        for seq in sequences:
            features = self._extract_compositional_features(seq)
            feature_matrix.append(features)
        
        matrix = np.array(feature_matrix)
        
        # Scale features
        self.scaler = StandardScaler()
        normalized_matrix = self.scaler.fit_transform(matrix)
        
        self.logger.info(f"Generated compositional matrix: {normalized_matrix.shape}")
        return normalized_matrix
    
    def _transform_composition(self, sequences: List[str]) -> np.ndarray:
        """Transform sequences using fitted compositional encoder."""
        if self.scaler is None:
            raise ValueError("Encoder not fitted. Call fit_transform first.")
        
        feature_matrix = []
        
        for seq in sequences:
            features = self._extract_compositional_features(seq)
            feature_matrix.append(features)
        
        matrix = np.array(feature_matrix)
        normalized_matrix = self.scaler.transform(matrix)
        
        return normalized_matrix
    
    def _extract_compositional_features(self, sequence: str) -> List[float]:
        """
        Extract compositional features from a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            List of compositional feature values
        """
        if not sequence:
            return [0.0] * 20  # Return zeros for empty sequences
        
        seq_len = len(sequence)
        counts = Counter(sequence)
        
        # Basic nucleotide composition
        a_freq = counts.get('A', 0) / seq_len
        t_freq = counts.get('T', 0) / seq_len
        g_freq = counts.get('G', 0) / seq_len
        c_freq = counts.get('C', 0) / seq_len
        
        # Derived composition features
        gc_content = g_freq + c_freq
        at_content = a_freq + t_freq
        purine_content = a_freq + g_freq  # A, G
        pyrimidine_content = c_freq + t_freq  # C, T
        
        # Dinucleotide frequencies (selected subset)
        dinucs = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                  'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        dinuc_freqs = []
        
        for dinuc in dinucs:
            count = 0
            for i in range(len(sequence) - 1):
                if sequence[i:i+2] == dinuc:
                    count += 1
            freq = count / max(1, len(sequence) - 1)
            dinuc_freqs.append(freq)
        
        # Combine all features
        features = [
            seq_len,
            a_freq, t_freq, g_freq, c_freq,
            gc_content, at_content,
            purine_content, pyrimidine_content
        ]
        features.extend(dinuc_freqs[:8])  # Take first 8 dinucleotide frequencies
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features generated by the encoder.
        
        Returns:
            List of feature names
        """
        if self.encoding_method == 'kmer' and self.kmer_vectorizer:
            return list(self.kmer_vectorizer.get_feature_names_out())
        elif self.encoding_method == 'onehot':
            # Generate one-hot feature names
            nucleotides = ['A', 'T', 'G', 'C']
            max_length = self.scaler.n_features_in_ // 4 if self.scaler else 100
            names = []
            for pos in range(max_length):
                for nt in nucleotides:
                    names.append(f'pos_{pos}_{nt}')
            return names
        elif self.encoding_method == 'composition':
            # Compositional feature names
            names = [
                'length', 'A_freq', 'T_freq', 'G_freq', 'C_freq',
                'GC_content', 'AT_content', 'purine_content', 'pyrimidine_content'
            ]
            dinucs = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC']
            names.extend([f'dinuc_{dinuc}' for dinuc in dinucs])
            return names
        else:
            return []
    
    def save_encoder(self, filepath: str):
        """
        Save the fitted encoder to disk.
        
        Args:
            filepath: Path to save the encoder
        """
        encoder_data = {
            'encoding_method': self.encoding_method,
            'kmer_size': self.kmer_size,
            'max_features': self.max_features,
            'kmer_vectorizer': self.kmer_vectorizer,
            'scaler': self.scaler,
            'config': self.config
        }
        
        joblib.dump(encoder_data, filepath)
        self.logger.info(f"Encoder saved to {filepath}")
    
    def load_encoder(self, filepath: str):
        """
        Load a fitted encoder from disk.
        
        Args:
            filepath: Path to the saved encoder
        """
        encoder_data = joblib.load(filepath)
        
        self.encoding_method = encoder_data['encoding_method']
        self.kmer_size = encoder_data['kmer_size']
        self.max_features = encoder_data['max_features']
        self.kmer_vectorizer = encoder_data['kmer_vectorizer']
        self.scaler = encoder_data['scaler']
        self.config = encoder_data['config']
        
        self.logger.info(f"Encoder loaded from {filepath}")


def main():
    """Main function for testing the encoder."""
    # Test sequences
    sequences = [
        "ATGCGTACGATCGTAGC",
        "CGATCGATCGATCGAT",
        "AAATTTGGGCCCAAAA",
        "TGCATGCATGCATGCA"
    ]
    
    # Test configuration
    config = {
        'encoding_method': 'kmer',
        'kmer_size': 4,
        'max_features': 1000
    }
    
    # Initialize encoder
    encoder = SequenceEncoder(config)
    
    # Fit and transform
    encoded_matrix = encoder.fit_transform(sequences)
    print(f"Encoded matrix shape: {encoded_matrix.shape}")
    print(f"Number of features: {len(encoder.get_feature_names())}")
    
    # Test transform on new sequences
    new_sequences = ["ATGCATGCATGC", "CGATCGATCGAT"]
    new_encoded = encoder.transform(new_sequences)
    print(f"New encoded matrix shape: {new_encoded.shape}")


if __name__ == "__main__":
    main()
