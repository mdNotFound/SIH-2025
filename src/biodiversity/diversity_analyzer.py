#!/usr/bin/env python3
"""
Biodiversity Analysis Module for Deep-Sea eDNA
==============================================

This module provides comprehensive biodiversity assessment tools including:
- Diversity indices (Shannon, Simpson, Chao1, etc.)
- Community structure analysis
- Abundance estimation
- Beta diversity calculations
- Statistical tests for biodiversity differences

Author: SIH 2025 Team
Date: 2025-09-09
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import networkx as nx
from collections import Counter
import warnings


class DiversityAnalyzer:
    """
    Comprehensive biodiversity analysis for eDNA sequences.
    
    This class provides methods for:
    - Alpha diversity calculation
    - Beta diversity analysis
    - Community structure assessment
    - Statistical testing
    - Ecological network analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DiversityAnalyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Analysis parameters
        self.diversity_indices = config.get('calculate_indices', ['shannon', 'simpson', 'chao1'])
        self.abundance_method = config.get('abundance_method', 'rarefaction')
        self.rarefaction_depth = config.get('rarefaction_depth', 1000)
        self.beta_diversity = config.get('beta_diversity', True)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the analyzer."""
        logger = logging.getLogger('DiversityAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def analyze_community(
        self, 
        abundance_data: pd.DataFrame,
        metadata: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Comprehensive community analysis.
        
        Args:
            abundance_data: DataFrame with samples as rows and taxa as columns
            metadata: Optional metadata DataFrame with sample information
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info(f"Starting community analysis for {abundance_data.shape[0]} samples")
        
        results = {
            'alpha_diversity': {},
            'beta_diversity': {},
            'community_structure': {},
            'statistical_tests': {},
            'network_analysis': {}
        }
        
        # Alpha diversity analysis
        self.logger.info("Calculating alpha diversity indices")
        results['alpha_diversity'] = self._calculate_alpha_diversity(abundance_data)
        
        # Beta diversity analysis
        if self.beta_diversity and abundance_data.shape[0] > 1:
            self.logger.info("Calculating beta diversity")
            results['beta_diversity'] = self._calculate_beta_diversity(abundance_data)
        
        # Community structure analysis
        self.logger.info("Analyzing community structure")
        results['community_structure'] = self._analyze_community_structure(abundance_data)
        
        # Statistical tests
        if metadata is not None:
            self.logger.info("Performing statistical tests")
            results['statistical_tests'] = self._perform_statistical_tests(
                abundance_data, metadata, results['alpha_diversity']
            )
        
        # Network analysis
        self.logger.info("Performing network analysis")
        results['network_analysis'] = self._perform_network_analysis(abundance_data)
        
        # Summary statistics
        results['summary'] = self._generate_summary(abundance_data, results)
        
        self.logger.info("Community analysis completed")
        return results
    
    def _calculate_alpha_diversity(self, abundance_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various alpha diversity indices."""
        diversity_results = {}
        
        for index in self.diversity_indices:
            if index == 'shannon':
                diversity_results['shannon'] = self._shannon_diversity(abundance_data)
            elif index == 'simpson':
                diversity_results['simpson'] = self._simpson_diversity(abundance_data)
            elif index == 'chao1':
                diversity_results['chao1'] = self._chao1_richness(abundance_data)
            elif index == 'ace':
                diversity_results['ace'] = self._ace_richness(abundance_data)
            elif index == 'observed_species':
                diversity_results['observed_species'] = self._observed_species(abundance_data)
            elif index == 'pielou_evenness':
                diversity_results['pielou_evenness'] = self._pielou_evenness(abundance_data)
        
        return diversity_results
    
    def _shannon_diversity(self, abundance_data: pd.DataFrame) -> pd.Series:
        """Calculate Shannon diversity index."""
        def shannon_sample(row):
            # Remove zeros
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0
            
            # Calculate proportions
            proportions = abundances / abundances.sum()
            
            # Shannon index: H = -sum(p * log(p))
            return -np.sum(proportions * np.log(proportions))
        
        return abundance_data.apply(shannon_sample, axis=1)
    
    def _simpson_diversity(self, abundance_data: pd.DataFrame) -> pd.Series:
        """Calculate Simpson diversity index."""
        def simpson_sample(row):
            # Remove zeros
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0
            
            # Calculate proportions
            proportions = abundances / abundances.sum()
            
            # Simpson index: D = 1 - sum(p^2)
            return 1 - np.sum(proportions ** 2)
        
        return abundance_data.apply(simpson_sample, axis=1)
    
    def _chao1_richness(self, abundance_data: pd.DataFrame) -> pd.Series:
        """Calculate Chao1 richness estimator."""
        def chao1_sample(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0
            
            # Count singletons and doubletons
            singletons = np.sum(abundances == 1)
            doubletons = np.sum(abundances == 2)
            observed = len(abundances)
            
            # Chao1 estimator
            if doubletons > 0:
                chao1 = observed + (singletons ** 2) / (2 * doubletons)
            else:
                chao1 = observed + (singletons * (singletons - 1)) / 2
            
            return chao1
        
        return abundance_data.apply(chao1_sample, axis=1)
    
    def _ace_richness(self, abundance_data: pd.DataFrame) -> pd.Series:
        """Calculate ACE (Abundance-based Coverage Estimator) richness."""
        def ace_sample(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0
            
            # Separate rare (≤10) and abundant (>10) species
            rare = abundances[abundances <= 10]
            abundant = abundances[abundances > 10]
            
            s_rare = len(rare)
            s_abund = len(abundant)
            n_rare = rare.sum()
            
            if s_rare == 0:
                return s_abund
            
            # Calculate coverage estimate
            f1 = np.sum(rare == 1)
            if n_rare == 0:
                c_ace = 1.0
            else:
                c_ace = 1 - (f1 / n_rare)
            
            if c_ace == 0:
                return s_abund + s_rare
            
            # Calculate coefficient of variation
            freqs = np.bincount(rare.astype(int))[1:11]  # frequencies 1-10
            if np.sum(freqs) == 0:
                gamma_ace = 0
            else:
                i_values = np.arange(1, 11)
                numerator = np.sum(i_values * (i_values - 1) * freqs)
                denominator = n_rare * (n_rare - 1)
                if denominator > 0:
                    gamma_ace = max(0, (s_rare / c_ace) * (numerator / denominator) - 1)
                else:
                    gamma_ace = 0
            
            # ACE estimate
            ace = s_abund + (s_rare / c_ace) + (f1 / c_ace) * gamma_ace
            return ace
        
        return abundance_data.apply(ace_sample, axis=1)
    
    def _observed_species(self, abundance_data: pd.DataFrame) -> pd.Series:
        """Calculate observed species richness."""
        return (abundance_data > 0).sum(axis=1)
    
    def _pielou_evenness(self, abundance_data: pd.DataFrame) -> pd.Series:
        """Calculate Pielou's evenness index."""
        shannon = self._shannon_diversity(abundance_data)
        observed = self._observed_species(abundance_data)
        
        # J = H / ln(S)
        return shannon / np.log(observed.replace(0, np.nan))
    
    def _calculate_beta_diversity(self, abundance_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate beta diversity metrics."""
        beta_results = {}
        
        # Bray-Curtis dissimilarity
        bray_curtis = self._bray_curtis_distance(abundance_data)
        beta_results['bray_curtis'] = bray_curtis
        
        # Jaccard dissimilarity
        jaccard = self._jaccard_distance(abundance_data)
        beta_results['jaccard'] = jaccard
        
        # UniFrac-like distance (simplified, based on phylogenetic relationships)
        unifrac = self._unifrac_distance(abundance_data)
        beta_results['unifrac'] = unifrac
        
        # Ordination analysis
        ordination_method = self.config.get('ordination_method', 'pcoa')
        if ordination_method == 'pcoa':
            ordination = self._pcoa(bray_curtis)
        elif ordination_method == 'nmds':
            ordination = self._nmds(bray_curtis)
        else:
            ordination = self._pcoa(bray_curtis)  # Default to PCoA
        
        beta_results['ordination'] = ordination
        
        return beta_results
    
    def _bray_curtis_distance(self, abundance_data: pd.DataFrame) -> np.ndarray:
        """Calculate Bray-Curtis dissimilarity matrix."""
        def bray_curtis_pairwise(x, y):
            numerator = np.sum(np.abs(x - y))
            denominator = np.sum(x + y)
            return numerator / denominator if denominator > 0 else 0
        
        n_samples = abundance_data.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = bray_curtis_pairwise(abundance_data.iloc[i], abundance_data.iloc[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def _jaccard_distance(self, abundance_data: pd.DataFrame) -> np.ndarray:
        """Calculate Jaccard dissimilarity matrix."""
        # Convert to presence/absence
        presence_absence = (abundance_data > 0).astype(int)
        
        def jaccard_pairwise(x, y):
            intersection = np.sum((x == 1) & (y == 1))
            union = np.sum((x == 1) | (y == 1))
            return 1 - (intersection / union) if union > 0 else 0
        
        n_samples = presence_absence.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = jaccard_pairwise(presence_absence.iloc[i], presence_absence.iloc[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def _unifrac_distance(self, abundance_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate simplified UniFrac-like distance.
        Note: This is a simplified version without phylogenetic tree.
        """
        # For now, use weighted Bray-Curtis as a proxy for UniFrac
        return self._bray_curtis_distance(abundance_data)
    
    def _pcoa(self, distance_matrix: np.ndarray) -> Dict[str, Any]:
        """Perform Principal Coordinates Analysis (PCoA)."""
        # Convert distance matrix to similarity matrix
        max_dist = np.max(distance_matrix)
        similarity_matrix = max_dist - distance_matrix
        
        # Center the matrix
        n = similarity_matrix.shape[0]
        centering_matrix = np.eye(n) - np.ones((n, n)) / n
        centered_matrix = centering_matrix @ similarity_matrix @ centering_matrix
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(centered_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep only positive eigenvalues
        positive_idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[positive_idx]
        eigenvectors = eigenvectors[:, positive_idx]
        
        # Calculate coordinates
        coordinates = eigenvectors * np.sqrt(eigenvalues)
        
        # Calculate explained variance
        total_inertia = np.sum(np.abs(eigenvalues))
        explained_variance = eigenvalues / total_inertia * 100
        
        return {
            'coordinates': coordinates,
            'eigenvalues': eigenvalues,
            'explained_variance': explained_variance,
            'method': 'PCoA'
        }
    
    def _nmds(self, distance_matrix: np.ndarray) -> Dict[str, Any]:
        """Perform Non-metric Multidimensional Scaling (NMDS)."""
        from sklearn.manifold import MDS
        
        # Perform MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coordinates = mds.fit_transform(distance_matrix)
        
        return {
            'coordinates': coordinates,
            'stress': mds.stress_,
            'method': 'NMDS'
        }
    
    def _analyze_community_structure(self, abundance_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze community structure patterns."""
        structure_results = {}
        
        # Rank abundance curves
        structure_results['rank_abundance'] = self._rank_abundance_curves(abundance_data)
        
        # Dominance patterns
        structure_results['dominance'] = self._dominance_analysis(abundance_data)
        
        # Rarity patterns
        structure_results['rarity'] = self._rarity_analysis(abundance_data)
        
        # Community similarity
        structure_results['similarity'] = self._community_similarity(abundance_data)
        
        return structure_results
    
    def _rank_abundance_curves(self, abundance_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate rank abundance curve data."""
        curves = {}
        
        for sample in abundance_data.index:
            abundances = abundance_data.loc[sample]
            abundances = abundances[abundances > 0].sort_values(ascending=False)
            
            ranks = np.arange(1, len(abundances) + 1)
            curves[sample] = {
                'ranks': ranks,
                'abundances': abundances.values,
                'log_abundances': np.log(abundances.values)
            }
        
        return curves
    
    def _dominance_analysis(self, abundance_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dominance patterns in communities."""
        dominance_results = {}
        
        for sample in abundance_data.index:
            abundances = abundance_data.loc[sample]
            total_abundance = abundances.sum()
            
            if total_abundance > 0:
                proportions = abundances / total_abundance
                
                # Identify dominant species (>5% of total abundance)
                dominant_species = proportions[proportions > 0.05]
                
                dominance_results[sample] = {
                    'n_dominant_species': len(dominant_species),
                    'dominant_species_abundance': dominant_species.sum(),
                    'dominance_ratio': dominant_species.max() / proportions[proportions > 0].mean()
                }
            else:
                dominance_results[sample] = {
                    'n_dominant_species': 0,
                    'dominant_species_abundance': 0,
                    'dominance_ratio': 0
                }
        
        return dominance_results
    
    def _rarity_analysis(self, abundance_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze rarity patterns in communities."""
        rarity_results = {}
        
        for sample in abundance_data.index:
            abundances = abundance_data.loc[sample]
            abundances = abundances[abundances > 0]
            
            if len(abundances) > 0:
                # Rare species (singletons and doubletons)
                singletons = np.sum(abundances == 1)
                doubletons = np.sum(abundances == 2)
                
                rarity_results[sample] = {
                    'n_singletons': singletons,
                    'n_doubletons': doubletons,
                    'rare_species_fraction': (singletons + doubletons) / len(abundances)
                }
            else:
                rarity_results[sample] = {
                    'n_singletons': 0,
                    'n_doubletons': 0,
                    'rare_species_fraction': 0
                }
        
        return rarity_results
    
    def _community_similarity(self, abundance_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate community similarity metrics."""
        # Calculate average pairwise similarity
        bray_curtis = self._bray_curtis_distance(abundance_data)
        similarity_matrix = 1 - bray_curtis
        
        # Average similarity (excluding diagonal)
        n = similarity_matrix.shape[0]
        if n > 1:
            avg_similarity = (np.sum(similarity_matrix) - np.trace(similarity_matrix)) / (n * (n - 1))
        else:
            avg_similarity = 1.0
        
        return {
            'average_similarity': avg_similarity,
            'similarity_matrix': similarity_matrix,
            'max_similarity': np.max(similarity_matrix[similarity_matrix < 1.0]) if n > 1 else 1.0,
            'min_similarity': np.min(similarity_matrix[similarity_matrix < 1.0]) if n > 1 else 1.0
        }
    
    def _perform_statistical_tests(
        self,
        abundance_data: pd.DataFrame,
        metadata: pd.DataFrame,
        alpha_diversity: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Perform statistical tests for biodiversity differences."""
        statistical_results = {}
        
        # Test for differences in alpha diversity
        for factor in metadata.columns:
            if metadata[factor].dtype in ['object', 'category']:
                factor_results = {}
                
                for diversity_index in alpha_diversity:
                    diversity_values = alpha_diversity[diversity_index]
                    
                    # Group samples by factor levels
                    groups = []
                    factor_levels = metadata[factor].unique()
                    
                    for level in factor_levels:
                        mask = metadata[factor] == level
                        if mask.any():
                            group_values = diversity_values[mask]
                            groups.append(group_values.dropna().values)
                    
                    # Perform statistical tests
                    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                        # Kruskal-Wallis test (non-parametric)
                        try:
                            kruskal_stat, kruskal_p = stats.kruskal(*groups)
                        except:
                            kruskal_stat, kruskal_p = np.nan, np.nan
                        
                        # ANOVA (parametric)
                        try:
                            anova_stat, anova_p = stats.f_oneway(*groups)
                        except:
                            anova_stat, anova_p = np.nan, np.nan
                        
                        factor_results[diversity_index] = {
                            'kruskal_wallis': {'statistic': kruskal_stat, 'p_value': kruskal_p},
                            'anova': {'statistic': anova_stat, 'p_value': anova_p},
                            'group_means': {level: np.mean(groups[i]) for i, level in enumerate(factor_levels)},
                            'group_sizes': {level: len(groups[i]) for i, level in enumerate(factor_levels)}
                        }
                
                statistical_results[factor] = factor_results
        
        return statistical_results
    
    def _perform_network_analysis(self, abundance_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform ecological network analysis."""
        network_results = {}
        
        # Calculate species co-occurrence network
        correlation_matrix = abundance_data.T.corr()
        
        # Create network from significant correlations
        threshold = 0.5  # Correlation threshold
        G = nx.Graph()
        
        # Add nodes (species)
        species = abundance_data.columns
        G.add_nodes_from(species)
        
        # Add edges for significant correlations
        for i, species1 in enumerate(species):
            for j, species2 in enumerate(species[i+1:], i+1):
                correlation = correlation_matrix.iloc[i, j]
                if abs(correlation) > threshold and not np.isnan(correlation):
                    G.add_edge(species1, species2, weight=correlation)
        
        # Calculate network metrics
        if G.number_of_nodes() > 0:
            network_results = {
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G),
                'clustering_coefficient': nx.average_clustering(G),
                'connected_components': nx.number_connected_components(G),
                'largest_component_size': len(max(nx.connected_components(G), key=len)) if nx.number_connected_components(G) > 0 else 0
            }
            
            # Centrality measures for connected graph
            if nx.is_connected(G):
                network_results['average_path_length'] = nx.average_shortest_path_length(G)
                network_results['diameter'] = nx.diameter(G)
            
            # Node-level metrics
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            network_results['centrality_stats'] = {
                'degree': {
                    'mean': np.mean(list(degree_centrality.values())),
                    'std': np.std(list(degree_centrality.values())),
                    'max': max(degree_centrality.values()) if degree_centrality else 0
                },
                'betweenness': {
                    'mean': np.mean(list(betweenness_centrality.values())),
                    'std': np.std(list(betweenness_centrality.values())),
                    'max': max(betweenness_centrality.values()) if betweenness_centrality else 0
                }
            }
        else:
            network_results = {
                'n_nodes': 0,
                'n_edges': 0,
                'density': 0,
                'clustering_coefficient': 0,
                'connected_components': 0,
                'largest_component_size': 0
            }
        
        return network_results
    
    def _generate_summary(self, abundance_data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'n_samples': abundance_data.shape[0],
            'n_species': abundance_data.shape[1],
            'total_abundance': abundance_data.sum().sum(),
            'samples_with_data': (abundance_data.sum(axis=1) > 0).sum(),
            'species_occurrence_frequency': (abundance_data > 0).sum(axis=0).describe().to_dict()
        }
        
        # Add alpha diversity summary
        if 'alpha_diversity' in results:
            alpha_summary = {}
            for index, values in results['alpha_diversity'].items():
                alpha_summary[index] = values.describe().to_dict()
            summary['alpha_diversity_summary'] = alpha_summary
        
        return summary


def main():
    """Main function for testing the diversity analyzer."""
    # Create test abundance data
    np.random.seed(42)
    
    # Generate synthetic abundance data
    n_samples = 10
    n_species = 50
    
    abundance_matrix = np.random.poisson(lam=2, size=(n_samples, n_species))
    abundance_matrix = abundance_matrix * np.random.binomial(1, 0.3, size=(n_samples, n_species))
    
    abundance_data = pd.DataFrame(
        abundance_matrix,
        index=[f'Sample_{i}' for i in range(n_samples)],
        columns=[f'Species_{i}' for i in range(n_species)]
    )
    
    # Generate test metadata
    metadata = pd.DataFrame({
        'Site': np.random.choice(['Deep', 'Shallow'], n_samples),
        'Season': np.random.choice(['Summer', 'Winter'], n_samples),
        'Depth': np.random.normal(1000, 200, n_samples)
    }, index=abundance_data.index)
    
    # Test configuration
    config = {
        'calculate_indices': ['shannon', 'simpson', 'chao1', 'observed_species'],
        'abundance_method': 'rarefaction',
        'rarefaction_depth': 1000,
        'beta_diversity': True,
        'ordination_method': 'pcoa'
    }
    
    # Initialize analyzer
    analyzer = DiversityAnalyzer(config)
    
    # Perform analysis
    results = analyzer.analyze_community(abundance_data, metadata)
    
    # Print results summary
    print("=== Biodiversity Analysis Results ===")
    print(f"Number of samples: {results['summary']['n_samples']}")
    print(f"Number of species: {results['summary']['n_species']}")
    print(f"Total abundance: {results['summary']['total_abundance']}")
    
    print("\n=== Alpha Diversity Summary ===")
    for index, summary_stats in results['summary']['alpha_diversity_summary'].items():
        print(f"{index}: mean={summary_stats['mean']:.3f}, std={summary_stats['std']:.3f}")
    
    print("\n=== Beta Diversity ===")
    if 'ordination' in results['beta_diversity']:
        ordination = results['beta_diversity']['ordination']
        print(f"Ordination method: {ordination['method']}")
        if 'explained_variance' in ordination:
            print(f"Explained variance (PC1): {ordination['explained_variance'][0]:.2f}%")
            print(f"Explained variance (PC2): {ordination['explained_variance'][1]:.2f}%")
    
    print("\n=== Network Analysis ===")
    network = results['network_analysis']
    print(f"Network nodes: {network['n_nodes']}")
    print(f"Network edges: {network['n_edges']}")
    print(f"Network density: {network['density']:.3f}")
    
    print("\n=== Statistical Tests ===")
    for factor, factor_results in results['statistical_tests'].items():
        print(f"\nFactor: {factor}")
        for diversity_index, test_results in factor_results.items():
            kruskal_p = test_results['kruskal_wallis']['p_value']
            print(f"  {diversity_index}: Kruskal-Wallis p-value = {kruskal_p:.4f}")


if __name__ == "__main__":
    main()
