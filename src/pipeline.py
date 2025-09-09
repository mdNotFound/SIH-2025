#!/usr/bin/env python3
"""
Main eDNA Analysis Pipeline for Deep-Sea Biodiversity Assessment
===============================================================

This module integrates all components of the SIH 2025 deep-sea eDNA
analysis pipeline, providing a unified interface for processing
eDNA sequences from raw data to biodiversity insights.

Author: SIH 2025 Team
Date: 2025-09-09
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse
import time

# Import pipeline components
from .preprocessing import SequenceProcessor
from .clustering import SequenceEncoder, NovelTaxaDiscoverer
from .models import DNAClassificationTrainer
from .biodiversity import DiversityAnalyzer


class eDNAAnalysisPipeline:
    """
    Main pipeline class for deep-sea eDNA analysis.
    
    This class orchestrates the entire analysis workflow:
    1. Preprocessing of raw eDNA sequences
    2. Novel taxa discovery using unsupervised learning
    3. Taxonomic classification using deep learning
    4. Biodiversity assessment and community analysis
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the eDNA analysis pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.sequence_processor = None
        self.taxa_discoverer = None
        self.classifier_trainer = None
        self.diversity_analyzer = None
        
        # Results storage
        self.results = {}
        
        self.logger.info("eDNA Analysis Pipeline initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the pipeline."""
        logger = logging.getLogger('eDNAAnalysisPipeline')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # Clear existing handlers
        logger.handlers = []
        
        # Console handler
        if self.config['logging']['console']:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(self.config['logging']['format'])
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if 'file' in self.config['logging']:
            log_dir = Path(self.config['logging']['file']).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self.config['logging']['file'])
            file_formatter = logging.Formatter(self.config['logging']['format'])
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def analyze_samples(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Run complete eDNA analysis pipeline.
        
        Args:
            input_path: Path to input directory containing raw FASTQ files
            output_path: Path to output directory for results
            
        Returns:
            Dictionary containing all analysis results
        """
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("STARTING DEEP-SEA eDNA ANALYSIS PIPELINE")
        self.logger.info("=" * 60)
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Preprocessing
            self.logger.info("STEP 1: PREPROCESSING RAW eDNA SEQUENCES")
            preprocessing_results = self._run_preprocessing(input_path, output_dir / "preprocessing")
            self.results['preprocessing'] = preprocessing_results
            
            # Step 2: Novel Taxa Discovery
            self.logger.info("STEP 2: NOVEL TAXA DISCOVERY")
            discovery_results = self._run_taxa_discovery(
                output_dir / "preprocessing", 
                output_dir / "discovery"
            )
            self.results['taxa_discovery'] = discovery_results
            
            # Step 3: Classification (optional, if reference data available)
            if self.config.get('database', {}).get('use_ncbi_validation', False):
                self.logger.info("STEP 3: TAXONOMIC CLASSIFICATION")
                classification_results = self._run_classification(
                    output_dir / "preprocessing",
                    output_dir / "classification"
                )
                self.results['classification'] = classification_results
            
            # Step 4: Biodiversity Analysis
            self.logger.info("STEP 4: BIODIVERSITY ASSESSMENT")
            biodiversity_results = self._run_biodiversity_analysis(
                discovery_results,
                output_dir / "biodiversity"
            )
            self.results['biodiversity'] = biodiversity_results
            
            # Step 5: Generate Reports
            self.logger.info("STEP 5: GENERATING REPORTS")
            self._generate_final_report(output_dir / "reports")
            
            # Calculate total runtime
            total_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            self.results['pipeline_info'] = {
                'total_runtime_seconds': total_time,
                'config_used': self.config_path,
                'output_directory': str(output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            self.results['error'] = str(e)
            raise
        
        return self.results
    
    def _run_preprocessing(self, input_path: str, output_path: Path) -> Dict[str, Any]:
        """Run sequence preprocessing step."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sequence processor
        self.sequence_processor = SequenceProcessor(self.config['preprocessing'])
        
        # Process FASTQ files
        processing_stats = self.sequence_processor.process_fastq_files(
            input_path, str(output_path)
        )
        
        # Extract marker sequences if configured
        marker_results = {}
        if self.config['preprocessing'].get('extract_markers', False):
            for marker in self.config['preprocessing']['supported_markers']:
                self.logger.info(f"Extracting {marker} marker sequences")
                marker_stats = self.sequence_processor.extract_marker_sequences(
                    str(output_path), marker, str(output_path)
                )
                marker_results[marker] = marker_stats
        
        # Generate sequence features
        if self.config['preprocessing'].get('generate_features', False):
            self.logger.info("Generating sequence features")
            feature_file = self.sequence_processor.generate_sequence_features(
                str(output_path), str(output_path)
            )
        
        return {
            'processing_stats': processing_stats,
            'marker_extraction': marker_results,
            'feature_file': feature_file if 'feature_file' in locals() else None
        }
    
    def _run_taxa_discovery(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Run novel taxa discovery step."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load processed sequences
        sequences, sequence_ids = self._load_processed_sequences(input_path)
        
        if not sequences:
            self.logger.warning("No processed sequences found for taxa discovery")
            return {'error': 'No sequences available'}
        
        # Initialize taxa discoverer
        self.taxa_discoverer = NovelTaxaDiscoverer(self.config['unsupervised'])
        
        # Discover taxa
        discovery_results = self.taxa_discoverer.discover_taxa(sequences, sequence_ids)
        
        # Save results
        self.taxa_discoverer.save_model(str(output_path / "taxa_discovery_model.joblib"))
        
        # Create visualizations
        if len(sequences) > 1:
            self.taxa_discoverer.visualize_clusters(str(output_path / "visualizations"))
        
        return discovery_results
    
    def _run_classification(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Run taxonomic classification step (optional)."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # This would integrate with NCBI database validation
        # For now, return placeholder results
        self.logger.info("Classification step not fully implemented - placeholder results")
        
        return {
            'status': 'not_implemented',
            'message': 'Optional taxonomic classification with NCBI validation'
        }
    
    def _run_biodiversity_analysis(self, discovery_results: Dict, output_path: Path) -> Dict[str, Any]:
        """Run biodiversity assessment step."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert discovery results to abundance matrix
        abundance_data = self._create_abundance_matrix(discovery_results)
        
        if abundance_data is None or abundance_data.empty:
            self.logger.warning("No abundance data available for biodiversity analysis")
            return {'error': 'No abundance data available'}
        
        # Initialize diversity analyzer
        self.diversity_analyzer = DiversityAnalyzer(self.config['biodiversity'])
        
        # Perform biodiversity analysis
        biodiversity_results = self.diversity_analyzer.analyze_community(abundance_data)
        
        # Save results
        abundance_data.to_csv(output_path / "abundance_matrix.csv")
        
        # Save diversity results to CSV
        if 'alpha_diversity' in biodiversity_results:
            alpha_df = pd.DataFrame(biodiversity_results['alpha_diversity'])
            alpha_df.to_csv(output_path / "alpha_diversity.csv")
        
        return biodiversity_results
    
    def _load_processed_sequences(self, input_path: Path) -> tuple[List[str], List[str]]:
        """Load processed sequences from preprocessing output."""
        sequences = []
        sequence_ids = []
        
        # Look for processed FASTA files
        fasta_files = list(input_path.glob("*_processed.fasta"))
        
        if not fasta_files:
            self.logger.warning(f"No processed FASTA files found in {input_path}")
            return sequences, sequence_ids
        
        from Bio import SeqIO
        
        for fasta_file in fasta_files:
            self.logger.info(f"Loading sequences from {fasta_file}")
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequences.append(str(record.seq))
                sequence_ids.append(record.id)
        
        self.logger.info(f"Loaded {len(sequences)} sequences total")
        return sequences, sequence_ids
    
    def _create_abundance_matrix(self, discovery_results: Dict) -> Optional[pd.DataFrame]:
        """Create abundance matrix from discovery results."""
        if 'cluster_representatives' not in discovery_results:
            return None
        
        # Create a simple abundance matrix based on cluster sizes
        cluster_data = discovery_results['cluster_representatives']
        
        if not cluster_data:
            return None
        
        # Create abundance data (using cluster sizes as abundances)
        abundance_dict = {}
        for cluster_id, cluster_info in cluster_data.items():
            abundance_dict[f'Cluster_{cluster_id}'] = cluster_info['cluster_size']
        
        # Create DataFrame with single sample (could be extended for multiple samples)
        abundance_df = pd.DataFrame([abundance_dict], index=['Sample_1'])
        
        return abundance_df
    
    def _generate_final_report(self, output_path: Path):
        """Generate final analysis report."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML report if configured
        if self.config['output'].get('generate_html_report', False):
            self._generate_html_report(output_path / "analysis_report.html")
        
        # Save results summary
        self._save_results_summary(output_path / "results_summary.json")
        
        self.logger.info(f"Reports generated in {output_path}")
    
    def _generate_html_report(self, output_file: Path):
        """Generate HTML report (placeholder)."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deep-Sea eDNA Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SIH 2025: Deep-Sea eDNA Biodiversity Analysis</h1>
                <p>Comprehensive analysis of environmental DNA from deep-sea ecosystems</p>
            </div>
            
            <div class="section">
                <h2>Pipeline Overview</h2>
                <p>This report presents the results of AI-driven analysis of deep-sea eDNA sequences
                   using novel unsupervised learning approaches for taxa discovery.</p>
            </div>
            
            <div class="section">
                <h2>Key Results</h2>
                <!-- Results would be inserted here -->
                <p>Analysis completed successfully. Detailed results are available in the output files.</p>
            </div>
            
            <div class="section">
                <h2>For CMLRE</h2>
                <p>This analysis pipeline provides database-independent biodiversity assessment
                   suitable for deep-sea ecosystem monitoring and conservation efforts.</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _save_results_summary(self, output_file: Path):
        """Save results summary to JSON file."""
        import json
        
        # Create serializable summary
        summary = {}
        for key, value in self.results.items():
            try:
                # Convert numpy arrays and other non-serializable objects
                if isinstance(value, dict):
                    summary[key] = self._make_serializable(value)
                else:
                    summary[key] = str(value)
            except:
                summary[key] = f"<{type(value).__name__} object>"
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="SIH 2025 Deep-Sea eDNA Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.pipeline --config config/config.yaml --input data/raw_sequences --output results/

For CMLRE Usage:
    This pipeline provides database-independent analysis of deep-sea eDNA sequences
    for biodiversity assessment and novel taxa discovery.
        """
    )
    
    parser.add_argument(
        '--config', 
        required=True, 
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--input', 
        required=True, 
        help='Path to input directory containing raw FASTQ files'
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help='Path to output directory for results'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = eDNAAnalysisPipeline(args.config)
    
    # Set verbose logging if requested
    if args.verbose:
        pipeline.logger.setLevel(logging.DEBUG)
    
    # Run analysis
    try:
        results = pipeline.analyze_samples(args.input, args.output)
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {args.output}")
        
        # Print summary
        if 'biodiversity' in results and 'summary' in results['biodiversity']:
            summary = results['biodiversity']['summary']
            print(f"\nSummary:")
            print(f"  Samples analyzed: {summary.get('n_samples', 'N/A')}")
            print(f"  Species detected: {summary.get('n_species', 'N/A')}")
            print(f"  Total abundance: {summary.get('total_abundance', 'N/A')}")
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
