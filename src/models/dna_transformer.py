#!/usr/bin/env python3
"""
DNA Transformer Model for Deep-Sea eDNA Classification
=====================================================

This module implements transformer-based neural networks for taxonomic
classification of eukaryotic DNA sequences. The model can classify
sequences at multiple taxonomic levels simultaneously.

Author: SIH 2025 Team
Date: 2025-09-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import math
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pytorch_lightning as pl
from transformers import PreTrainedModel, PretrainedConfig


class DNATokenizer:
    """Tokenizer for DNA sequences."""
    
    def __init__(self, k: int = 4, stride: int = 1):
        """
        Initialize DNA tokenizer.
        
        Args:
            k: K-mer size for tokenization
            stride: Stride for sliding window
        """
        self.k = k
        self.stride = stride
        
        # Create vocabulary of all possible k-mers
        nucleotides = ['A', 'T', 'G', 'C']
        self.vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        
        # Generate all k-mers
        def generate_kmers(length, current=''):
            if length == 0:
                self.vocab.append(current)
                return
            for nt in nucleotides:
                generate_kmers(length - 1, current + nt)
        
        generate_kmers(k)
        
        # Create mappings
        self.vocab_to_id = {kmer: i for i, kmer in enumerate(self.vocab)}
        self.id_to_vocab = {i: kmer for i, kmer in enumerate(self.vocab)}
        
        # Special tokens
        self.pad_token_id = self.vocab_to_id['[PAD]']
        self.unk_token_id = self.vocab_to_id['[UNK]']
        self.cls_token_id = self.vocab_to_id['[CLS]']
        self.sep_token_id = self.vocab_to_id['[SEP]']
        
    def tokenize(self, sequence: str) -> List[str]:
        """Tokenize DNA sequence into k-mers."""
        if len(sequence) < self.k:
            return ['[UNK]']
        
        tokens = []
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i:i + self.k]
            if all(nt in 'ATGC' for nt in kmer):
                tokens.append(kmer)
            else:
                tokens.append('[UNK]')
        
        return tokens
    
    def encode(self, sequence: str, max_length: int = 512, 
               add_special_tokens: bool = True) -> Dict[str, List[int]]:
        """
        Encode DNA sequence to token IDs.
        
        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        tokens = self.tokenize(sequence)
        
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert to IDs
        input_ids = [self.vocab_to_id.get(token, self.unk_token_id) for token in tokens]
        
        # Pad or truncate
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        else:
            input_ids.extend([self.pad_token_id] * (max_length - len(input_ids)))
        
        # Create attention mask
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in input_ids]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to sequence."""
        tokens = [self.id_to_vocab.get(token_id, '[UNK]') for token_id in token_ids]
        # Remove special tokens and concatenate k-mers
        filtered_tokens = [token for token in tokens if token not in ['[PAD]', '[CLS]', '[SEP]', '[UNK]']]
        
        if not filtered_tokens:
            return ""
        
        # Reconstruct sequence from overlapping k-mers
        sequence = filtered_tokens[0]
        for token in filtered_tokens[1:]:
            # Overlap by k-1 characters
            if sequence.endswith(token[:-1]):
                sequence += token[-1]
            else:
                sequence += token  # No overlap found, just concatenate
        
        return sequence


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class DNATransformerConfig(PretrainedConfig):
    """Configuration for DNA Transformer model."""
    
    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 512,
        max_position_embeddings: int = 512,
        num_labels: int = 2,
        dropout: float = 0.1,
        kmer_size: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_labels = num_labels
        self.dropout = dropout
        self.kmer_size = kmer_size


class DNATransformer(PreTrainedModel):
    """
    Transformer model for DNA sequence classification.
    """
    
    config_class = DNATransformerConfig
    
    def __init__(self, config: DNATransformerConfig):
        super().__init__(config)
        
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = PositionalEncoding(config.hidden_size, 
                                                    config.max_position_embeddings)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            labels: Labels for training of shape [batch_size, num_labels]
            
        Returns:
            Dictionary with logits and loss (if labels provided)
        """
        # Embeddings
        embeddings = self.embeddings(input_ids)
        embeddings = self.position_embeddings(embeddings.transpose(0, 1)).transpose(0, 1)
        embeddings = self.dropout(embeddings)
        
        # Create padding mask for transformer
        if attention_mask is not None:
            # Convert attention mask to transformer padding mask
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None
        
        # Transformer
        transformer_output = self.transformer(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Global average pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(transformer_output.size()).float()
            sum_embeddings = torch.sum(transformer_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = transformer_output.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        outputs = {'logits': logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            if self.config.num_labels == 1:
                # Regression
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
            
            outputs['loss'] = loss
        
        return outputs


class eDNADataset(Dataset):
    """Dataset for eDNA sequences."""
    
    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer: DNATokenizer,
        max_length: int = 512
    ):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize sequence
        encoded = self.tokenizer.encode(sequence, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DNAClassifier(pl.LightningModule):
    """PyTorch Lightning wrapper for DNA Transformer."""
    
    def __init__(
        self,
        config: DNATransformerConfig,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = DNATransformer(config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Metrics storage
        self.validation_predictions = []
        self.validation_labels = []
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask, labels)
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs['loss']
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs['loss']
        logits = outputs['logits']
        
        predictions = torch.argmax(logits, dim=-1)
        
        # Store predictions and labels for epoch-end metrics
        self.validation_predictions.extend(predictions.cpu().numpy())
        self.validation_labels.extend(batch['labels'].cpu().numpy())
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        if self.validation_predictions:
            # Calculate accuracy
            accuracy = accuracy_score(self.validation_labels, self.validation_predictions)
            self.log('val_accuracy', accuracy, prog_bar=True)
            
            # Clear stored predictions
            self.validation_predictions = []
            self.validation_labels = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class DNAClassificationTrainer:
    """Trainer class for DNA classification models."""
    
    def __init__(self, config: Dict):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize tokenizer
        self.tokenizer = DNATokenizer(
            k=config['transformer']['kmer_size'] if 'kmer_size' in config['transformer'] else 4
        )
        
        # Model configuration
        self.model_config = DNATransformerConfig(
            vocab_size=len(self.tokenizer.vocab),
            hidden_size=config['transformer']['embedding_dim'],
            num_hidden_layers=config['transformer']['num_layers'],
            num_attention_heads=config['transformer']['num_heads'],
            intermediate_size=config['transformer']['hidden_dim'],
            max_position_embeddings=config['transformer']['max_length'],
            dropout=config['transformer']['dropout']
        )
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the trainer."""
        logger = logging.getLogger('DNAClassificationTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def prepare_data(
        self,
        sequences: List[str],
        labels: List[str],
        test_size: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, LabelEncoder]:
        """
        Prepare data for training.
        
        Args:
            sequences: List of DNA sequences
            labels: List of taxonomic labels
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (train_loader, val_loader, label_encoder)
        """
        self.logger.info(f"Preparing data for {len(sequences)} sequences")
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Update model config with number of labels
        self.model_config.num_labels = len(label_encoder.classes_)
        
        # Split data
        train_sequences, val_sequences, train_labels, val_labels = train_test_split(
            sequences, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
        )
        
        # Create datasets
        train_dataset = eDNADataset(
            train_sequences, train_labels, self.tokenizer, 
            max_length=self.config['transformer']['max_length']
        )
        val_dataset = eDNADataset(
            val_sequences, val_labels, self.tokenizer,
            max_length=self.config['transformer']['max_length']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        self.logger.info(f"Training set: {len(train_dataset)} samples")
        self.logger.info(f"Validation set: {len(val_dataset)} samples")
        self.logger.info(f"Number of classes: {len(label_encoder.classes_)}")
        
        return train_loader, val_loader, label_encoder
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str = "models/"
    ) -> DNAClassifier:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Directory to save the model
            
        Returns:
            Trained model
        """
        self.logger.info("Starting model training")
        
        # Initialize model
        model = DNAClassifier(
            config=self.model_config,
            learning_rate=self.config['training']['learning_rate']
        )
        
        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=self.config['training']['num_epochs'],
            accelerator='auto',
            devices=1,
            log_every_n_steps=10,
            enable_checkpointing=True,
            default_root_dir=output_dir,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['training']['early_stopping_patience'],
                    mode='min'
                ),
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                    filename='best-{epoch}-{val_loss:.2f}'
                )
            ]
        )
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        self.logger.info("Training completed")
        
        return model
    
    def predict(
        self,
        model: DNAClassifier,
        sequences: List[str],
        label_encoder: LabelEncoder
    ) -> List[str]:
        """
        Make predictions on new sequences.
        
        Args:
            model: Trained model
            sequences: List of DNA sequences
            label_encoder: Label encoder from training
            
        Returns:
            List of predicted taxonomic labels
        """
        model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for sequence in sequences:
                # Tokenize sequence
                encoded = self.tokenizer.encode(sequence, 
                                              max_length=self.config['transformer']['max_length'])
                
                # Convert to tensors
                input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long)
                attention_mask = torch.tensor([encoded['attention_mask']], dtype=torch.long)
                
                # Get prediction
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits']
                predicted_class = torch.argmax(logits, dim=-1).item()
                
                # Decode label
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                predictions.append(predicted_label)
        
        return predictions


def main():
    """Main function for testing the transformer model."""
    # Test configuration
    config = {
        'transformer': {
            'embedding_dim': 128,
            'num_heads': 4,
            'num_layers': 3,
            'hidden_dim': 256,
            'max_length': 256,
            'dropout': 0.1,
            'kmer_size': 4
        },
        'training': {
            'batch_size': 8,
            'learning_rate': 0.001,
            'num_epochs': 5,
            'early_stopping_patience': 3
        }
    }
    
    # Test data
    sequences = [
        "ATGCGTACGATCGTAGCATGCAAATTTGGGCCC",
        "CGATCGATCGATCGATCGATAAATTTGGGCCC",
        "AAATTTGGGCCCAAAATGCGTACGATCGTAGC",
        "TGCATGCATGCATGCATGCACGATCGATCGAT"
    ]
    labels = ["Class_A", "Class_B", "Class_A", "Class_B"]
    
    # Initialize trainer
    trainer = DNAClassificationTrainer(config)
    
    # Prepare data
    train_loader, val_loader, label_encoder = trainer.prepare_data(sequences, labels, test_size=0.5)
    
    # Train model
    model = trainer.train(train_loader, val_loader)
    
    # Test predictions
    test_sequences = ["ATGCGTACGATCGTAGCATGC", "CGATCGATCGATCGATCGAT"]
    predictions = trainer.predict(model, test_sequences, label_encoder)
    
    print(f"Test sequences: {test_sequences}")
    print(f"Predictions: {predictions}")


if __name__ == "__main__":
    main()
