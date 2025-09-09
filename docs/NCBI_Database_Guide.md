# NCBI Database Integration Guide
## SIH 2025 Deep-Sea eDNA Analysis Pipeline

### Overview

The SIH 2025 eDNA pipeline is designed to work **database-independently** using AI and machine learning approaches. However, it also provides optional NCBI database integration for validation of novel taxa discoveries.

### 🔧 **Core Philosophy: Database-Independent Analysis**

- **Primary Analysis**: Uses unsupervised ML (clustering, UMAP) to discover novel taxa WITHOUT relying on reference databases
- **Optional Validation**: NCBI databases can be used to validate discoveries, not as the primary classification method
- **Novel Taxa Focus**: Optimized for discovering species not present in current databases

---

## NCBI Database Setup

### Option 1: Quick Setup (Recommended for Testing)

```powershell
# Download small eukaryotic subset (~5GB)
.\download_ncbi_databases.ps1
```

### Option 2: Custom Database Selection

```powershell
# Show available options
.\download_ncbi_databases.ps1 -ShowAvailable

# Download specific files
.\download_ncbi_databases.ps1 -DatabaseFiles @("nt_euk.00.tar.gz", "nt_euk.01.tar.gz")
```

### Option 3: Manual Download (If Script Fails)

1. Visit: https://ftp.ncbi.nlm.nih.gov/blast/db/
2. Download eukaryotic files (nt_euk.*.tar.gz) 
3. Place in `data/ncbi_nt_euk/` directory

---

## Database Files Reference

From your provided NCBI FTP listing, recommended files for eukaryotic analysis:

### For Deep-Sea eDNA (Recommended)
- `nt_euk.00.tar.gz` - Eukaryotic nucleotide sequences (subset 1)
- `nt_euk.01.tar.gz` - Eukaryotic nucleotide sequences (subset 2)
- `nt_euk.02.tar.gz` - Eukaryotic nucleotide sequences (subset 3)

### Full Protein Database (Large - 500GB+)
- `nr.000.tar.gz` through `nr.027.tar.gz` - Complete non-redundant protein database
- **Warning**: These are 2-5GB each, totaling hundreds of GB

---

## Configuration

### Enable NCBI Validation in `config/config.yaml`:

```yaml
database:
  # Enable optional NCBI validation
  use_ncbi_validation: true
  ncbi_db_path: "data/ncbi_nt_euk/"
  blast_evalue: 0.001
  blast_max_targets: 10
  
  # Validation thresholds
  validation_threshold: 0.8
  min_identity: 0.7
```

---

## Pipeline Integration

### How NCBI Validation Works:

1. **Primary Analysis**: Pipeline discovers taxa using unsupervised ML
2. **Novel Taxa Identification**: Clustering identifies potential new species
3. **Optional Validation**: BLAST searches validate discoveries against NCBI
4. **Classification**:
   - **Novel Taxa**: Sequences with <70% identity to known species
   - **Validated Taxa**: Sequences with >70% identity to database entries

### Usage in Pipeline:

```python
# The pipeline automatically uses NCBI validation if configured
python -m src.pipeline \
    --config config/config.yaml \
    --input data/raw_sequences \
    --output results

# Results will include both:
# - Database-independent novel taxa discoveries  
# - Optional NCBI validation results
```

---

## Storage Requirements

### Recommended Setup (Eukaryotic Focus):
- **nt_euk.00-02.tar.gz**: ~7.5 GB compressed, ~15 GB extracted
- **Total Space Needed**: ~25 GB (including intermediate files)

### Full Database Setup:
- **nr.000-027.tar.gz**: ~500+ GB compressed, ~1+ TB extracted
- **Only recommended for comprehensive validation**

---

## For CMLRE Deployment

### Production Recommendations:

1. **Start Small**: Use eukaryotic subset for initial deployment
2. **Database-Independent Priority**: Rely primarily on ML-based discovery
3. **Validation Optional**: Use NCBI only for validating interesting discoveries
4. **Storage Planning**: Allocate 50-100 GB for database storage

### Docker Deployment with NCBI:

```bash
# Download databases first
powershell -ExecutionPolicy Bypass -File download_ncbi_databases.ps1

# Enable NCBI validation in config
# Set use_ncbi_validation: true in config/config.yaml

# Deploy with database volume
docker-compose up -d
```

---

## Troubleshooting

### Common Issues:

**1. Download Failures**
```powershell
# Retry with checksum verification disabled
.\download_ncbi_databases.ps1 -VerifyChecksums:$false
```

**2. BLAST Not Found**
- Install NCBI BLAST+ tools: https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download
- Add BLAST to system PATH

**3. Disk Space Issues**
- Use smaller database subset
- Monitor available space during download

**4. Network Timeouts**
- NCBI servers can be slow
- Consider downloading during off-peak hours
- Use resume capability if available

---

## Key Benefits of This Approach

### ✅ **Database-Independent Core**
- Discovers novel taxa even if not in any database
- No reliance on incomplete reference databases
- Suitable for deep-sea ecosystems with unknown biodiversity

### ✅ **Optional Validation**
- Validates discoveries against known sequences
- Distinguishes truly novel taxa from known species
- Provides confidence scores for classifications

### ✅ **Scalable Implementation**  
- Works with or without NCBI databases
- Configurable validation thresholds
- Suitable for CMLRE operational deployment

---

## Performance Notes

- **Database-Independent Analysis**: Fast, requires minimal storage
- **With NCBI Validation**: Slower due to BLAST searches, requires significant storage
- **Recommendation**: Start database-independent, add NCBI validation as needed

This approach aligns perfectly with the SIH 2025 problem statement: providing AI-driven biodiversity assessment that minimizes database dependency while enabling optional validation for critical discoveries.
