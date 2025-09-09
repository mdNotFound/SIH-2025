# SIH 2025 Deep-Sea eDNA Analysis Pipeline Installation Script
# =============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SIH 2025 Deep-Sea eDNA Analysis Pipeline" -ForegroundColor Cyan  
Write-Host "Installation and Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Check Python version
$version = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([Version]$version -lt [Version]"3.8") {
    Write-Host "✗ Python 3.8+ required. Found version: $version" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python version $version is compatible" -ForegroundColor Green

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Removing..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}

python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Virtual environment created" -ForegroundColor Green

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "`nInstalling Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green

# Create necessary directories
Write-Host "`nCreating directory structure..." -ForegroundColor Yellow
$directories = @("data", "data/raw_sequences", "results", "logs", "models/pretrained")
foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "✓ Created: $dir" -ForegroundColor Green
}

# Set up environment variables
Write-Host "`nSetting up environment..." -ForegroundColor Yellow
$envFile = @"
# SIH 2025 Deep-Sea eDNA Pipeline Environment Variables
PYTHONPATH=src
OMP_NUM_THREADS=8
NUMBA_NUM_THREADS=8
"@

$envFile | Out-File -FilePath ".env" -Encoding UTF8
Write-Host "✓ Environment file created" -ForegroundColor Green

# Test installation
Write-Host "`nTesting installation..." -ForegroundColor Yellow
try {
    python -c "import sys; sys.path.append('src'); from pipeline import eDNAAnalysisPipeline; print('✓ Pipeline imports successfully')"
    Write-Host "✓ Installation test passed" -ForegroundColor Green
} catch {
    Write-Host "✗ Installation test failed" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

# Success message
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "INSTALLATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Place your FASTQ files in: data/raw_sequences/" -ForegroundColor White  
Write-Host "3. Run analysis: python -m src.pipeline --config config/config.yaml --input data/raw_sequences --output results" -ForegroundColor White
Write-Host ""
Write-Host "For CMLRE deployment, see Docker instructions in README.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "Need help? Check the documentation or run: python -m src.pipeline --help" -ForegroundColor Gray
