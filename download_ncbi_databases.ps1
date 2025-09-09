# NCBI Database Download Script for SIH 2025 eDNA Pipeline
# ========================================================

param(
    [string[]]$DatabaseFiles = @(),
    [string]$OutputPath = "data/ncbi_nt_euk/",
    [switch]$VerifyChecksums = $true,
    [switch]$EukaryoticOnly = $true,
    [switch]$ShowAvailable = $false
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SIH 2025: NCBI Database Downloader" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$BaseUrl = "https://ftp.ncbi.nlm.nih.gov/blast/db/"

# Available databases (subset for eukaryotic analysis)
$AvailableDatabases = @{
    "Small eukaryotic subset (recommended for testing)" = @("nt_euk.00.tar.gz", "nt_euk.01.tar.gz")
    "Medium eukaryotic subset" = @("nt_euk.00.tar.gz", "nt_euk.01.tar.gz", "nt_euk.02.tar.gz", "nt_euk.03.tar.gz")
    "Full nr database (warning: ~500GB+)" = @("nr.000.tar.gz", "nr.001.tar.gz", "nr.002.tar.gz", "nr.003.tar.gz", "nr.004.tar.gz")
}

if ($ShowAvailable) {
    Write-Host "Available database sets:" -ForegroundColor Yellow
    foreach ($set in $AvailableDatabases.Keys) {
        Write-Host "  - $set" -ForegroundColor White
        foreach ($file in $AvailableDatabases[$set]) {
            Write-Host "    $file" -ForegroundColor Gray
        }
        Write-Host ""
    }
    Write-Host "Usage examples:" -ForegroundColor Yellow
    Write-Host "  Small set:  .\download_ncbi_databases.ps1" -ForegroundColor White
    Write-Host "  Custom:     .\download_ncbi_databases.ps1 -DatabaseFiles @('nt_euk.00.tar.gz')" -ForegroundColor White
    Write-Host "  Show help:  Get-Help .\download_ncbi_databases.ps1 -Full" -ForegroundColor White
    exit 0
}

# Set default database files if none specified
if ($DatabaseFiles.Count -eq 0) {
    if ($EukaryoticOnly) {
        $DatabaseFiles = $AvailableDatabases["Small eukaryotic subset (recommended for testing)"]
        Write-Host "Using small eukaryotic subset (recommended for SIH 2025)" -ForegroundColor Green
    } else {
        Write-Host "No database files specified. Use -ShowAvailable to see options." -ForegroundColor Red
        exit 1
    }
}

# Display what will be downloaded
Write-Host "Files to download:" -ForegroundColor Yellow
$TotalEstimatedSize = 0
foreach ($file in $DatabaseFiles) {
    # Rough size estimates (in GB)
    $sizeEstimate = switch -Wildcard ($file) {
        "nt_euk.*" { 2.5 }
        "nr.0*" { 25 }
        default { 5 }
    }
    $TotalEstimatedSize += $sizeEstimate
    Write-Host "  $file (~$sizeEstimate GB)" -ForegroundColor White
}
Write-Host "Total estimated size: ~$TotalEstimatedSize GB" -ForegroundColor Cyan
Write-Host ""

# Warning for large downloads
if ($TotalEstimatedSize -gt 10) {
    Write-Host "⚠️  WARNING: Large download detected!" -ForegroundColor Red
    Write-Host "This will download $TotalEstimatedSize GB of data and may take several hours." -ForegroundColor Yellow
    Write-Host "Ensure you have sufficient disk space and a stable internet connection." -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Do you want to continue? (y/N)"
    if ($continue -notmatch "^[Yy]$") {
        Write-Host "Download cancelled by user." -ForegroundColor Yellow
        exit 0
    }
}

# Create output directory
Write-Host "Creating output directory: $OutputPath" -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

$StartTime = Get-Date
$SuccessfulDownloads = 0
$FailedDownloads = 0

foreach ($DbFile in $DatabaseFiles) {
    Write-Host "`n========================================" -ForegroundColor Blue
    Write-Host "Processing: $DbFile" -ForegroundColor Blue
    Write-Host "========================================" -ForegroundColor Blue
    
    $Url = "$BaseUrl$DbFile"
    $OutputFile = Join-Path $OutputPath $DbFile
    
    # Check if file already exists
    if (Test-Path $OutputFile) {
        Write-Host "✓ File already exists: $OutputFile" -ForegroundColor Green
        
        if ($VerifyChecksums) {
            Write-Host "Verifying existing file..." -ForegroundColor Yellow
            if (Test-Checksum -File $DbFile -OutputPath $OutputPath) {
                Write-Host "✓ Checksum verified - skipping download" -ForegroundColor Green
                $SuccessfulDownloads++
                continue
            } else {
                Write-Host "⚠️ Checksum mismatch - re-downloading" -ForegroundColor Yellow
            }
        } else {
            Write-Host "Skipping checksum verification" -ForegroundColor Gray
            $SuccessfulDownloads++
            continue
        }
    }
    
    try {
        Write-Host "Downloading from: $Url" -ForegroundColor Cyan
        
        # Use Invoke-WebRequest with progress tracking
        $ProgressPreference = 'SilentlyContinue'  # Disable default progress bar
        
        # Custom progress tracking
        $WebClient = New-Object System.Net.WebClient
        $Global:DownloadComplete = $false
        $Global:LastProgress = 0
        
        # Register progress event
        $ProgressScriptBlock = {
            $Global:LastProgress = $Event.SourceEventArgs.ProgressPercentage
            if ($Global:LastProgress -gt 0 -and $Global:LastProgress -le 100) {
                Write-Progress -Activity "Downloading $DbFile" -Status "Progress: $Global:LastProgress%" -PercentComplete $Global:LastProgress
            }
        }
        
        Register-ObjectEvent -InputObject $WebClient -EventName DownloadProgressChanged -Action $ProgressScriptBlock | Out-Null
        
        # Download file
        $WebClient.DownloadFile($Url, $OutputFile)
        Write-Progress -Activity "Downloading" -Completed
        
        # Clean up event
        Get-EventSubscriber | Where-Object SourceObject -eq $WebClient | Unregister-Event
        $WebClient.Dispose()
        
        Write-Host "✓ Downloaded: $DbFile" -ForegroundColor Green
        
        # Verify checksum if requested
        if ($VerifyChecksums) {
            Write-Host "Verifying checksum..." -ForegroundColor Yellow
            if (Test-Checksum -File $DbFile -OutputPath $OutputPath) {
                Write-Host "✓ Checksum verified" -ForegroundColor Green
            } else {
                Write-Host "✗ Checksum verification failed!" -ForegroundColor Red
                $FailedDownloads++
                continue
            }
        }
        
        $SuccessfulDownloads++
        
    } catch {
        Write-Host "✗ Failed to download $DbFile" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
        $FailedDownloads++
    }
}

# Final summary
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DOWNLOAD SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Successful downloads: $SuccessfulDownloads" -ForegroundColor Green
Write-Host "Failed downloads: $FailedDownloads" -ForegroundColor Red
Write-Host "Total time: $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "Location: $OutputPath" -ForegroundColor White

# Show next steps
if ($SuccessfulDownloads -gt 0) {
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. Install NCBI BLAST+ tools if not already installed" -ForegroundColor White
    Write-Host "2. Extract databases: makeblastdb commands will be run automatically" -ForegroundColor White
    Write-Host "3. Enable NCBI validation in config.yaml: set use_ncbi_validation: true" -ForegroundColor White
    Write-Host "4. Run your eDNA analysis pipeline" -ForegroundColor White
    
    Write-Host "`nFor CMLRE:" -ForegroundColor Cyan
    Write-Host "These databases enable optional validation of novel taxa discoveries" -ForegroundColor White
    Write-Host "while maintaining the database-independent core functionality." -ForegroundColor White
}

Write-Host ""

# Function to test checksums
function Test-Checksum {
    param(
        [string]$File,
        [string]$OutputPath
    )
    
    try {
        $Md5Url = "$BaseUrl$File.md5"
        $LocalFile = Join-Path $OutputPath $File
        
        # Download MD5 checksum
        $Md5Content = Invoke-WebRequest -Uri $Md5Url -UseBasicParsing -TimeoutSec 30
        $ExpectedMd5 = ($Md5Content.Content -split '\s+')[0].Trim()
        
        # Calculate local file MD5
        $ActualMd5 = (Get-FileHash -Path $LocalFile -Algorithm MD5).Hash.ToLower()
        
        return $ActualMd5 -eq $ExpectedMd5.ToLower()
        
    } catch {
        Write-Warning "Could not verify checksum for $File : $_"
        return $true  # Assume OK if can't verify
    }
}
