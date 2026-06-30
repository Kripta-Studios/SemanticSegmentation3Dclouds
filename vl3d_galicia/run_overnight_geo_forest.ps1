param(
    [string]$Python = "python",
    [int]$Workers = 12,
    [int]$PrepareWorkers = 16,
    [int]$PointsPerBlock = 8192,
    [int]$BatchSize = 32,
    [int]$JepaBatchSize = 64,
    [int]$BaselineEpochs = 60,
    [int]$JepaEpochs = 120,
    [int]$FinetuneEpochs = 60,
    [int]$EarlyStoppingPatience = 10,
    [double]$EarlyStoppingMinDelta = 0.001,
    [int]$PretrainEarlyStoppingPatience = 15,
    [double]$PretrainEarlyStoppingMinDelta = 0.0001,
    [int]$Seed = 42,
    [double]$GridSize = 2.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:OMP_NUM_THREADS = "16"
$env:MKL_NUM_THREADS = "16"
$env:OPENBLAS_NUM_THREADS = "16"
$env:CUDA_MODULE_LOADING = "LAZY"

$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "outputs/overnight/logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$TranscriptPath = Join-Path $LogDir "overnight_$RunStamp.log"
Start-Transcript -Path $TranscriptPath -Append | Out-Null

function Invoke-Python {
    param(
        [string]$StepName,
        [string[]]$CommandArgs
    )
    Write-Host ""
    Write-Host "===== $StepName ====="
    Write-Host ("> " + $Python + " " + ($CommandArgs -join " "))
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Pipeline {
    param(
        [string]$StepName,
        [int]$TrainBatchSize,
        [int]$PretrainBatchSize
    )
    Write-Host ""
    Write-Host "===== $StepName ====="
    & powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 `
        -Python $Python `
        -Raw data/raw/pnoa_galicia `
        -Blocks data/processed/galicia_blocks_full `
        -BlocksTW data/processed/galicia_blocks_full_tw `
        -Reports reports `
        -GeoExperimentsOut outputs/full `
        -JepaPretrainOut outputs/full/tw_jepa_pretrain `
        -Workers $Workers `
        -PrepareWorkers $PrepareWorkers `
        -PointsPerBlock $PointsPerBlock `
        -BatchSize $TrainBatchSize `
        -JepaBatchSize $PretrainBatchSize `
        -BaselineEpochs $BaselineEpochs `
        -JepaEpochs $JepaEpochs `
        -FinetuneEpochs $FinetuneEpochs `
        -Seed $Seed `
        -ClassWeightMode inverse_sqrt `
        -MaxClassWeight 20.0 `
        -EarlyStoppingPatience $EarlyStoppingPatience `
        -EarlyStoppingMinDelta $EarlyStoppingMinDelta `
        -PretrainEarlyStoppingPatience $PretrainEarlyStoppingPatience `
        -PretrainEarlyStoppingMinDelta $PretrainEarlyStoppingMinDelta
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Get-BestJepaRun {
    param([string]$ComparisonCsv)
    $rows = Import-Csv $ComparisonCsv | Where-Object { $_.model_name -like "jepa_*" }
    if (-not $rows) {
        throw "No JEPA rows found in $ComparisonCsv"
    }
    return $rows | Sort-Object { [double]$_.mIoU } -Descending | Select-Object -First 1
}

function Export-Analysis {
    param(
        [string]$Name,
        [string]$Data,
        [string]$OutRoot,
        [string]$ForestOut,
        [int]$MapTiles = 8,
        [int]$LazTiles = 4
    )
    $ComparisonCsv = Join-Path $OutRoot "comparison/test_comparison.csv"
    if (-not (Test-Path $ComparisonCsv)) {
        Invoke-Python "$Name comparison" @(
            "scripts/07_compare_results.py",
            "--experiments-root", $OutRoot,
            "--out-csv", (Join-Path $OutRoot "comparison/test_comparison.csv"),
            "--out-md", (Join-Path $OutRoot "comparison/test_comparison.md")
        )
    }
    $BestJepa = Get-BestJepaRun $ComparisonCsv
    $BestName = $BestJepa.model_name
    $BaselineCkpt = Join-Path $OutRoot "baseline/best_model.pt"
    $BaselineCfg = Join-Path $OutRoot "baseline/run_config.json"
    $JepaCkpt = Join-Path $OutRoot "$BestName/best_model.pt"
    $JepaCfg = Join-Path $OutRoot "$BestName/run_config.json"
    Write-Host "$Name best JEPA: $BestName mIoU=$($BestJepa.mIoU) macro_F1=$($BestJepa.macro_F1)"

    Invoke-Python "$Name geo maps" @(
        "scripts/09_export_geo_predictions.py",
        "--data", $Data,
        "--baseline-checkpoint", $BaselineCkpt,
        "--jepa-checkpoint", $JepaCkpt,
        "--baseline-run-config", $BaselineCfg,
        "--jepa-run-config", $JepaCfg,
        "--out", (Join-Path $ForestOut "geo_maps"),
        "--max-tiles", "$MapTiles"
    )
    Invoke-Python "$Name LAZ exports" @(
        "scripts/10_export_predictions_laz.py",
        "--data", $Data,
        "--baseline-checkpoint", $BaselineCkpt,
        "--jepa-checkpoint", $JepaCkpt,
        "--baseline-run-config", $BaselineCfg,
        "--jepa-run-config", $JepaCfg,
        "--out", (Join-Path $ForestOut "laz_exports"),
        "--max-tiles", "$LazTiles"
    )
    Invoke-Python "$Name forest eval" @(
        "scripts/08_forest_eval.py",
        "--data", $Data,
        "--baseline-checkpoint", $BaselineCkpt,
        "--jepa-checkpoint", $JepaCkpt,
        "--baseline-run-config", $BaselineCfg,
        "--jepa-run-config", $JepaCfg,
        "--out", $ForestOut,
        "--grid-size", "$GridSize"
    )
    Invoke-Python "$Name forest maps" @(
        "scripts/13_export_forest_maps.py",
        "--grid-csv", (Join-Path $ForestOut "forest_grid_metrics.csv"),
        "--anomalies-csv", (Join-Path $ForestOut "forest_anomalies.csv"),
        "--out", (Join-Path $ForestOut "maps")
    )
    Invoke-Python "$Name GIS grid" @(
        "scripts/12_export_geopackage_grid.py",
        "--grid-csv", (Join-Path $ForestOut "forest_grid_metrics.csv"),
        "--out-dir", (Join-Path $ForestOut "geopackage"),
        "--grid-size", "$GridSize"
    )
}

try {
    Write-Host "Overnight Geo-JEPA / Forest-JEPA run started: $(Get-Date)"
    Write-Host "Transcript: $TranscriptPath"

    Invoke-Python "medium semifrozen 0.05" @(
        "scripts/05_finetune_jepa.py",
        "--data", "data/processed/galicia_blocks_medium_tw",
        "--checkpoint", "outputs/medium/tw_jepa_pretrain/best_jepa.pt",
        "--out", "outputs/medium/jepa_semifrozen_005",
        "--epochs", "30",
        "--batch-size", "$BatchSize",
        "--num-workers", "$Workers",
        "--seed", "$Seed",
        "--encoder-lr-scale", "0.05",
        "--probe-type", "mlp",
        "--class-weight-mode", "inverse_sqrt",
        "--max-class-weight", "20.0",
        "--max-train-blocks", "5000",
        "--max-val-blocks", "1000",
        "--early-stopping-patience", "8",
        "--early-stopping-min-delta", "$EarlyStoppingMinDelta"
    )

    Invoke-Python "medium semifrozen 0.10" @(
        "scripts/05_finetune_jepa.py",
        "--data", "data/processed/galicia_blocks_medium_tw",
        "--checkpoint", "outputs/medium/tw_jepa_pretrain/best_jepa.pt",
        "--out", "outputs/medium/jepa_semifrozen_01",
        "--epochs", "30",
        "--batch-size", "$BatchSize",
        "--num-workers", "$Workers",
        "--seed", "$Seed",
        "--encoder-lr-scale", "0.1",
        "--probe-type", "mlp",
        "--class-weight-mode", "inverse_sqrt",
        "--max-class-weight", "20.0",
        "--max-train-blocks", "5000",
        "--max-val-blocks", "1000",
        "--early-stopping-patience", "8",
        "--early-stopping-min-delta", "$EarlyStoppingMinDelta"
    )

    Invoke-Python "medium comparison" @(
        "scripts/07_compare_results.py",
        "--experiments-root", "outputs/medium",
        "--out-csv", "outputs/medium/comparison/test_comparison.csv",
        "--out-md", "outputs/medium/comparison/test_comparison.md"
    )
    Export-Analysis -Name "medium" -Data "data/processed/galicia_blocks_medium_tw" -OutRoot "outputs/medium" -ForestOut "outputs/forest_jepa_medium" -MapTiles 8 -LazTiles 4

    try {
        Invoke-Pipeline -StepName "full pipeline high batch" -TrainBatchSize $BatchSize -PretrainBatchSize $JepaBatchSize
    }
    catch {
        Write-Host "High-batch full pipeline failed: $_"
        Write-Host "Retrying full pipeline with stable batch sizes 24/48 and resume enabled."
        Invoke-Pipeline -StepName "full pipeline stable retry" -TrainBatchSize 24 -PretrainBatchSize 48
    }

    Export-Analysis -Name "full" -Data "data/processed/galicia_blocks_full_tw" -OutRoot "outputs/full" -ForestOut "outputs/forest_jepa_full" -MapTiles 12 -LazTiles 6

    $MediumBest = Get-BestJepaRun "outputs/medium/comparison/test_comparison.csv"
    $FullBest = Get-BestJepaRun "outputs/full/comparison/test_comparison.csv"
    $Summary = @(
        "# Overnight Geo-JEPA / Forest-JEPA Summary",
        "",
        "- Finished: $(Get-Date)",
        "- Transcript: $TranscriptPath",
        "- Medium best JEPA: $($MediumBest.model_name) mIoU=$($MediumBest.mIoU) macro_F1=$($MediumBest.macro_F1)",
        "- Full best JEPA: $($FullBest.model_name) mIoU=$($FullBest.mIoU) macro_F1=$($FullBest.macro_F1)",
        "",
        "## Outputs",
        "",
        "- Medium comparison: outputs/medium/comparison/test_comparison.md",
        "- Medium forest report: outputs/forest_jepa_medium/forest_report.md",
        "- Full comparison: outputs/full/comparison/test_comparison.md",
        "- Full forest report: outputs/forest_jepa_full/forest_report.md"
    )
    New-Item -ItemType Directory -Force -Path "outputs/overnight" | Out-Null
    $Summary | Set-Content -Path "outputs/overnight/summary.md" -Encoding UTF8
    Write-Host "Overnight run completed: $(Get-Date)"
}
finally {
    Stop-Transcript | Out-Null
}
