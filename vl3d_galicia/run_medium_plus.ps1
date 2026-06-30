param(
    [string]$Python = "python",
    [string]$Data = "data/processed/galicia_blocks_medium_tw",
    [string]$OutRoot = "outputs/medium_plus",
    [int]$Workers = 8,
    [int]$BatchSize = 24,
    [int]$JepaBatchSize = 32,
    [int]$PretrainEpochs = 30,
    [int]$TrainEpochs = 30,
    [int]$MaxTrainBlocks = 12000,
    [int]$MaxValBlocks = 2000,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_MODULE_LOADING = "LAZY"
$env:OMP_NUM_THREADS = "12"
$env:MKL_NUM_THREADS = "12"
$env:OPENBLAS_NUM_THREADS = "12"

$ClassArgs = @("--class-weight-mode", "inverse_sqrt", "--max-class-weight", "20.0")
$FocusArgs = @(
    "--loss-type", "focal",
    "--focal-gamma", "1.5",
    "--balanced-sampler",
    "--sampler-alpha", "1.2",
    "--sampler-max-weight", "10.0",
    "--sampler-class-boost", "1:1.5,2:2.0,4:4.0",
    "--early-stopping-patience", "8",
    "--early-stopping-min-delta", "0.001"
)
$LimitArgs = @("--max-train-blocks", "$MaxTrainBlocks", "--max-val-blocks", "$MaxValBlocks")

function Invoke-Step {
    param([string]$Name, [string[]]$CommandArgs)
    Write-Host ""
    Write-Host "===== $Name ====="
    Write-Host ("> " + $Python + " " + ($CommandArgs -join " "))
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$BaselineArgs = @(
    "scripts/03_train_baseline.py",
    "--data", $Data,
    "--out", "$OutRoot/baseline",
    "--epochs", "$TrainEpochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "$Workers",
    "--use-tw-input",
    "--probe-type", "mlp",
    "--seed", "$Seed"
)
Invoke-Step "baseline TW focal sampler" ($BaselineArgs + $ClassArgs + $FocusArgs + $LimitArgs)

$JepaFrozenArgs = @(
    "scripts/05_finetune_jepa.py",
    "--data", $Data,
    "--checkpoint", "outputs/medium/tw_jepa_pretrain/best_jepa.pt",
    "--out", "$OutRoot/jepa_frozen_mlp",
    "--epochs", "$TrainEpochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "$Workers",
    "--freeze-encoder",
    "--probe-type", "mlp",
    "--seed", "$Seed"
)
Invoke-Step "JEPA 8ch frozen MLP focal sampler" ($JepaFrozenArgs + $ClassArgs + $FocusArgs + $LimitArgs)

Invoke-Step "TW-input JEPA pretrain" @(
    "scripts/04_pretrain_jepa.py",
    "--data", $Data,
    "--out", "$OutRoot/twinput_jepa_pretrain",
    "--epochs", "$PretrainEpochs",
    "--batch-size", "$JepaBatchSize",
    "--num-workers", "$Workers",
    "--use-tw-input",
    "--tw-target",
    "--seed", "$Seed",
    "--max-blocks", "$MaxTrainBlocks",
    "--early-stopping-patience", "8",
    "--early-stopping-min-delta", "0.0001"
)

$JepaTwFrozenArgs = @(
    "scripts/05_finetune_jepa.py",
    "--data", $Data,
    "--checkpoint", "$OutRoot/twinput_jepa_pretrain/best_jepa.pt",
    "--out", "$OutRoot/jepa_twinput_frozen_mlp",
    "--epochs", "$TrainEpochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "$Workers",
    "--use-tw-input",
    "--freeze-encoder",
    "--probe-type", "mlp",
    "--seed", "$Seed"
)
Invoke-Step "TW-input JEPA frozen MLP focal sampler" ($JepaTwFrozenArgs + $ClassArgs + $FocusArgs + $LimitArgs)

Invoke-Step "medium plus comparison" @(
    "scripts/07_compare_results.py",
    "--experiments-root", "$OutRoot",
    "--out-csv", "$OutRoot/comparison/test_comparison.csv",
    "--out-md", "$OutRoot/comparison/test_comparison.md"
)

Write-Host ""
Write-Host "medium_plus finished. Comparison: $OutRoot/comparison/test_comparison.md"
