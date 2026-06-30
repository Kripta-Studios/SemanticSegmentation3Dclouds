param(
    [string]$Python = "python",
    [string]$Data = "data/processed/galicia_blocks_medium_tw",
    [string]$FeatureRoot = "data/processed/galicia_blocks_medium_dino",
    [string]$OutRoot = "outputs/dino_fusion",
    [string]$CompareRoot = "outputs/medium_plus",
    [ValidateSet("auto", "hf", "timm", "torchhub", "dinov2", "stat")]
    [string]$Backend = "hf",
    [string]$DinoModel = "facebook/dinov3-vits16-pretrain-lvd1689m",
    [string]$DinoRepoDir = "",
    [string]$DinoWeights = "",
    [ValidateSet("imagenet", "sat493m")]
    [string]$Normalize = "imagenet",
    [ValidateSet("rgb", "cir", "height", "nir_height_density", "rgb_nir_height")]
    [string]$ImageMode = "rgb_nir_height",
    [int]$GridSize = 128,
    [int]$DinoOutDim = 64,
    [ValidateSet("concat", "gated")]
    [string]$FusionType = "concat",
    [int]$Workers = 8,
    [int]$BatchSize = 24,
    [int]$TrainEpochs = 30,
    [int]$MaxTrainBlocks = 12000,
    [int]$MaxValBlocks = 2000,
    [int]$MaxTestBlocks = 0,
    [int]$MaxFeatureBlocksPerSplit = 0,
    [int]$Seed = 42,
    [switch]$ForceFeatures,
    [switch]$ForceTrain
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_MODULE_LOADING = "LAZY"
$env:OMP_NUM_THREADS = "12"
$env:MKL_NUM_THREADS = "12"
$env:OPENBLAS_NUM_THREADS = "12"

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
$ExperimentName = if ($FusionType -eq "gated") { "dino_tw_gated" } else { "dino_tw_fusion" }

$FeatureArgs = @(
    "scripts/14_build_dino_features.py",
    "--data", $Data,
    "--out", $FeatureRoot,
    "--backend", $Backend,
    "--model", $DinoModel,
    "--normalize", $Normalize,
    "--grid-size", "$GridSize",
    "--image-mode", $ImageMode,
    "--out-dim", "$DinoOutDim",
    "--projection-seed", "$Seed",
    "--max-blocks-per-split", "$MaxFeatureBlocksPerSplit",
    "--max-train-blocks", "$MaxTrainBlocks",
    "--max-val-blocks", "$MaxValBlocks",
    "--max-test-blocks", "$MaxTestBlocks"
)
if ($DinoRepoDir -ne "") { $FeatureArgs += @("--repo-dir", $DinoRepoDir) }
if ($DinoWeights -ne "") { $FeatureArgs += @("--weights", $DinoWeights) }
if ($ForceFeatures) { $FeatureArgs += @("--force") }
Invoke-Step "build DINO raster feature cache" $FeatureArgs

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
if ($MaxTestBlocks -gt 0) { $LimitArgs += @("--max-test-blocks", "$MaxTestBlocks") }

$TrainArgs = @(
    "scripts/03_train_baseline.py",
    "--data", $Data,
    "--out", "$OutRoot/$ExperimentName",
    "--epochs", "$TrainEpochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "$Workers",
    "--use-tw-input",
    "--external-feature-dir", $FeatureRoot,
    "--external-feature-key", "dino_features",
    "--fusion-type", $FusionType,
    "--probe-type", "mlp",
    "--seed", "$Seed"
)
if ($ForceTrain) { $TrainArgs += @("--no-resume") }
Invoke-Step "train DINO/TW point fusion" ($TrainArgs + $ClassArgs + $FocusArgs + $LimitArgs)

Invoke-Step "compare DINO fusion against medium_plus" @(
    "scripts/07_compare_results.py",
    "--experiments-root", $CompareRoot,
    "--experiments-root", $OutRoot,
    "--out-csv", "$OutRoot/comparison/test_comparison.csv",
    "--out-md", "$OutRoot/comparison/test_comparison.md"
)

Write-Host ""
Write-Host "DINO fusion finished. Comparison: $OutRoot/comparison/test_comparison.md"
