param(
    [string]$Python = "python",
    [string]$Raw = "data/raw/pnoa_galicia",
    [string]$Blocks = "data/processed/galicia_blocks_mixed",
    [string]$BlocksTW = "data/processed/galicia_blocks_mixed_tw",
    [string]$Reports = "reports",
    [string]$BaselineOut = "outputs/baseline",
    [string]$JepaPretrainOut = "outputs/tw_jepa_pretrain",
    [string]$JepaFinetuneOut = "outputs/tw_jepa_finetune",
    [string]$ComparisonOut = "outputs/comparison",
    [string]$GeoExperimentsOut = "outputs/geo_jepa",
    [int]$Workers = 8,
    [int]$PrepareWorkers = 0,
    [int]$PointsPerBlock = 8192,
    [string]$SplitMode = "mixed",
    [double]$ValRatio = 0.1,
    [double]$TestRatio = 0.1,
    [int]$BatchSize = 24,
    [int]$JepaBatchSize = 48,
    [int]$BaselineEpochs = 40,
    [int]$JepaEpochs = 80,
    [int]$FinetuneEpochs = 60,
    [int]$EarlyStoppingPatience = 10,
    [double]$EarlyStoppingMinDelta = 0.001,
    [int]$PretrainEarlyStoppingPatience = 15,
    [double]$PretrainEarlyStoppingMinDelta = 0.0001,
    [int]$Seed = 42,
    [string]$ClassWeightMode = "inverse_sqrt",
    [double]$MaxClassWeight = 20.0,
    [int]$MaxTiles = 0,
    [int]$MaxTWBlocks = 0,
    [int]$MaxTrainBlocks = 0,
    [int]$MaxValBlocks = 0,
    [switch]$Quick,
    [switch]$SkipTW,
    [switch]$SkipTraining,
    [switch]$SkipComparison,
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

if ($PrepareWorkers -le 0) {
    $PrepareWorkers = $Workers
}

if ($Quick) {
    $Blocks = "data/processed/galicia_blocks_pilot"
    $BlocksTW = "data/processed/galicia_blocks_pilot_tw"
    $BaselineOut = "outputs/pilot/baseline"
    $JepaPretrainOut = "outputs/pilot/tw_jepa_pretrain"
    $JepaFinetuneOut = "outputs/pilot/tw_jepa_finetune"
    $ComparisonOut = "outputs/pilot/comparison"
    $GeoExperimentsOut = "outputs/pilot"
    $PointsPerBlock = 4096
    $BaselineEpochs = 5
    $JepaEpochs = 10
    $FinetuneEpochs = 5
    $MaxTiles = 24
    $MaxTWBlocks = 0
    $MaxTrainBlocks = 1500
    $MaxValBlocks = 300
}

$NoSkipExistingArgs = @()
$NoResumeArgs = @()
if ($ForceRestart) {
    $NoSkipExistingArgs = @("--no-skip-existing")
    $NoResumeArgs = @("--no-resume")
}

$PrepareLimitArgs = @()
if ($MaxTiles -gt 0) {
    $PrepareLimitArgs = @("--max-tiles", $MaxTiles)
}

$TWLimitArgs = @()
if ($MaxTWBlocks -gt 0) {
    $TWLimitArgs = @("--max-blocks", $MaxTWBlocks)
}

$TrainLimitArgs = @()
if ($MaxTrainBlocks -gt 0) {
    $TrainLimitArgs += @("--max-train-blocks", $MaxTrainBlocks)
}
if ($MaxValBlocks -gt 0) {
    $TrainLimitArgs += @("--max-val-blocks", $MaxValBlocks)
}

$JepaLimitArgs = @()
if ($MaxTrainBlocks -gt 0) {
    $JepaLimitArgs = @("--max-blocks", $MaxTrainBlocks)
}

& $Python scripts/00_audit_pnoa_pairs.py --raw $Raw --reports $Reports
& $Python scripts/01_prepare_tiles.py --raw $Raw --out $Blocks --reports $Reports --points-per-block $PointsPerBlock --split-mode $SplitMode --val-ratio $ValRatio --test-ratio $TestRatio --num-workers $PrepareWorkers @PrepareLimitArgs @NoSkipExistingArgs

if (-not $SkipTW) {
    & $Python scripts/02_compute_tw_features.py --input $Blocks --out $BlocksTW --reports $Reports --num-workers $Workers @TWLimitArgs @NoSkipExistingArgs
}

if (-not $SkipTraining) {
    $DataForTraining = $(if ($SkipTW) { $Blocks } else { $BlocksTW })
    & $Python scripts/11_run_geo_jepa_pilot.py --python $Python --data $DataForTraining --out-root $GeoExperimentsOut --pretrain-out $JepaPretrainOut --epochs-baseline $BaselineEpochs --epochs-pretrain $JepaEpochs --epochs-probe $FinetuneEpochs --batch-size $BatchSize --jepa-batch-size $JepaBatchSize --num-workers $Workers --max-train-blocks $MaxTrainBlocks --max-val-blocks $MaxValBlocks --seed $Seed --class-weight-mode $ClassWeightMode --max-class-weight $MaxClassWeight --early-stopping-patience $EarlyStoppingPatience --early-stopping-min-delta $EarlyStoppingMinDelta --pretrain-early-stopping-patience $PretrainEarlyStoppingPatience --pretrain-early-stopping-min-delta $PretrainEarlyStoppingMinDelta @NoResumeArgs
}
