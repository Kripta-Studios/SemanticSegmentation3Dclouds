param(
    [string]$Python = "python",
    [int]$Workers = 8,
    [int]$PrepareWorkers = 8,
    [int]$BatchSize = 24,
    [int]$JepaBatchSize = 48,
    [int]$Seed = 42,
    [double]$GridSize = 2.0,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$PipelineArgs = @()
if ($Quick) {
    $PipelineArgs += "-Quick"
}

& powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 `
    -Python $Python `
    -Workers $Workers `
    -PrepareWorkers $PrepareWorkers `
    -BatchSize $BatchSize `
    -JepaBatchSize $JepaBatchSize `
    -Seed $Seed `
    @PipelineArgs

$Data = $(if ($Quick) { "data/processed/galicia_blocks_pilot_tw" } else { "data/processed/galicia_blocks_mixed_tw" })
$OutRoot = $(if ($Quick) { "outputs/pilot" } else { "outputs/geo_jepa" })

& $Python scripts/09_export_geo_predictions.py `
    --data $Data `
    --baseline-checkpoint "$OutRoot/baseline/best_model.pt" `
    --jepa-checkpoint "$OutRoot/jepa_full_finetune/best_model.pt" `
    --baseline-run-config "$OutRoot/baseline/run_config.json" `
    --jepa-run-config "$OutRoot/jepa_full_finetune/run_config.json" `
    --out outputs/pilot_demo/maps

& $Python scripts/10_export_predictions_laz.py `
    --data $Data `
    --baseline-checkpoint "$OutRoot/baseline/best_model.pt" `
    --jepa-checkpoint "$OutRoot/jepa_full_finetune/best_model.pt" `
    --baseline-run-config "$OutRoot/baseline/run_config.json" `
    --jepa-run-config "$OutRoot/jepa_full_finetune/run_config.json" `
    --out outputs/pilot_demo/laz_exports

& $Python scripts/08_forest_eval.py `
    --data $Data `
    --baseline-checkpoint "$OutRoot/baseline/best_model.pt" `
    --jepa-checkpoint "$OutRoot/jepa_full_finetune/best_model.pt" `
    --baseline-run-config "$OutRoot/baseline/run_config.json" `
    --jepa-run-config "$OutRoot/jepa_full_finetune/run_config.json" `
    --out outputs/forest_jepa `
    --grid-size $GridSize

& $Python scripts/13_export_forest_maps.py `
    --grid-csv outputs/forest_jepa/forest_grid_metrics.csv `
    --anomalies-csv outputs/forest_jepa/forest_anomalies.csv `
    --out outputs/forest_jepa/maps

& $Python scripts/12_export_geopackage_grid.py `
    --grid-csv outputs/forest_jepa/forest_grid_metrics.csv `
    --out-dir outputs/pilot_demo/geopackage `
    --grid-size $GridSize
