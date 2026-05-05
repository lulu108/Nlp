#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$TestPath = "../datasets/auto/test.txt",
    [string]$ModelPath = "model/hmm_bmes_model.json",
    [string]$OutputRoot = "output_grid",
    [double[]]$IllegalTransitionPenaltyList = @(-0.2, -0.5, -1.0),
    [double[]]$StartPenaltyList = @(-0.2, -0.6, -1.0),
    [switch]$CleanOutputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-FileExists([string]$path, [string]$name) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing file [$name]: $path"
    }
}

function Get-FileSha256([string]$path) {
    (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}

function Get-MLabelMetrics([string]$labelReportPath) {
    $lines = Get-Content -LiteralPath $labelReportPath
    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line -match '^label\t') {
            continue
        }
        $cols = $line -split "\t"
        if ($cols.Length -lt 5) {
            continue
        }
        if ($cols[0] -eq "M") {
            return [PSCustomObject]@{
                m_precision = [double]$cols[1]
                m_recall = [double]$cols[2]
                m_f1 = [double]$cols[3]
                m_support = [int]$cols[4]
            }
        }
    }
    return [PSCustomObject]@{
        m_precision = 0.0
        m_recall = 0.0
        m_f1 = 0.0
        m_support = 0
    }
}

$initialDir = Get-Location
Set-Location $ProjectDir

Assert-FileExists $TestPath "TestPath"
Assert-FileExists $ModelPath "ModelPath"
Assert-FileExists "target/classes/Nlp4jHmmPredictor.class" "PredictorClass"

if ($CleanOutputRoot -and (Test-Path -LiteralPath $OutputRoot)) {
    Remove-Item -LiteralPath $OutputRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$summaryRows = New-Object System.Collections.Generic.List[object]

foreach ($itp in $IllegalTransitionPenaltyList) {
    foreach ($sp in $StartPenaltyList) {
        $ep = $sp
        $tag = "itp${itp}_sp${sp}_ep${ep}".Replace("-", "m").Replace(".", "p")
        $outDir = Join-Path $OutputRoot $tag
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null

        Write-Host "Running itp=$itp sp=$sp ep=$ep" -ForegroundColor Yellow

        $argLine = "$TestPath $ModelPath output $itp $sp $ep"
        $process = Start-Process -FilePath "mvn" -ArgumentList @(
            "-q",
            "exec:java",
            "-Dexec.mainClass=Nlp4jHmmPredictor",
            "-Dexec.args=$argLine"
        ) -NoNewWindow -Wait -PassThru

        if ($process.ExitCode -ne 0) {
            throw "mvn exec:java failed (ExitCode=$($process.ExitCode)) for itp=$itp sp=$sp ep=$ep"
        }

        $metrics = "output/nlp4j_hmm_metrics.json"
        $label = "output/nlp4j_hmm_label_report.tsv"
        $cm = "output/nlp4j_hmm_confusion_matrix.tsv"

        Assert-FileExists $metrics "Metrics"
        Assert-FileExists $label "LabelReport"
        Assert-FileExists $cm "ConfusionMatrix"

        $metricsObj = Get-Content -LiteralPath $metrics -Raw | ConvertFrom-Json
        if ($null -eq $metricsObj.macro_f1 -or $null -eq $metricsObj.tag_accuracy) {
            throw "metrics JSON missing fields for itp=$itp sp=$sp ep=$ep"
        }

        Copy-Item -LiteralPath $metrics -Destination (Join-Path $outDir "nlp4j_hmm_metrics.json") -Force
        Copy-Item -LiteralPath $label -Destination (Join-Path $outDir "nlp4j_hmm_label_report.tsv") -Force
        Copy-Item -LiteralPath $cm -Destination (Join-Path $outDir "nlp4j_hmm_confusion_matrix.tsv") -Force

        $mMetrics = Get-MLabelMetrics (Join-Path $outDir "nlp4j_hmm_label_report.tsv")

        $summaryRows.Add([PSCustomObject]@{
            tag = $tag
            illegal_transition_penalty = $itp
            start_penalty = $sp
            end_penalty = $ep
            tag_accuracy = [double]$metricsObj.tag_accuracy
            macro_f1 = [double]$metricsObj.macro_f1
            m_precision = [double]$mMetrics.m_precision
            m_recall = [double]$mMetrics.m_recall
            m_f1 = [double]$mMetrics.m_f1
            metrics_sha256 = Get-FileSha256 (Join-Path $outDir "nlp4j_hmm_metrics.json")
            confusion_sha256 = Get-FileSha256 (Join-Path $outDir "nlp4j_hmm_confusion_matrix.tsv")
        }) | Out-Null
    }
}

$best = $summaryRows |
    Sort-Object -Property macro_f1, m_f1, tag_accuracy -Descending |
    Select-Object -First 1

$summaryRows = $summaryRows | ForEach-Object {
    $_ | Add-Member -NotePropertyName "best" -NotePropertyValue ($_.tag -eq $best.tag) -Force
    $_
}

$summaryPath = Join-Path $OutputRoot "grid_summary.tsv"
$summaryRows |
    Sort-Object -Property illegal_transition_penalty, start_penalty |
    Export-Csv -LiteralPath $summaryPath -Delimiter "`t" -NoTypeInformation -Encoding UTF8

$uniqueMetrics = ($summaryRows | Select-Object -ExpandProperty metrics_sha256 | Select-Object -Unique).Count
$uniqueConfusion = ($summaryRows | Select-Object -ExpandProperty confusion_sha256 | Select-Object -Unique).Count
if ($uniqueMetrics -eq 1 -and $uniqueConfusion -eq 1) {
    Write-Warning "All outputs are identical; penalties did not change paths."
}

Write-Host "Done. Summary: $summaryPath" -ForegroundColor Green
Write-Host "Best: $($best.tag) | macro_f1=$($best.macro_f1) | m_f1=$($best.m_f1) | acc=$($best.tag_accuracy)" -ForegroundColor Green

Set-Location $initialDir
