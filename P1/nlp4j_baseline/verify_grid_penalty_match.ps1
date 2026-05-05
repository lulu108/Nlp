#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$GridRoot = "output_grid",
    [switch]$FailOnMismatch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $GridRoot -PathType Container)) {
    throw "Grid root not found: $GridRoot"
}

$dirs = Get-ChildItem -LiteralPath $GridRoot -Directory | Sort-Object Name
if ($dirs.Count -eq 0) {
    throw "No run directories found under: $GridRoot"
}

$results = New-Object System.Collections.Generic.List[object]
$badCount = 0

foreach ($d in $dirs) {
    $tag = $d.Name
    if ($tag -notmatch '^itp(?<itp>m?\d+p\d+)_sp(?<sp>m?\d+p\d+)_ep(?<ep>m?\d+p\d+)$') {
        $results.Add([PSCustomObject]@{
            tag = $tag
            status = 'BAD_TAG_FORMAT'
            expected_itp = $null
            expected_sp = $null
            expected_ep = $null
            actual_itp = $null
            actual_sp = $null
            actual_ep = $null
            metrics_path = ''
        }) | Out-Null
        $badCount++
        continue
    }

    function Decode-TagNum([string]$s) {
        $x = $s.Replace('m','-').Replace('p','.')
        return [double]$x
    }

    $expectedItp = Decode-TagNum $Matches['itp']
    $expectedSp  = Decode-TagNum $Matches['sp']
    $expectedEp  = Decode-TagNum $Matches['ep']

    $metricsPath = Join-Path $d.FullName 'nlp4j_hmm_metrics.json'
    if (-not (Test-Path -LiteralPath $metricsPath -PathType Leaf)) {
        $results.Add([PSCustomObject]@{
            tag = $tag
            status = 'MISSING_METRICS'
            expected_itp = $expectedItp
            expected_sp = $expectedSp
            expected_ep = $expectedEp
            actual_itp = $null
            actual_sp = $null
            actual_ep = $null
            metrics_path = $metricsPath
        }) | Out-Null
        $badCount++
        continue
    }

    $obj = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    $actualItp = [double]$obj.illegal_transition_penalty
    $actualSp  = [double]$obj.start_penalty
    $actualEp  = [double]$obj.end_penalty

    $ok = ($actualItp -eq $expectedItp) -and ($actualSp -eq $expectedSp) -and ($actualEp -eq $expectedEp)
    if (-not $ok) { $badCount++ }

    $results.Add([PSCustomObject]@{
        tag = $tag
        status = $(if ($ok) { 'OK' } else { 'MISMATCH' })
        expected_itp = $expectedItp
        expected_sp = $expectedSp
        expected_ep = $expectedEp
        actual_itp = $actualItp
        actual_sp = $actualSp
        actual_ep = $actualEp
        metrics_path = $metricsPath
    }) | Out-Null
}

$results | Sort-Object tag | Format-Table -AutoSize

$reportPath = Join-Path $GridRoot 'penalty_match_report.tsv'
$results | Sort-Object tag | Export-Csv -LiteralPath $reportPath -Delimiter "`t" -NoTypeInformation -Encoding UTF8
Write-Host "`nReport saved: $reportPath"

if ($badCount -gt 0) {
    Write-Host "Found $badCount problematic directories." -ForegroundColor Yellow
    if ($FailOnMismatch) {
        throw "Penalty verification failed with $badCount mismatches/problems."
    }
} else {
    Write-Host "All directories matched tag penalties." -ForegroundColor Green
}
