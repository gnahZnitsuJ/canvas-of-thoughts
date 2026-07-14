[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet(
        "plan",
        "build-check",
        "checkpoint-check",
        "architecture-check",
        "compile-baseline",
        "compile-dev",
        "shell-dev",
        "smoke-no-compile",
        "smoke-compile"
    )]
    [string]$Workflow,

    [string]$CheckpointPath = "architecture_transition_baseline.pkl",
    [string]$DevelopmentCheckpointPath = "development_zero_nosolver.pkl",
    [string]$PythonPath,
    [switch]$ShowOnly,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $PythonPath) {
    $venvPython = Join-Path $repoRoot "..\.venv\Scripts\python.exe"
    $PythonPath = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
}

$modelMain = Join-Path $repoRoot "model\main.py"
$baseline = @("--probe-mode", "debug", "--compile-profile", "full", "--learned-init-mode", "random-function")
$development = @("--probe-mode", "minimal", "--compile-profile", "fast-solver", "--learned-init-mode", "zero-nosolver")

function Invoke-ModelCommand {
    param([string[]]$Arguments)

    $resolved = @($modelMain) + $Arguments + $ExtraArgs
    $display = @($PythonPath) + $resolved | ForEach-Object {
        if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ }
    }
    Write-Host ("Resolved command: " + ($display -join " "))
    if (-not $ShowOnly) {
        & $PythonPath @resolved
        if ($LASTEXITCODE -ne 0) {
            throw "Model command failed with exit code $LASTEXITCODE."
        }
    }
}

switch ($Workflow) {
    "plan" {
        Invoke-ModelCommand @("--dry-run", "--checkpoint-path", $CheckpointPath)
    }
    "build-check" {
        Invoke-ModelCommand (@("--build-only") + $development)
    }
    "checkpoint-check" {
        Invoke-ModelCommand @("--inspect-checkpoint", "--checkpoint-path", $CheckpointPath)
    }
    "architecture-check" {
        Invoke-ModelCommand (
            @(
                "--build-only",
                "--inspect-checkpoint",
                "--compare-current-architecture",
                "--checkpoint-path",
                $CheckpointPath
            ) + $baseline
        )
    }
    "compile-baseline" {
        Invoke-ModelCommand (@("--benchmark", "compile-current") + $baseline)
    }
    "compile-dev" {
        Invoke-ModelCommand (@("--benchmark", "compile-current") + $development)
    }
    "shell-dev" {
        Invoke-ModelCommand (
            @("--shell", "--checkpoint-path", $DevelopmentCheckpointPath) + $development
        )
    }
    "smoke-no-compile" {
        Invoke-ModelCommand @("--dry-run", "--checkpoint-path", $CheckpointPath)
        Invoke-ModelCommand (@("--build-only") + $development)
        Invoke-ModelCommand (
            @(
                "--build-only",
                "--inspect-checkpoint",
                "--compare-current-architecture",
                "--checkpoint-path",
                $CheckpointPath
            ) + $baseline
        )
    }
    "smoke-compile" {
        Invoke-ModelCommand (
            @(
                "--eval",
                "--max-examples",
                "1",
                "--checkpoint-path",
                $DevelopmentCheckpointPath
            ) + $development
        )
    }
}
