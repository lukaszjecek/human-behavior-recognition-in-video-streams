[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [Alias("Input")]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [string]$Config = "configs/data_pipeline.yml",
    [string]$Output = "data/logs/benchmark_summary.json",
    [ValidateSet("auto", "cpu", "cuda", "mps")]
    [string]$Device = "auto",
    [switch]$Gpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-DockerComposeCommand {
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        & docker compose version *> $null
        if ($LASTEXITCODE -eq 0) {
            return @("docker", "compose")
        }
    }

    if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
        & docker-compose version *> $null
        if ($LASTEXITCODE -eq 0) {
            return @("docker-compose")
        }
    }

    throw "Docker Compose is unavailable. Install Docker Desktop or ensure 'docker compose' is on PATH."
}

function Invoke-Compose {
    param(
        [string[]]$Arguments
    )

    $exe = $script:ComposeCommand[0]
    $baseArgs = @()
    if ($script:ComposeCommand.Count -gt 1) {
        $baseArgs = $script:ComposeCommand[1..($script:ComposeCommand.Count - 1)]
    }

    & $exe @baseArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Test-IsAppPath {
    param([string]$Path)

    $normalized = $Path.Replace("\", "/")
    return $normalized.StartsWith("/app/")
}

function Convert-AppPathToHostPath {
    param([string]$Path)

    $relative = $Path.Replace("\", "/").Substring(5)
    return [System.IO.Path]::GetFullPath((Join-Path $script:RepoRoot ($relative.Replace("/", [System.IO.Path]::DirectorySeparatorChar))))
}

function Get-HostPath {
    param(
        [string]$Path,
        [switch]$MustExist
    )

    if (Test-IsAppPath -Path $Path) {
        $hostPath = Convert-AppPathToHostPath -Path $Path
    }
    elseif ([System.IO.Path]::IsPathRooted($Path)) {
        $hostPath = [System.IO.Path]::GetFullPath($Path)
    }
    else {
        $hostPath = [System.IO.Path]::GetFullPath((Join-Path $script:RepoRoot $Path))
    }

    if ($MustExist -and -not (Test-Path -LiteralPath $hostPath -PathType Leaf)) {
        throw "Required file does not exist: $hostPath"
    }

    return $hostPath
}

function Convert-HostPathToContainerPath {
    param([string]$HostPath)

    $repoRootWithSeparator = $script:RepoRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    $fullPath = [System.IO.Path]::GetFullPath($HostPath)

    if (-not $fullPath.StartsWith($repoRootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path must be inside the repository so Docker Compose can mount it: $fullPath"
    }

    $relative = $fullPath.Substring($repoRootWithSeparator.Length).Replace("\", "/")
    return "/app/$relative"
}

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $script:RepoRoot

$inputHostPath = Get-HostPath -Path $InputPath -MustExist
$checkpointHostPath = Get-HostPath -Path $Checkpoint -MustExist
$configHostPath = Get-HostPath -Path $Config -MustExist
$outputHostPath = Get-HostPath -Path $Output

if ([System.IO.Path]::GetExtension($inputHostPath).ToLowerInvariant() -ne ".mp4") {
    throw "Input must be an MP4 file: $inputHostPath"
}

$outputDirectory = Split-Path -Parent $outputHostPath
if ($outputDirectory) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

$containerInput = if (Test-IsAppPath -Path $InputPath) { $InputPath.Replace("\", "/") } else { Convert-HostPathToContainerPath -HostPath $inputHostPath }
$containerCheckpoint = if (Test-IsAppPath -Path $Checkpoint) { $Checkpoint.Replace("\", "/") } else { Convert-HostPathToContainerPath -HostPath $checkpointHostPath }
$containerConfig = if (Test-IsAppPath -Path $Config) { $Config.Replace("\", "/") } else { Convert-HostPathToContainerPath -HostPath $configHostPath }
$containerOutput = if (Test-IsAppPath -Path $Output) { $Output.Replace("\", "/") } else { Convert-HostPathToContainerPath -HostPath $outputHostPath }

$script:ComposeCommand = @(Get-DockerComposeCommand)
Write-Host "Using Docker Compose command: $($script:ComposeCommand -join ' ')"
if ($Gpu) {
    Write-Host "Compose mode: GPU override (compose.yaml + compose.gpu.yaml)"
}
else {
    Write-Host "Compose mode: default CPU-safe compose.yaml"
}
Write-Host "Running MP4 inference benchmark..."
Write-Host "  Input:      $inputHostPath"
Write-Host "  Checkpoint: $checkpointHostPath"
Write-Host "  Config:     $configHostPath"
Write-Host "  Output:     $outputHostPath"
Write-Host "  Device:     $Device"

$composeArgs = @()
if ($Gpu) {
    $composeArgs += @(
        "-f",
        "compose.yaml",
        "-f",
        "compose.gpu.yaml"
    )
}

$composeArgs += @(
    "run",
    "--rm",
    "inference",
    "python",
    "scripts/benchmark_mp4_inference.py",
    "--input",
    $containerInput,
    "--checkpoint",
    $containerCheckpoint,
    "--config",
    $containerConfig,
    "--output",
    $containerOutput,
    "--device",
    $Device
)

Invoke-Compose -Arguments $composeArgs

Write-Host ""
Write-Host "MP4 inference benchmark completed."
Write-Host "Benchmark JSON: $outputHostPath"
