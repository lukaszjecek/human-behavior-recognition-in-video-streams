[CmdletBinding()]
param(
    [string]$ApiUrl = "http://localhost:8000",
    [switch]$SkipBuild,
    [switch]$NoStart,
    [int]$HealthTimeoutSeconds = 90
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RequiredDirectories = @(
    "data/raw",
    "data/logs",
    "data/uploads",
    "data/subset"
)

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

function Wait-ApiHealth {
    param(
        [string]$Url,
        [int]$TimeoutSeconds
    )

    $healthUrl = $Url.TrimEnd("/") + "/health"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    Write-Host "Waiting for API health at $healthUrl ..."
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3
            if ($response.status -eq "ok") {
                Write-Host "API is healthy."
                return
            }
        }
        catch {
            Start-Sleep -Seconds 2
            continue
        }

        Start-Sleep -Seconds 2
    }

    throw "API did not become healthy within $TimeoutSeconds seconds at $healthUrl."
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$script:ComposeCommand = @(Get-DockerComposeCommand)
Write-Host "Using Docker Compose command: $($script:ComposeCommand -join ' ')"

foreach ($directory in $RequiredDirectories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $directory) | Out-Null
}
Write-Host "Required local data folders are present."

if (-not $NoStart) {
    $upArgs = @("up", "-d")
    if (-not $SkipBuild) {
        $upArgs += "--build"
    }

    Write-Host "Starting Docker Compose stack..."
    Invoke-Compose -Arguments $upArgs
}
else {
    Write-Host "Skipping stack startup because -NoStart was provided."
}

Wait-ApiHealth -Url $ApiUrl -TimeoutSeconds $HealthTimeoutSeconds

Write-Host "Running integration smoke test in the API container..."
Invoke-Compose -Arguments @("exec", "-T", "-e", "API_URL=$ApiUrl", "api", "python", "scripts/integration_smoke_test.py")

Write-Host ""
Write-Host "Final demo infrastructure smoke verification passed."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  Frontend:       http://localhost:5173"
Write-Host "  Swagger UI:     $($ApiUrl.TrimEnd('/'))/docs"
Write-Host "  API health:     $($ApiUrl.TrimEnd('/'))/health"
Write-Host "  Logs directory: $((Join-Path $RepoRoot 'data/logs'))"
Write-Host "  Shutdown:       $($script:ComposeCommand -join ' ') down"
