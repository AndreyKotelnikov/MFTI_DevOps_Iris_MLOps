param(
    [string]$ImageName       = "iris-api",
    [string]$ImageTag        = "0.0.1",
    [string]$RegistryId      = "crpdc99upo405hm8s93v",
    [string]$ContainerName   = "iris-api",
    [string]$ServiceAccountId,
    [string]$FolderId
)

$ErrorActionPreference = "Stop"

Write-Host "==> Checking tools..."

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker not found in PATH."
    exit 1
}

if (-not (Get-Command yc -ErrorAction SilentlyContinue)) {
    Write-Error "Yandex Cloud CLI (yc) not found in PATH."
    exit 1
}

if (-not $ServiceAccountId -and -not $env:YC_SA_ID) {
    Write-Error "Service Account ID is not set. Use -ServiceAccountId or set env var YC_SA_ID."
    exit 1
}

if (-not $FolderId -and -not $env:YC_FOLDER_ID) {
    Write-Error "Folder ID is not set. Use -FolderId or set env var YC_FOLDER_ID."
    exit 1
}

# If parameters are passed, use them; otherwise take from env
$saId     = if ($ServiceAccountId) { $ServiceAccountId } else { $env:YC_SA_ID }
$folderId = if ($FolderId)        { $FolderId }        else { $env:YC_FOLDER_ID }

# Go to repo root (from deploy/)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

# ---- БИЛД ЧЕРЕЗ deploy/Dockerfile.api ----
$dockerfile = Join-Path "deploy" "Dockerfile.api"
Write-Host "==> Building Docker image ${ImageName}:${ImageTag} using $dockerfile ..."
docker build -f $dockerfile -t "${ImageName}:${ImageTag}" .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed."
    exit 1
}

Write-Host "==> Configuring Docker login for Yandex Container Registry..."
yc container registry configure-docker | Out-Null

$imageFullName = "cr.yandex/${RegistryId}/${ImageName}:${ImageTag}"
Write-Host "==> Tagging image as ${imageFullName}"
docker tag "${ImageName}:${ImageTag}" $imageFullName

Write-Host "==> Pushing image to Yandex Container Registry..."
docker push $imageFullName

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker push failed."
    exit 1
}

Write-Host "==> Checking if serverless container '${ContainerName}' exists..."

# Temporarily relax error handling so yc error does not stop script
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"

yc serverless container get `
    --name $ContainerName `
    --folder-id $folderId 2>$null

$containerExistsExitCode = $LASTEXITCODE

# Restore previous error handling
$ErrorActionPreference = $prevEAP

if ($containerExistsExitCode -ne 0) {
    Write-Host "==> Container not found, creating..."
    yc serverless container create `
        --name $ContainerName `
        --folder-id $folderId | Out-Null
}

Write-Host "==> Deploying new serverless container revision..."
yc serverless container revision deploy `
    --container-name $ContainerName `
    --image $imageFullName `
    --cores 1 `
    --memory 512MB `
    --concurrency 1 `
    --execution-timeout 30s `
    --service-account-id $saId `
    --folder-id $folderId

if ($LASTEXITCODE -ne 0) {
    Write-Error "Serverless container revision deploy failed."
    exit 1
}

Write-Host "==> Getting current container status..."
yc serverless container get `
    --name $ContainerName `
    --folder-id $folderId

Write-Host "==> Allowing unauthenticated invoke for container..."
yc serverless container allow-unauthenticated-invoke `
    $ContainerName `
    --folder-id $folderId

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to allow unauthenticated invoke."
    exit 1
}

Write-Host "==> Done."
