param()

$root = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$wheelDir = Join-Path $root "vendor/python_wheels"
$nodeVendorDir = Join-Path $root "vendor/node_modules"
$packageLockVendor = Join-Path $root "vendor/package-lock.json"
$frontendDir = Join-Path $root "frontend"

New-Item -ItemType Directory -Force -Path $wheelDir | Out-Null
New-Item -ItemType Directory -Force -Path $nodeVendorDir | Out-Null

Write-Host "[bootstrap] Cleaning previous wheel cache"
Get-ChildItem -Path $wheelDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

$pythonArgs = @('-m', 'pip', 'download', '--dest', $wheelDir, '--requirement', (Join-Path $root 'REQUIREMENTS.txt'))
if ($env:PYTHON_BIN) {
    Write-Host "[bootstrap] Downloading Python wheels using $($env:PYTHON_BIN)"
    & $env:PYTHON_BIN @pythonArgs
} else {
    Write-Host "[bootstrap] Downloading Python wheels using py -3"
    & py -3 @pythonArgs
}

$npmBin = if ($env:NPM_BIN) { $env:NPM_BIN } else { 'npm' }
Write-Host "[bootstrap] Installing frontend dependencies with $npmBin"
Push-Location $frontendDir
& $npmBin install --no-audit --no-fund
$playwrightPath = Join-Path $frontendDir 'node_modules/.bin/playwright.cmd'
if (-not (Test-Path $playwrightPath)) {
    $playwrightPath = Join-Path $frontendDir 'node_modules/.bin/playwright'
}
if (Test-Path $playwrightPath) {
    Write-Host "[bootstrap] Installing Playwright browsers"
    & $playwrightPath install --with-deps
} else {
    Write-Warning "Playwright binary not found; skipping browser installation"
}
Pop-Location

Write-Host "[bootstrap] Syncing node modules into vendor directory"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $nodeVendorDir
New-Item -ItemType Directory -Force -Path $nodeVendorDir | Out-Null
Copy-Item -Recurse -Force (Join-Path $frontendDir 'node_modules/*') $nodeVendorDir
Copy-Item -Force (Join-Path $frontendDir 'package-lock.json') $packageLockVendor

Write-Host "[bootstrap] Creating node_modules.zip archive"
$zipPath = Join-Path $root 'vendor/node_modules.zip'
Remove-Item -ErrorAction SilentlyContinue $zipPath
Compress-Archive -Path (Join-Path $nodeVendorDir '*') -DestinationPath $zipPath

Write-Host "[bootstrap] Done. Vendored artifacts are ready for offline use."
