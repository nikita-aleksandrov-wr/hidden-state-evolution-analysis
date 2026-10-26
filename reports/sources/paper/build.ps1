$ErrorActionPreference = 'Stop'
$buildPath = Join-Path $PSScriptRoot '.build'
New-Item -ItemType Directory -Path $buildPath -Force | Out-Null
Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($pass in 1..2) {
        & pdflatex --disable-installer -interaction=nonstopmode -halt-on-error -file-line-error "-output-directory=$buildPath" main.tex |
            Out-File -LiteralPath (Join-Path $buildPath 'compiler_output.txt') -Encoding utf8
        if ($LASTEXITCODE -ne 0) {
            throw "LaTeX failed. See .build/main.log."
        }
    }
    Copy-Item -LiteralPath (Join-Path $buildPath 'main.pdf') -Destination (Join-Path $PSScriptRoot 'main.pdf') -Force
    Write-Output 'Built main.pdf'
}
finally {
    Pop-Location
}
