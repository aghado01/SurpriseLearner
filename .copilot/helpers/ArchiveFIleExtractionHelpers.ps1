function Expand-ZipSafe {
    param($ZipUrl, $Dest)
    $tmp = New-TemporaryFile
    Write-Host "↓  $ZipUrl" -ForegroundColor Cyan
    Invoke-WebRequest $ZipUrl -OutFile $tmp -UseBasicParsing
    Expand-Archive -LiteralPath $tmp -DestinationPath $Dest -Force
    Remove-Item $tmp -Force
}

function Expand-7zSelfExtractor {
    param($ExeUrl, $Dest)
    $tmp = New-TemporaryFile -Suffix '.exe'
    Write-Host "↓  $ExeUrl" -ForegroundColor Cyan
    Invoke-WebRequest $ExeUrl -OutFile $tmp -UseBasicParsing
    & $tmp -o"$Dest" -y  | Out-Null
    Remove-Item $tmp -Force
}

# add nuget filetype helpers
# add nuget cleanup
