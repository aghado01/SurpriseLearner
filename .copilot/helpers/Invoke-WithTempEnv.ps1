<#
.SYNOPSIS
Executes a script block with temporary environment variables that are automatically cleaned up.

.DESCRIPTION
This function sets temporary environment variables, executes a script block, and then
restores the original environment state. It ensures that temporary environment changes
don't persist beyond the intended scope.

.PARAMETER EnvironmentVariables
Hashtable of environment variables to set temporarily. Keys are variable names, values are the values to set.

.PARAMETER ScriptBlock
The script block to execute with the temporary environment variables.

.EXAMPLE
Invoke-WithTempEnv -EnvironmentVariables @{
    'TEMP_CONFIG' = 'special-value'
    'DEBUG_MODE' = 'true'
} -ScriptBlock {
    Write-Host "Inside block: $env:TEMP_CONFIG"
    # Your code here
}

.EXAMPLE
Invoke-WithTempEnv -EnvironmentVariables @{
    'ChocolateyToolsLocation' = 'C:\Temp\ChocoExtract'
} -ScriptBlock {
    choco install notepadplusplus.commandline -y
}

.INPUTS
None

.OUTPUTS
Returns the result of the script block execution.

.NOTES
Environment variables are automatically restored even if the script block throws an exception.
#>
function Invoke-WithTempEnv {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$EnvironmentVariables,
        
        [Parameter(Mandatory = $true)]
        [scriptblock]$ScriptBlock
    )
    
    # Store original values
    $originalValues = @{}
    foreach ($key in $EnvironmentVariables.Keys) {
        $originalValues[$key] = [Environment]::GetEnvironmentVariable($key)
        Write-Verbose "Storing original value for $key`: $($originalValues[$key])"
    }
    
    try {
        # Set temporary environment variables
        foreach ($kvp in $EnvironmentVariables.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($kvp.Key, $kvp.Value, [EnvironmentVariableTarget]::Process)
            Write-Verbose "Set temporary env var: $($kvp.Key) = $($kvp.Value)"
        }
        
        # Execute the script block and return its result
        $result = & $ScriptBlock
        return $result
        
    } finally {
        # Restore original environment (guaranteed cleanup)
        foreach ($kvp in $originalValues.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($kvp.Key, $kvp.Value, [EnvironmentVariableTarget]::Process)
            Write-Verbose "Restored env var: $($kvp.Key) to original value"
        }
    }
}
