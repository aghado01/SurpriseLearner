# PowerShell 7+ Standards for Copilot Generation
# This file demonstrates the coding standards expected in this project

#Requires -Version 7.0

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ProjectPath,

    [Parameter()]
    [ValidateSet('Development', 'Production', 'Testing')]
    [string]$Environment = 'Development'
)

function Invoke-CopilotStandardsExample {
    <#
    .SYNOPSIS
        Example function demonstrating PowerShell 7+ standards for Copilot

    .DESCRIPTION
        This function serves as a template for all PowerShell code generated
        by Copilot, ensuring consistent syntax and best practices

    .PARAMETER InputObject
        The object to process with strict typing

    .EXAMPLE
        Invoke-CopilotStandardsExample -InputObject $myData
    #>

    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, ValueFromPipeline = $true)]
        [PSCustomObject]$InputObject
    )

    begin {
        Write-Information "Starting processing with environment: $Environment" -InformationAction Continue
        $processedCount = 0
    }

    process {
        try {
            # Demonstrate proper string interpolation
            $message = "Processing item: $($InputObject.Name) at path: $($InputObject.Path)"
            Write-Verbose $message

            # Demonstrate proper hashtable splatting
            $parameters = @{
                Path = $InputObject.Path
                Recurse = $true
                Force = $Environment -eq 'Production'
            }

            # Demonstrate call operator usage
            $results = & Get-ChildItem @parameters

            # Proper nested bracket structure
            if ($results) {
                foreach ($result in $results) {
                    if ($result.Extension -in @('.ps1', '.py', '.md')) {
                        Write-Output "Found valid file: $($result.FullName)"
                        $processedCount++
                    }
                }
            }

        }
        catch {
            Write-Error "Failed to process $($InputObject.Name): $($_.Exception.Message)"
            throw
        }
    }

    end {
        Write-Information "Processed $processedCount items successfully" -InformationAction Continue
    }
}
