---
applyTo: "{**/*.ps1,**/*.py}"
---

# Memory Orchestration Architecture Instructions

## Core Architecture Overview
This project implements a sophisticated memory orchestration system for AI-assisted development with four-tier memory management:
1. **Session Memory**: Current terminal session context
2. **Sequence Memory**: Post-instruction execution logging
3. **Project Memory**: Cross-session continuity
4. **Global Memory**: Pattern learning across projects

## Memory Orchestration Integration Patterns

### For PowerShell Scripts
Memory-aware script initialization
param(
[string]$SessionId = (New-Guid).ToString(),
[string]$MemoryContext = "Development",
[switch]$PersistSession
)

Session boundary detection
function Test-SessionContinuity {
[CmdletBinding()]
param(
[Parameter(Mandatory = $true)]
[string]$WorkspaceType
)

text
$sessionMarkers = @{
    GitRepository = Test-Path ".git"
    CopilotWorkspace = Test-Path ".copilot"
    ProjectMemory = Test-Path "memory/project.json"
}

return $sessionMarkers[$WorkspaceType]
}

Console-aware capture integration
function Write-MemoryLog {
param(
[Parameter(Mandatory = $true)]
[string]$Message,
[ValidateSet('Session', 'Sequence', 'Project', 'Global')]
[string]$Scope = 'Session',
[hashtable]$Context = @{}
)

text
$logEntry = @{
    Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Scope = $Scope
    Message = $Message
    Context = $Context
    SessionId = $SessionId
}

# Hierarchical logging integration
$logPath = switch ($Scope) {
    'Session' { "memory/session/$SessionId.log" }
    'Sequence' { "memory/sequence/$(Get-Date -Format 'yyyyMMdd').log" }
    'Project' { "memory/project/activity.log" }
    'Global' { "$env:USERPROFILE/.copilot/global/patterns.log" }
}

$logEntry | ConvertTo-Json -Compress | Out-File -FilePath $logPath -Append -Encoding UTF8
}

text

### For Python Components
from typing import Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime

class MemoryOrchestrator:
"""
Central intelligence hub for memory orchestration across development sessions.

text
Manages context flow between human developer, AI assistants, and development environment.
Implements cadence-based memory injection and rolling context windows.
"""

def __init__(self, session_id: str, workspace_type: str = "development"):
    self.session_id = session_id
    self.workspace_type = workspace_type
    self.memory_scopes = {
        'session': Path(f"memory/session/{session_id}"),
        'sequence': Path("memory/sequence"),
        'project': Path("memory/project"),
        'global': Path.home() / ".copilot" / "global"
    }
    self._ensure_memory_structure()

def inject_context(self, scope: str, context_data: Dict[str, Any]) -> None:
    """Inject context data into specified memory scope."""
    memory_path = self.memory_scopes[scope] / f"{datetime.now().isoformat()}.json"

    memory_entry = {
        'timestamp': datetime.now().isoformat(),
        'scope': scope,
        'session_id': self.session_id,
        'workspace_type': self.workspace_type,
        'context': context_data
    }

    with open(memory_path, 'w', encoding='utf-8') as f:
        json.dump(memory_entry, f, indent=2)

def synthesize_context(self, query_scope: str, max_entries: int = 50) -> Dict[str, Any]:
    """Synthesize relevant context from memory hierarchy."""
    context_synthesis = {
        'session_context': self._load_recent_context('session', max_entries),
        'project_patterns': self._load_recent_context('project', 20),
        'global_insights': self._load_recent_context('global', 10)
    }

    return self._filter_relevant_context(context_synthesis, query_scope)
text

## Development Workflow Integration Requirements

### Session Boundary Management
- **Automatic Detection**: Scripts must detect workspace transitions (Git repos, AI workspaces, ephemeral sessions)
- **State Persistence**: Critical state must survive session boundaries through JSON serialization
- **Context Migration**: Support context transfer between development environments

### Console-Aware Capture
- **Command Monitoring**: Register hooks for development commands (test, build, debug)
- **Output Interception**: Capture and filter console output for AI consumption
- **Bidirectional Communication**: Enable AI feedback integration with terminal workflows

### Adaptive Assistance Coordination
- **Context-Aware Triggers**: Map development phases to appropriate AI agent allocation
- **Dynamic Handoff**: Support seamless agent transitions based on workflow context
- **Learning Integration**: Incorporate pattern recognition from the adaptive Bayesian learning system

## Cross-Project Pattern Learning
All components should contribute to global pattern learning:
- Log recurring development patterns with structured metadata
- Support pattern similarity scoring for context transfer
- Enable recommendation generation based on historical patterns
- Maintain compatibility with the RecursiveBayesianLearner for surprise detection in development workflows

## Production-Grade Reliability
- Implement graceful degradation when memory systems are unavailable
- Use atomic file operations for memory persistence
- Include comprehensive error recovery for corrupted memory states
- Support memory cleanup and archival for long-running projects
