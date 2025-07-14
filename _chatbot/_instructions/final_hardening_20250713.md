<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Copilot Execution Instructions

## Immediate Task: Repository Validation and Final Hardening

### Primary Objective

Complete repository hardening validation and implement any missing critical components identified in the post-mortem analysis[^1].

### Execution Instructions

#### Phase 1: Validation and Gap Analysis (15 minutes max)

1. **Run comprehensive validation suite**

```bash
python _chatbot/_diagnostics/cicd_check.py
python _chatbot/_diagnostics/format_scanner.py
```

2. **Verify package import functionality**

```bash
python -c "import adaptive_bayesian_driver; print('✓ Package import successful')"
```

3. **Check pre-commit hook installation status**

```bash
pre-commit --version
git config --get core.hooksPath
```


#### Phase 2: Critical Gap Resolution (30 minutes max)

**If package import fails:**

- Create missing `adaptive_bayesian_driver/__init__.py` with complete content from attached specifications[^2]
- Ensure all module `__init__.py` files exist in subdirectories

**If pre-commit hooks not active:**

- Execute: `pre-commit install`
- Run: `pre-commit run --all-files` (allow up to 2 failures for initial run)

**If requirements inconsistencies found:**

- Update `requirements.txt` to match pyproject.toml dependencies exactly
- Verify CUDA installation notes are present and accurate


#### Phase 3: Final Integration Test (15 minutes max)

1. **Execute complete build verification**

```bash
pip install -e .
python -m pytest tests/ --tb=short -x
```

2. **Generate validation report**
    - Document any remaining issues in `_chatbot/_reports/validation_final.md`
    - Include pass/fail status for each critical component

### Chatbot Instruction File Updates

#### Update `_chatbot/Guidelines.md`

Add this section at the end:

```markdown
## Loop Detection and Efficiency Rules

### Automatic Termination Conditions
- Stop if the same file is edited more than 2 times in one session
- Escalate to human if any operation exceeds 15 minutes
- Abort if import errors persist after 3 fix attempts

### Progress Validation Requirements
- Test package import after any structural changes
- Run diagnostic scripts after configuration updates
- Verify git status shows expected changes before proceeding

### Communication Protocol
- Report completion percentage after each phase
- Surface critical errors immediately, not at session end
- Request human review only for genuine technical blockers
```


#### Update `_chatbot/README.md`

Replace the workflow section with:

```markdown
## Autonomous Execution Workflow

### Standard Operating Procedure
1. **Pre-flight Check**: Run diagnostics to establish baseline
2. **Targeted Execution**: Implement changes in logical dependency order
3. **Incremental Validation**: Test each change before proceeding
4. **Status Reporting**: Provide concise progress updates

### Human Intervention Triggers
- Package import failures after fixes
- Test suite failures exceeding 20% of total tests
- Git conflicts or repository state corruption
- Execution time exceeding planned window

### Success Criteria
- All diagnostic checks pass
- Package imports successfully
- Core functionality demonstrable
- Repository in clean git state
```


### Execution Parameters

- **Time Limit**: 60 minutes total
- **Approval Required**: Only for critical failures or scope expansion
- **Success Metric**: Complete diagnostic suite passes + package import works
- **Rollback Plan**: `git reset --hard HEAD` if critical failures occur


### Expected Deliverables

1. **Validation Report**: `_chatbot/_reports/final_validation_YYYY-MM-DD.md`
2. **Updated Package**: Working `adaptive_bayesian_driver` import
3. **Clean Repository**: All pre-commit hooks passing
4. **Documentation**: Any critical setup notes for future development

**Begin execution immediately upon confirmation.**

<div style="text-align: center">⁂</div>

[^1]: repo-cleanup-part1.txt

[^2]: repo-cleanup-part1_report.md

[^3]: Perplexity_Repo_Hardening_Complete-Update.md

[^4]: Perplexity_Repo_Hardening_Strategy_Reconciliation_with_Gemini.md

[^5]: Perplexity_Repo_Hardening_Guidelines.md

[^6]: copilot-thread-and-feedback-20250713.txt

[^7]: repo_current.json

