# Archive Plan

> **Updated**: 2026-02-10  
> **Execution order**: A0 → A → B  
> **Principle**: No files deleted. Move + rename + DEPRECATED header only.

---

## Directory Structure

```
CellSam/
├── archive/
│   └── root_scripts/              # Phase A: deprecated root .py files
├── anti_test/
│   └── archive/
│       └── deprecated_py/         # Phase A0: deprecated anti_test .py files
├── tools/
│   └── archive/
│       ├── tests_deprecated/      # Phase A: deprecated test scripts
│       ├── legacy_eval/           # Phase A: legacy eval scripts
│       ├── legacy_experiment/     # Phase B: one-off experiments
│       ├── legacy_visualization/  # Phase B: legacy viz scripts
│       └── legacy_compare/        # Phase B: legacy compare scripts
└── scripts/
    └── archive/                   # Future: legacy training shell scripts
```

## Naming Convention

- All archived files: `deprecated_<original_name>.py`
- All files include a DEPRECATED header block with:
  - Archive date
  - Archive reason
  - Replacement entry point(s)
  - DeprecationWarning import

## Rules

1. **No new .py files in root directory** (training/eval/test scripts)
2. **No duplicate entry points** with the same functionality
3. **No legacy scripts without DEPRECATED marking** at same level as active scripts
