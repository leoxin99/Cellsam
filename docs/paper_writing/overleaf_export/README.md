# overleaf_export

This directory is the active LaTeX workspace for thesis writing and Overleaf upload.

## Legacy export command (optional)

```powershell
powershell -ExecutionPolicy Bypass -File tools/export_md_to_latex.ps1
```

## Optional (legacy copy support files)

```powershell
powershell -ExecutionPolicy Bypass -File tools/export_md_to_latex.ps1 -CopySupportFiles
```

## Notes

1. Source of truth for active drafting is this directory: `overleaf_export/main.tex` and `overleaf_export/chapters/*.tex`.
2. `main.tex` is the recommended Overleaf entry file.
3. Markdown export remains available only for historical migration/recovery, not for daily writing.
4. Upload this folder as-is and compile with `main.tex`.
