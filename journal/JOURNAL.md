# ML Experimentation Journal

## Status

**Workspace decisions**
- Package: `wing_corrosion` — recorded: 2026-06-11
- Tabular library: `pandas` — recorded: 2026-06-11
- Environment manager: `uv` — recorded: 2026-06-11
- Skore mode: `hub` — recorded: 2026-06-11
- Skore hub workspace: `debray.yann` — recorded: 2026-06-11
- Optional features: none — recorded: 2026-06-11

**Current focus**
- Baseline experiment: predict corrosion risk from environmental features

## Experiments

| ID | Short name | Status | Brier score (CV) | Notes |
|----|------------|--------|------------------|-------|
| 01 | baseline   | planned | — | First baseline with all environmental features |

## Backlog

| ID | Item | Source | Priority |
|----|------|--------|----------|
| — | — | — | — |

## History

### 2026-06-11
- Initialized workspace with uv
- Created package structure: `src/wing_corrosion/`
- Installed Tier 1 dependencies: scikit-learn, skrub, skore[hub,jupyter], pandas, pyarrow
- Set up Skore Hub integration with workspace `debray.yann`