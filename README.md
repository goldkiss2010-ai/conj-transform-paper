# CONJ paper reproducibility package

This repository contains the LaTeX source of the paper and minimal scripts to reproduce the reported metrics.

## Repository structure

- `paper/`   : LaTeX source and compiled paper PDF
- `scripts/` : reproducibility scripts
- `assets/`  : input assets (see also `assets/ASSETS_LICENSE.md`)
- `results/` : bundled example metrics JSON
- `.github/workflows/` : CI smoke check workflow

## Requirements

- Python 3.10+ (recommended)
- NumPy
- Pillow (for the real-image experiment)

Install:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
````

## Reproducing experiments

### 1) Real-image experiment (JSON output)

Place the input image at:

* `assets/images/YKT_3336.jpg`

Run:

```bash
python scripts/vsLegacy_paper.py
```

Outputs:

* `output/YKT_3336_paper_metrics.json`

### 2) Uniform random samples experiment (stdout)

```bash
python scripts/conj_experiment.py
```

## License / Notices

* Code: Apache License 2.0 (see `LICENSE`)
* Paper text/figures: see `paper/`
* Photo asset(s): see `assets/ASSETS_LICENSE.md`

See `NOTICE` for disclaimers (no warranty, no FTO search, etc.).

## Related repositories

* CONJ core implementation: [https://github.com/goldkiss2010-ai/conj-core](https://github.com/goldkiss2010-ai/conj-core)