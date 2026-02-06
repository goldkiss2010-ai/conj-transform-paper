# CONJ Preprint – Reproducibility Repository

This repository provides **reproducibility materials** for the CONJ preprint paper.
It contains scripts and assets required to reproduce the numerical experiments
reported in the manuscript.

本リポジトリは、CONJ プレプリント論文に記載された
**数値実験の再現専用リポジトリ**です。

---

## Paper / 論文

- **Title**: CONJ: Dual-Pair-Based 1+2 Decomposition with Residual Invariance  
- **Author**: Bungaku Yokota (横田ブンガク)  
- **Status**: Self-published preprint (not peer-reviewed)  
- **License (paper)**: CC BY 4.0  

The paper PDF is included in this repository.
See the preprint notice inside the PDF for license and disclaimer details.

---

## Purpose of This Repository

- Reproduce the numerical results reported in the paper
- Provide transparent reference implementations for verification
- Serve as a **paper-linked, minimal, reproducibility-focused** archive

This repository is **not** intended to be a general-purpose library.

---

## Repository Structure

```

.
├─ assets/
│  └─ images/
│     └─ YKT_3336.jpg          # Input image (author-photographed, consent obtained)
├─ scripts/
│  ├─ conj_experiment.py       # Uniform random sample experiments (Section 4)
│  └─ vsLegacy_paper.py        # Real-image experiment (Section 5)
├─ results/
│  └─ YKT_3336_paper_metrics.json
├─ paper/
│  └─ CONJ_preprint.pdf
├─ README.md
└─ LICENSE

````

---

## How to Reproduce the Experiments

### Requirements

- Python 3.x
- NumPy
- Pillow (for image I/O)

Example:
```bash
pip install numpy pillow
````

---

### Real Image Experiment (Section 5)

```bash
python scripts/vsLegacy_paper.py
```

* Input image: `assets/images/YKT_3336.jpg`
* Output metrics:
  `results/YKT_3336_paper_metrics.json`

The JSON file contains:

* Crosstalk metric (`C_img`)
* Chroma covariance change (`DeltaSigma_img`)
  for both **Legacy** and **CONJ** pipelines, as reported in the paper.

---

### Uniform Random Sample Experiment (Section 4)

```bash
python scripts/conj_experiment.py
```

Metrics are printed to standard output.
These correspond to Table 1 in the paper.

---

## Output Format (JSON)

The real-image experiment outputs a JSON file with the following keys:

```json
{
  "input": "assets/images/YKT_3336.jpg",
  "alpha": 0.8,
  "C_legacy": 6.96e-4,
  "C_conj": 8.29e-33,
  "DeltaSigma_legacy_Fro": 3.52e-3,
  "DeltaSigma_conj_Fro": 9.45e-18
}
```

---

## License

* **Paper (PDF)**:
  Creative Commons Attribution 4.0 International (CC BY 4.0)

* **Code**:
  Apache License, Version 2.0 (Apache-2.0)

See `LICENSE` for details.

---

## Disclaimer

* This repository is provided for research and reproducibility purposes only.
* No warranties are provided.
* **No FTO (freedom-to-operate) search is provided.**
* Nothing here constitutes legal advice.

---

## Versioning

* Paper version: v0.9
* Repository tag: **v0.1-smoke-ok**

The paper explicitly references this repository tag for reproducibility.