# depresjon-clean

Code for the paper:

**Subject-Aware Detection of Depression from Wrist Actigraphy Using Per-Subject Distributional Aggregation and Interpretable Biomarker Analysis**
Yildiz, Subasi M.E., Karabulut, Subasi A. (2025, under review).

Headline result: 0.927 accuracy (51/55) on Depresjon under leave-one-subject-out cross-validation, sensitivity 0.913, specificity 0.938, Cohen's κ 0.851.

## Setup

```bash
git clone https://github.com/osmyildiz/depresjon-clean.git
cd depresjon-clean
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Depresjon dataset from https://datasets.simula.no/depresjon/ and put it under `data/depresjon/`, so that `data/depresjon/control/` and `data/depresjon/condition/` exist.

## Reproduce the main result

```bash
python tsfel_only_confusion.py --data-dir data/depresjon --cache-dir cache
```

First run extracts and caches the TSFEL features (a few minutes); subsequent runs are fast. Output ends with:

```
accuracy   = 0.9273  (51/55)
kappa      = 0.8505
sensitivity = 0.913
specificity = 0.938
```

For the circadian-augmented variant (0.909 accuracy, ablation in the paper):

```bash
python confusion_best.py --data-dir data/depresjon --cache-dir cache
```

## Full benchmark

15 classifiers × 7 selectors × 55 LOSO folds, plus the statistical post-hoc:

```bash
python run_subject.py --data-dir data/depresjon --cache-dir cache --out-dir results/full
python analysis_regime_b.py --results-dir results/full
```

McNemar p-values (Table 3 in the paper) end up in `results/full/mcnemar_classifiers_per_selector.csv`.

## Layout

- `src/` — pipeline modules: data loader, TSFEL extraction, per-subject aggregation, circadian biomarkers, selectors
- `run_subject.py`, `run.py` — entry points for subject-level (Setting B) and day-level (Setting A) experiments
- `analysis_regime_b.py` — Wilson CI + Holm-corrected exact McNemar
- `confusion_best.py`, `tsfel_only_confusion.py` — confusion matrices for the two top configurations
- `scripts/submit.sbatch` — SLURM template for cluster runs

## Citation

```bibtex
@article{yildiz2025subjectaware,
  author = {Yildiz, Osman and Subasi, Muhammed Enes and Karabulut, Mustafa and Subasi, Abdulhamit},
  title  = {Subject-Aware Detection of Depression from Wrist Actigraphy Using Per-Subject Distributional Aggregation and Interpretable Biomarker Analysis},
  year   = {2025},
  note   = {Under review}
}
```

If you use the Depresjon dataset itself, also cite Garcia-Ceja et al. 2018 (doi:10.1145/3204949.3208125).

## License

MIT. Contact: Osman Yildiz, oyildiz@albany.edu.
