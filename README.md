# depresjon-clean

Code for the paper:

**Subject-Aware Detection of Depression from Wrist Actigraphy Using Per-Subject Distributional Aggregation and Interpretable Biomarker Analysis**
Yildiz, Subasi M.E., Karabulut, Subasi A. (2025, under review).

Headline result: 0.855 accuracy (47/55) on Depresjon under leave-one-subject-out cross-validation with nested model selection, sensitivity 0.783, specificity 0.906, Cohen's κ 0.697. Range over five seeds: 0.800-0.855.

The best single configuration (Information Gain + Bagging) reaches 0.927, but that is the maximum over a grid of 105 selector-classifier pairs, so it is not an estimate of generalization. The gap between the two is what post-hoc configuration selection is worth on 55 subjects.

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

Nested LOSO. Selector, classifier and k are all chosen by inner 5-fold CV on the
54 training subjects, from 840 candidate configurations. Takes about 12 minutes
of CPU per outer fold; use `--n-jobs` to parallelise across folds.

```bash
python nested_loso.py --data-dir data/depresjon --cache-dir cache \
    --feature-set tsfel_only --cohort full --seed 0 --n-jobs 11
```

Grid maximum (the 0.927 figure, kept for comparison with the earlier literature):

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

## Other analyses in the paper

```bash
# recording-length confound: min/max ablation, overlap cohort, 12-day truncation
python confound_checks.py  --data-dir data/depresjon
python confound_checks2.py
python daycount_confound.py
python daycount_baseline.py --data-dir data/depresjon

# nested runs on the confound-controlled cohorts and extra seeds
python nested_loso.py --cohort overlap_13_28 --seed 0 --data-dir data/depresjon --cache-dir cache
python nested_loso.py --cohort first12       --seed 0 --data-dir data/depresjon --cache-dir cache
python nested_loso.py --cohort no_nonwear    --seed 0 --data-dir data/depresjon --cache-dir cache

# subgroup sensitivity, feature collinearity, ZCR definition, non-wear structure
python reviewer2_analyses.py
python headline_errors.py

# figures and tables
python make_figures.py
python make_tsfel_table.py
python fix_table5.py
```

## Reproducibility

One seed, 0, is used for fold shuffling, the selectors and the classifier
constructors. Per-day TSFEL features are cached under `cache/` and the LOSO fold
structure is deterministic, so a rerun on the same machine reproduces the
numbers exactly. Across seeds the nested estimate moves between 44 and 47
correctly classified subjects.

Pinned versions (see `requirements.txt`): TSFEL 0.2.0, scikit-learn 1.3.2,
XGBoost 2.1.4, NumPy 1.24.4, SciPy 1.10.1, PyWavelets 1.4.1.

`results/` holds the archived summaries the paper cites. Everything else written
there by a rerun is gitignored.
