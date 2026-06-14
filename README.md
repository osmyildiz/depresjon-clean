# Subject-Aware Detection of Depression from Wrist Actigraphy

Reproducible pipeline for subject-level depression detection on the Depresjon
benchmark, using per-subject distributional aggregation of TSFEL features,
strict leave-one-subject-out (LOSO) cross-validation, Wilson 95% confidence
intervals, and Holm-corrected exact McNemar tests.

Accompanying code for the manuscript:

> Yildiz, O., Subasi, M. E., Karabulut, M., & Subasi, A. (2025).
> *Subject-Aware Detection of Depression from Wrist Actigraphy Using
> Per-Subject Distributional Aggregation and Interpretable Biomarker
> Analysis.* (Under review.)

---

## Headline result

Under strict LOSO across the 55 subjects of the Depresjon dataset
(23 depressed, 32 controls):

| Metric                            | Value                  |
| --------------------------------- | ---------------------- |
| Accuracy                          | **0.927** (51/55)      |
| Wilson 95% CI                     | [0.827, 0.971]         |
| Cohen's κ                         | 0.851                  |
| Sensitivity (recall on depressed) | 0.913 (21/23)          |
| Specificity (recall on control)   | 0.938 (30/32)          |
| Median features retained per fold | 10                     |

Best configuration: per-subject distributional aggregation of 156 TSFEL
features into 1092 distributional descriptors (7 order statistics × 156),
Information Gain feature selection (`k = 10`), Bagging classifier
(scikit-learn defaults), strict LOSO.

An ablation augmenting the feature set with 16 circadian biomarkers
(Interdaily Stability, Intradaily Variability, Relative Amplitude, cosinor
parameters, bout statistics) shifts accuracy to 0.909 (50/55). The
augmentation is reported as an interpretability analysis (four circadian
biomarkers enter the top selected features) rather than as a default
component of the pipeline. See Section 4.3 of the manuscript.

---

## Repository structure

```
depresjon-clean/
├── data/
│   └── depresjon/                # Place the Depresjon dataset here
│       ├── control/              # control_*.csv files
│       └── condition/            # condition_*.csv files
├── src/
│   ├── data.py                   # Subject-aware data loader
│   ├── features.py               # TSFEL extraction + unstable-feature drop
│   ├── aggregation.py            # Per-subject distributional aggregation
│   ├── circadian.py              # IS, IV, RA, cosinor, bout features
│   ├── selectors.py              # 7 feature selectors
│   ├── experiment.py             # Setting A (day-level grouped CV)
│   └── experiment_subject.py     # Setting B (subject-level LOSO)
├── scripts/
│   └── submit.sbatch             # SLURM submission template (DGX cluster)
├── analysis_regime_b.py          # Wilson CI + exact McNemar (Holm) post-hoc
├── run_subject.py                # Setting B entry point (15 classifiers × 7 selectors)
├── run.py                        # Setting A entry point (10-fold × 3 seeds)
├── confusion_best.py             # Confusion matrix: Info Gain + Bagging (TSFEL+circadian)
├── tsfel_only_confusion.py       # Confusion matrix: Info Gain + Bagging (TSFEL-only, headline)
├── diagnose_circadian.py         # Stand-alone diagnostic for circadian feature quality
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Requires Python 3.10 or later. We recommend a fresh virtual environment.

```bash
git clone https://github.com/osmyildiz/depresjon-clean.git
cd depresjon-clean
python3 -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .venv\Scripts\activate          # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Tested on:

- macOS 14 (Apple Silicon, M3 Max), Python 3.11, scikit-learn 1.5, XGBoost 2.0
- Linux (DGX A100), Python 3.10, SLURM job submission via `scripts/submit.sbatch`

No GPU is required. The full 55-fold LOSO benchmark across 15 classifiers and
7 selectors completes in under one hour on a single modern CPU once TSFEL
features are cached (see "Reproducing the results" below).

---

## Data

The Depresjon dataset (Garcia-Ceja et al., 2018) is openly available:

- DOI: [10.1145/3204949.3208125](https://doi.org/10.1145/3204949.3208125)
- Hosted at: https://datasets.simula.no/depresjon/

After downloading, place the `control/` and `condition/` directories under
`data/depresjon/` so that the layout matches the structure shown above. No
preprocessing or relabelling is required on top of the original release; the
loader (`src/data.py`) selects only complete recording days (1440 valid
minutes) and produces a subject-aware index used by every downstream stage.

---

## Reproducing the headline result

The headline accuracy of **0.927** is produced by the TSFEL-only
configuration of Information Gain + Bagging under strict LOSO. After
installing dependencies and placing the dataset, run:

```bash
python tsfel_only_confusion.py --data-dir data/depresjon --cache-dir cache
```

The first run extracts TSFEL features per day and caches them under
`cache/` (a few minutes); subsequent runs reuse the cache. Expected output:

```
accuracy   = 0.9273  (51/55)
kappa      = 0.8505
TP=21  TN=30  FP=2  FN=2
sensitivity = 0.913
specificity = 0.938
median_k   = 10
feature_dim= 1092
```

These match the numbers reported in Section 4.1 of the manuscript and in the
confusion matrix in Table 2's caption.

To obtain the corresponding result for the circadian-augmented setting
(Section 4.3 ablation, 0.909 accuracy), run:

```bash
python confusion_best.py --data-dir data/depresjon --cache-dir cache
```

---

## Reproducing all benchmarks and statistical analysis

To reproduce the full Setting B grid (7 selectors × 15 classifiers × 55 LOSO
folds, including all results that populate Tables 2, 3, 5):

```bash
python run_subject.py --data-dir data/depresjon --cache-dir cache --out-dir results/full_subject_level
```

Outputs are written to `results/full_subject_level/`:

- `summary.csv` — per-configuration accuracy, κ, F1, etc.
- `per_fold_correctness.csv` — boolean correctness for every (configuration, subject) pair
- `selected_features.csv` — features retained in each LOSO fold

The statistical post-hoc analysis (Wilson 95% CI for every configuration,
exact paired McNemar tests with Holm correction across all classifier pairs
within each selector) is then computed by:

```bash
python analysis_regime_b.py --results-dir results/full_subject_level
```

This produces `mcnemar_classifiers_per_selector.csv`, which contains the
adjusted p-values reported in Table 3 of the manuscript.

---

## Optional: Setting A (day-level grouped cross-validation)

Setting A is a methodological control reported in the manuscript Discussion
only; it uses day-level samples with a `StratifiedGroupKFold` (10 folds × 3
seeds) that preserves subject disjointness across the outer splits. To run:

```bash
python run.py --data-dir data/depresjon --cache-dir cache --out-dir results/setting_a
```

This produces an analogous `summary.csv` for day-level evaluation.

---

## Citation

If you use this code, please cite the accompanying manuscript:

```bibtex
@article{yildiz2025subjectaware,
  author  = {Yildiz, Osman and Subasi, Muhammed Enes and Karabulut, Mustafa and Subasi, Abdulhamit},
  title   = {Subject-Aware Detection of Depression from Wrist Actigraphy Using Per-Subject Distributional Aggregation and Interpretable Biomarker Analysis},
  year    = {2025},
  note    = {Under review}
}
```

If you use the Depresjon dataset, please also cite the original release:

```bibtex
@inproceedings{garciaceja2018depresjon,
  author    = {Garcia-Ceja, Enrique and Riegler, Michael and Jakobsen, Petter and T{\o}rresen, Jim and Nordgreen, Tine and Oedegaard, Ketil J. and Fasmer, Ole Bernt},
  title     = {Depresjon: A Motor Activity Database of Depression Episodes in Unipolar and Bipolar Patients},
  booktitle = {Proceedings of the 9th ACM Multimedia Systems Conference},
  year      = {2018},
  pages     = {472--477},
  doi       = {10.1145/3204949.3208125}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

The Depresjon dataset is distributed under its own terms by Simula Research
Laboratory; review the [dataset license](https://datasets.simula.no/depresjon/)
before any clinical or commercial use.

---

## Contact

For questions about the code or the manuscript, open an issue on this
repository or contact the corresponding author:

**Osman Yildiz** — `oyildiz@albany.edu`
Information Sciences & Technology Dept., College of Emergency Preparedness,
Homeland Security & Cybersecurity, University at Albany, SUNY, NY, USA
