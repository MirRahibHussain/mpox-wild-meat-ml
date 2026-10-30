# Mpox-relevant food safety risk in the wild meat value chain

Analysis code for two companion papers that share one dataset and one analytic sample:

**Paper 1 — "Area-specific risk factors for unsafe wild meat handling: a machine learning
analysis of mpox-relevant food safety practice."** Classification of safe-handling practice,
grouped Shapley attribution, fairness audit, a learned risk index, and a within-area analysis
showing that the pooled attribution ranking does not hold in any single area.

**Paper 2 — "Exposure to monkeypox virus along the wild meat chain: a quantitative exposure
assessment and the limits of risk prediction."** A Codex-framework exposure assessment:
annual contact-event counts for four pathways, prediction of exposure intensity, and a
leave-one-area-out transferability test.

## Data

The analysis uses a public dataset that is **not redistributed here**:

> Ickowitz, A. (2026). *Mitigating risks of disease transmission in the wild meat food chain
> from forest to fork in Cameroon* [Data set]. Center for International Forestry Research.
> https://doi.org/10.17528/CIFOR/DATA.00328 (CC BY 4.0)

Download `hunter_food_preparer_Dataverse.tab` from that record and place it in `data/`.

## Reproducing the reported numbers

Run every script from the repository root.

```
pip install -r requirements.txt
```

### Paper 1

```
python src/inference.py        # discrimination, bootstrap CIs, calibration, thresholds
python src/robustness.py       # outcome-definition robustness
python src/fairness.py         # subgroup audit with bootstrap CIs
python src/shapley_final.py    # grouped Shapley, CIs, alternative partition
python src/risk_index.py       # scorecard and calibration bands
python src/reviewer_fixes.py   # leave-one-area-out, card pruning, excluded-group description
python src/within_area.py      # per-area Shapley attribution
```

### Paper 2

```
python src/exposure_model.py   # exposure profile, sensitivity, prediction, transferability
python src/figures_paper2.py   # Figures 1-3 (writes SVG + PNG to figures/)
```

`src/exposure_model.py` writes `results/exposure_model.json`, which contains every number
reported in Tables 3-7 of paper 2: the frequency and blood-contact mappings, the exposure
profile overall and by area, the three sensitivity variants, cross-validated performance for
three outcomes and three model classes with bootstrap confidence intervals, and the
leave-one-area-out results. `src/figures_paper2.py` reads that file, so run it second.
Rendering the PNGs needs `wkhtmltoimage` on the path; the SVGs are written regardless.

Every script imports the analytic sample from `src/common.py`, which applies the single
restriction used throughout: the four outcome items are defined only for respondents who
handle wild meat carcasses, so the sample is the 2,262 handlers in 56 villages.

## Notes on the method

`src/shapley.py` implements Shapley values by complete enumeration of coalitions, with all
coalitions stacked into one prediction call. `src/shapley_final.py` groups the predictors into
thematic blocks, so the coalition space is 2^11 and can be enumerated exactly, and uses an
interventional value function over a background sample. TreeSHAP is already exact for tree
ensembles under a conditional value function; what differs here is the unit of attribution
(blocks, not columns) and the value function, not the exactness.

For paper 2, the exposure metrics in `src/exposure_model.py` are built from reported
frequencies only; the frequency-to-event mapping (`FREQ`), the treatment of the undocumented
"Every months" label (`AMB`) and the blood-contact proportions (`BLOOD_MAIN`, with `BLOOD_LOW`
and `BLOOD_HIGH` as the sensitivity variants) are declared at the top of the file so each
convention can be inspected and changed. Every variable that feeds an exposure metric is
listed in `LEAK` and removed from the predictor set, so the prediction results are not
circular. Cross-validation is grouped by village (`dlocality`); the leave-one-area-out test
additionally drops `cstudyarea` from the predictors.

No values were simulated and no oversampling was applied. Missing values are imputed by
median and mode within training folds only.

## Licence

Code: MIT (see LICENSE). The dataset keeps its own CC BY 4.0 licence.
