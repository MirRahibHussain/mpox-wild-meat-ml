# Machine learning assessment of mpox-relevant food safety risk in the wild meat value chain

Analysis code for the paper of the same title.

## Data

The analysis uses a public dataset that is **not redistributed here**:

> Ickowitz, A. (2026). *Mitigating risks of disease transmission in the wild meat food chain
> from forest to fork in Cameroon* [Data set]. Center for International Forestry Research.
> https://doi.org/10.17528/CIFOR/DATA.00328 (CC BY 4.0)

Download `hunter_food_preparer_Dataverse.tab` from that record and place it in `data/`.

## Reproducing the reported figures

```
pip install -r requirements.txt
python src/inference.py        # Table 3: discrimination, bootstrap CIs, calibration, thresholds
python src/robustness.py       # Table 4: outcome-definition robustness
python src/fairness.py         # Table 5: subgroup audit with bootstrap CIs
python src/shapley_final.py    # Figures 3-4: grouped Shapley, CIs, alternative partition
python src/risk_index.py       # Table 6, Figures 5-6: scorecard and calibration bands
```

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

No values were simulated and no oversampling was applied. Missing values are imputed by
median and mode within training folds only.

## Licence

Code: MIT (see LICENSE). The dataset keeps its own CC BY 4.0 licence.
