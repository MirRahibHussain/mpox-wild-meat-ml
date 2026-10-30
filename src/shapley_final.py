import sys, json, time; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from itertools import combinations
from math import comb
from common import load, CAT, NUM
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

h = load(); y = (h['safe'] <= 1).astype(int)
X = pd.concat([h[CAT].astype(str), h[NUM].apply(pd.to_numeric, errors='coerce')], axis=1)
prep = ColumnTransformer([
  ('c', Pipeline([('i',SimpleImputer(strategy='most_frequent')),
                  ('o',OneHotEncoder(handle_unknown='ignore',sparse_output=False,min_frequency=20))]), CAT),
  ('n', Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]), NUM)])
pipe = Pipeline([('p',prep),('m',RandomForestClassifier(n_estimators=300,min_samples_leaf=5,
                  class_weight='balanced',random_state=0,n_jobs=-1))]).fit(X,y)
f = lambda d: pipe.predict_proba(d)[:,1]

PRIMARY = {
 'Demographics (sex, age)': ['egender','lage'],
 'Ethnicity': ['nethnicity'],
 'Location (study area)': ['cstudyarea'],
 'Occupation & job security': ['fprimaryoccupation','isecondaryoccupation','hprimaryoccupationstatus',
                               'gprimaryoccupationpaidorunpaid','hunter_PO','farmer'],
 'Household size & hunters': ['opeopleliveinyourhousehold','q8howmanypeople_huntinhh'],
 'Hunting participation & motive': ['q1ever_hunt','hunt_food','hunt_income','hunt_both'],
 'Species hunted': ['primates','rats','bathunt'],
 'Handling role': ['butcher_wildmeat','cook_wildmeat'],
 'Consumption frequency': ['q27frequently_eatwildmeat_rainys','q28frequently_eatwildmeat_drysea','q34eat_rawmeat'],
 'Risk perception & knowledge': ['concerned','feardis','q48know_illnesses_someonehassuff'],
 'Water, livestock, injuries': ['q51obtainwaterfordrinking','q44types_ownlivestock','q45havedogsathome',
                                'q16have_sufferedanyinjuries','inj','bite','claws','nails','treatment'],
}
# alternative grouping: split the largest block, merge the smallest ones -> tests block-size sensitivity
ALT = {
 'Demographics & household': ['egender','lage','opeopleliveinyourhousehold','q8howmanypeople_huntinhh'],
 'Ethnicity': ['nethnicity'],
 'Location (study area)': ['cstudyarea'],
 'Occupation & job security': ['fprimaryoccupation','isecondaryoccupation','hprimaryoccupationstatus',
                               'gprimaryoccupationpaidorunpaid','hunter_PO','farmer'],
 'Hunting & species': ['q1ever_hunt','hunt_food','hunt_income','hunt_both','primates','rats','bathunt'],
 'Handling role': ['butcher_wildmeat','cook_wildmeat'],
 'Consumption frequency': ['q27frequently_eatwildmeat_rainys','q28frequently_eatwildmeat_drysea','q34eat_rawmeat'],
 'Risk perception & knowledge': ['concerned','feardis','q48know_illnesses_someonehassuff'],
 'Injuries while hunting': ['q16have_sufferedanyinjuries','inj','bite','claws','nails','treatment'],
 'Water & livestock': ['q51obtainwaterfordrinking','q44types_ownlivestock','q45havedogsathome'],
}

def run(BLOCKS, idx, bg, tag):
    names, groups = list(BLOCKS), list(BLOCKS.values()); n = len(groups)
    Ss = [S for r in range(n+1) for S in combinations(range(n), r)]
    def one(x):
        blocks = []
        for S in Ss:
            Xs = bg.copy()
            for j in S:
                for c in groups[j]: Xs[c] = x[c].iloc[0]
            blocks.append(Xs)
        p = f(pd.concat(blocks, ignore_index=True)[X.columns]).reshape(len(Ss), len(bg)).mean(axis=1)
        v = dict(zip(Ss, p)); phi = np.zeros(n)
        for j in range(n):
            rest = [k for k in range(n) if k != j]
            for r in range(n):
                w = 1.0/(n*comb(n-1, r))
                for S in combinations(rest, r):
                    phi[j] += w*(v[tuple(sorted(S+(j,)))] - v[tuple(sorted(S))])
        return phi, v[()], v[tuple(range(n))]
    t0 = time.time(); P = []
    for i in idx:
        phi, b, full = one(X.iloc[[i]]); P.append(phi)
    P = np.vstack(P)
    print(f"\n=== {tag}: {len(idx)} respondents, {len(Ss)} coalitions, {time.time()-t0:.0f}s ===")
    mean_abs = np.abs(P).mean(axis=0)
    rng2 = np.random.default_rng(7); bs = []
    for _ in range(2000):
        s = rng2.integers(0, len(P), len(P)); bs.append(np.abs(P[s]).mean(axis=0))
    bs = np.vstack(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    order = np.argsort(-mean_abs); rows = []
    print(f"{'block':34s} {'cols':>4s} {'mean |phi|':>10s} {'95% CI':>18s} {'per column':>11s}")
    for j in order:
        rows.append({'block': names[j], 'cols': len(groups[j]), 'mean_abs': round(float(mean_abs[j]),4),
                     'ci': [round(float(lo[j]),4), round(float(hi[j]),4)],
                     'per_col': round(float(mean_abs[j]/len(groups[j])),4)})
        print(f"{names[j]:34s} {len(groups[j]):4d} {mean_abs[j]:10.4f} "
              f"[{lo[j]:.4f}, {hi[j]:.4f}] {mean_abs[j]/len(groups[j]):11.4f}")
    print(f"additivity: base {b:.4f} + sum(phi) {phi.sum():+.4f} = {b+phi.sum():.4f}  model {full:.4f}")
    return rows, P

rng = np.random.default_rng(0)
bg = X.iloc[rng.choice(len(X), 30, replace=False)].reset_index(drop=True)
idx = rng.choice(len(X), 40, replace=False)
primary_rows, P = run(PRIMARY, idx, bg, 'primary grouping (11 blocks)')
alt_rows, _ = run(ALT, idx, bg, 'alternative grouping (10 blocks)')
json.dump({'primary': primary_rows, 'alt': alt_rows,
           'phi': [[round(float(v),5) for v in r] for r in P],
           'names': list(PRIMARY)}, open('results/shapley_final.json','w'), indent=1)
print('\nsaved results/shapley_final.json')
