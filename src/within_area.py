import sys, json, time; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from itertools import combinations
from math import comb
from common import load, design, rf, CAT, NUM
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score

H = load()
BLOCKS = {
 'Demographics (sex, age)': ['egender','lage'],
 'Ethnicity': ['nethnicity'],
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
CATa = [c for c in CAT if c != 'cstudyarea']
out = {}
for area in ['Yokadouma', 'Bengbis', 'Mintom']:
    h = H[H['cstudyarea'] == area].reset_index(drop=True)
    y = (h['safe'] <= 1).astype(int)
    X = pd.concat([h[CATa].astype(str), h[NUM].apply(pd.to_numeric, errors='coerce')], axis=1)
    prep = ColumnTransformer([
      ('c', Pipeline([('i',SimpleImputer(strategy='most_frequent')),
                      ('o',OneHotEncoder(handle_unknown='ignore',sparse_output=False,min_frequency=10))]), CATa),
      ('n', Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]), NUM)])
    g = h['dlocality'].astype(str)
    Xd, pd_ = design(h, CATa, NUM)
    auc = cross_val_score(rf(pd_), Xd, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                          groups=g, scoring='roc_auc')
    pipe = Pipeline([('p',prep),('m',RandomForestClassifier(n_estimators=300,min_samples_leaf=5,
                      class_weight='balanced',random_state=0,n_jobs=-1))]).fit(X,y)
    f = lambda d: pipe.predict_proba(d)[:,1]
    names, groups = list(BLOCKS), list(BLOCKS.values()); n = len(groups)
    Ss = [S for r in range(n+1) for S in combinations(range(n), r)]
    rng = np.random.default_rng(0)
    bg = X.iloc[rng.choice(len(X), 30, replace=False)].reset_index(drop=True)
    idx = rng.choice(len(X), 30, replace=False)
    t0 = time.time(); P = []
    for i in idx:
        x = X.iloc[[i]]; blocks = []
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
        P.append(phi)
    P = np.vstack(P); ma = np.abs(P).mean(axis=0)
    r2 = np.random.default_rng(7); bs = np.vstack([np.abs(P[r2.integers(0,len(P),len(P))]).mean(axis=0) for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    order = np.argsort(-ma)
    print(f"\n=== {area}: n={len(h)}, prevalence {y.mean():.3f}, within-area village-grouped AUC {auc.mean():.3f} "
          f"(+/-{auc.std():.3f}), {len(Ss)} coalitions, {time.time()-t0:.0f}s ===")
    print(f"{'block':34s} {'mean |phi|':>10s} {'95% CI':>20s}  rank")
    rows = []
    for k, j in enumerate(order):
        rows.append({'block': names[j], 'mean_abs': round(float(ma[j]),4), 'ci':[round(float(lo[j]),4), round(float(hi[j]),4)], 'rank': k+1})
        print(f"{names[j]:34s} {ma[j]:10.4f} [{lo[j]:.4f}, {hi[j]:.4f}]  {k+1}")
    out[area] = {'n': int(len(h)), 'prev': round(float(y.mean()),3), 'auc': round(float(auc.mean()),3),
                 'auc_sd': round(float(auc.std()),3), 'blocks': rows}
json.dump(out, open('results/within_area.json','w'), indent=1)
print('\nsaved results/within_area.json')
