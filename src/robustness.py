import sys; sys.path.insert(0,'src')
import pandas as pd, numpy as np
from common import load, design, rf, OUT
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

h = load(); g = h['dlocality'].astype(str)
X, prep = design(h)
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)

print("=== A. each safe-practice item modelled separately ===")
print(f"{'outcome':24s} {'prevalence':>10s} {'AUC':>12s}")
for item in OUT:
    yi = 1 - pd.to_numeric(h[item], errors='coerce').fillna(0)      # 1 = unsafe
    auc = cross_val_score(rf(prep), X, yi, cv=cv, groups=g, scoring='roc_auc')
    print(f"{item:24s} {yi.mean():10.3f} {auc.mean():8.3f} (+/-{auc.std():.3f})")

print("\n=== B. alternative composite thresholds ===")
print(f"{'definition of high risk':30s} {'prevalence':>10s} {'AUC':>12s}")
for thr, lab in [(0,'0 safe practices'),(1,'<=1 safe practice'),(2,'<=2 safe practices')]:
    yt = (h['safe'] <= thr).astype(int)
    auc = cross_val_score(rf(prep), X, yt, cv=cv, groups=g, scoring='roc_auc')
    print(f"{lab:30s} {yt.mean():10.3f} {auc.mean():8.3f} (+/-{auc.std():.3f})")

print("\n=== C. continuous safe-score (regression, Spearman) ===")
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from scipy.stats import spearmanr
reg = Pipeline([('p',prep),('m',RandomForestRegressor(n_estimators=600,min_samples_leaf=5,
                random_state=0,n_jobs=-1))])
pred = cross_val_predict(reg, X, h['safe'].fillna(h['safe'].median()),
                         cv=GroupKFold(5), groups=g)
r,p = spearmanr(pred, h['safe'].fillna(h['safe'].median()))
print(f"Spearman rho = {r:.3f}  (p = {p:.1e})")

print("\n=== D. random-seed stability (composite <=1) ===")
y = (h['safe'] <= 1).astype(int)
aucs=[]
for seed in range(5):
    cvs = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    aucs.append(cross_val_score(rf(prep), X, y, cv=cvs, groups=g, scoring='roc_auc').mean())
print(f"AUC across 5 seeds: {np.mean(aucs):.3f}  range {min(aucs):.3f}-{max(aucs):.3f}")
