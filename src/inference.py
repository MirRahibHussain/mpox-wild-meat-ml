import sys, json; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from common import load, design, rf, lr
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, average_precision_score

h = load(); g = h['dlocality'].astype(str).values; y = (h['safe'] <= 1).astype(int).values
X, prep = design(h); cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
models = {'Random forest': rf(prep), 'Logistic regression': lr(prep),
          'Gradient boosting': Pipeline([('p', prep), ('m', HistGradientBoostingClassifier(random_state=0))])}
P, res = {}, {}
print(f"n = {len(y)}, villages = {len(set(g))}, prevalence = {y.mean():.3f}\n")
for name, m in models.items():
    P[name] = cross_val_predict(m, X, y, cv=cv, groups=g, method='predict_proba')[:, 1]
    folds = cross_val_score(m, X, y, cv=cv, groups=g, scoring='roc_auc')
    res[name] = {'fold_mean': round(float(folds.mean()), 3), 'fold_sd': round(float(folds.std()), 3)}

# cluster bootstrap over villages
vill = np.array(sorted(set(g))); idx_by = {v: np.where(g == v)[0] for v in vill}
rng = np.random.default_rng(0); B = 1000
boot = {k: [] for k in models}; diff = {'rf-lr': [], 'rf-gb': []}
for _ in range(B):
    pick = rng.choice(vill, len(vill), replace=True)
    rows = np.concatenate([idx_by[v] for v in pick])
    if len(set(y[rows])) < 2: continue
    a = {k: roc_auc_score(y[rows], P[k][rows]) for k in models}
    for k in models: boot[k].append(a[k])
    diff['rf-lr'].append(a['Random forest'] - a['Logistic regression'])
    diff['rf-gb'].append(a['Random forest'] - a['Gradient boosting'])
print("=== discrimination, village-clustered bootstrap (B = 1000) ===")
print(f"{'model':22s} {'fold mean (SD)':>16s} {'pooled AUC':>11s} {'95% CI':>18s} {'PR-AUC':>8s} {'Brier':>7s}")
for k in models:
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    res[k].update({'pooled': round(float(roc_auc_score(y, P[k])), 3), 'ci': [round(float(lo),3), round(float(hi),3)],
                   'ap': round(float(average_precision_score(y, P[k])), 3),
                   'brier': round(float(brier_score_loss(y, P[k])), 3)})
    r = res[k]
    print(f"{k:22s} {r['fold_mean']:.3f} ({r['fold_sd']:.3f}){'':>4s} {r['pooled']:11.3f} {str(r['ci']):>18s} {r['ap']:8.3f} {r['brier']:7.3f}")
print("\n=== paired differences in pooled AUC (same bootstrap resamples) ===")
for k, d in diff.items():
    lo, hi = np.percentile(d, [2.5, 97.5]); m = np.mean(d)
    res[k] = {'mean': round(float(m),3), 'ci': [round(float(lo),3), round(float(hi),3)],
              'p_gt0': round(float(np.mean(np.array(d) > 0)), 3)}
    print(f"  {k}: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(diff>0) = {np.mean(np.array(d)>0):.3f}")

p = P['Random forest']
print("\n=== threshold sensitivity, random forest ===")
from sklearn.metrics import precision_recall_curve
for lab, thr in [('0.5', 0.5), ('prevalence-matched', float(np.quantile(p, 1 - y.mean())))]:
    yh = (p >= thr).astype(int)
    print(f"  {lab:20s} thr={thr:.3f}  F1={f1_score(y,yh):.3f}  sens={yh[y==1].mean():.3f}  spec={1-yh[y==0].mean():.3f}")
pr, rc, th = precision_recall_curve(y, p)
f1s = 2*pr*rc/np.maximum(pr+rc, 1e-9); b = int(np.nanargmax(f1s[:-1]))
print(f"  {'F1-optimal':20s} thr={th[b]:.3f}  F1={f1s[b]:.3f}")
res['thresholds'] = {'f1_opt': round(float(th[b]),3), 'f1_opt_value': round(float(f1s[b]),3)}

print("\n=== calibration of the random forest (deciles of predicted risk) ===")
q = pd.qcut(p, 10, labels=False, duplicates='drop')
cal = pd.DataFrame({'q': q, 'p': p, 'y': y}).groupby('q').agg(n=('y','size'), pred=('p','mean'), obs=('y','mean'))
for i, r in cal.iterrows(): print(f"   decile {int(i)+1:2d}  n={int(r['n']):4d}  predicted {r['pred']:.3f}  observed {r['obs']:.3f}")
res['calibration'] = [[int(r['n']), round(float(r['pred']),3), round(float(r['obs']),3)] for _, r in cal.iterrows()]
from sklearn.linear_model import LogisticRegression as LRc
lo = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
c = LRc().fit(lo, y)
print(f"   calibration slope {float(c.coef_[0][0]):.3f}  intercept {float(c.intercept_[0]):+.3f}  (ideal 1.000 / 0.000)")
res['calibration_slope'] = round(float(c.coef_[0][0]),3); res['calibration_intercept'] = round(float(c.intercept_[0]),3)
np.save('results/oof_rf.npy', p); json.dump(res, open('results/inference.json','w'), indent=1)
print('\nsaved results/inference.json')
