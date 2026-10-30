import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from common import load, design, rf, CAT, NUM
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

h = load(); g = h['dlocality'].astype(str).values
y = (h['safe'] <= 1).astype(int).values

# non-stigmatising, field-answerable candidate pool (model C set)
CATc = [c for c in CAT if c not in ('nethnicity','egender')]
NUMc = [c for c in NUM if c not in ('primates','rats','bathunt')]
X, prep = design(h, CATc, NUMc)
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)

# ---------- reference 1: full ML model ----------
p_ml = cross_val_predict(rf(prep), X, y, cv=cv, groups=g, method='predict_proba')[:,1]

# ---------- binary design matrix for the scorecard ----------
prep.fit(X)
names = list(prep.get_feature_names_out())
Z = pd.DataFrame(prep.transform(X), columns=names)
# make every candidate item a yes/no flag: numeric items split at their median
num_cols = [c for c in names if c.startswith('n__')]
for c in num_cols:
    Z[c] = (Z[c] > 0).astype(int)        # standardised -> above-average = 1
Z = Z.loc[:, Z.var() > 0]
names = list(Z.columns)
Zv = Z.values

def fit_card(Zt, yt, k_target=12, max_pts=3):
    """L1-select ~k items, then round coefficients to small integer points."""
    lo, hi = 1e-4, 5.0
    for _ in range(30):
        C = (lo*hi)**0.5
        m = LogisticRegression(penalty='l1', solver='liblinear', C=C,
                               class_weight='balanced', max_iter=4000).fit(Zt, yt)
        nz = int((m.coef_[0] != 0).sum())
        if nz > k_target: hi = C
        else: lo = C
    coef = m.coef_[0]
    idx = np.where(coef != 0)[0]
    if len(idx) == 0: return idx, np.array([])
    w = coef[idx]
    pts = np.round(w / np.abs(w).max() * max_pts).astype(int)
    keep = pts != 0
    return idx[keep], pts[keep]

# ---------- reference 2: learned scorecard, selection inside each fold ----------
p_card = np.zeros(len(y)); p_hand = np.zeros(len(y))
for tr, te in cv.split(Zv, y, groups=g):
    idx, pts = fit_card(Zv[tr], y[tr])
    p_card[te] = Zv[te][:, idx] @ pts
    # hand-weighted benchmark: same items, +-1 each (direction only), Quiner-style
    p_hand[te] = Zv[te][:, idx] @ np.sign(pts)

def line(lab, p):
    thr = np.quantile(p, 1 - y.mean())
    yh = (p >= thr).astype(int)
    print(f"{lab:46s} AUC {roc_auc_score(y,p):.3f}   F1 {f1_score(y,yh):.3f}")

print("=== village-grouped 5-fold CV, high risk = <=1 safe practice ===")
line("full ML model (random forest, 34 inputs)", p_ml)
line("learned scorecard (<=12 items, integer points)", p_card)
line("hand-weighted index, same items (+/-1 each)", p_hand)

print("\n=== complexity / accuracy trade-off (items vs out-of-fold AUC) ===")
print(f"{'target items':>13s} {'max points':>11s} {'mean items used':>16s} {'learned AUC':>12s} {'hand AUC':>9s}")
for kt, mp in [(6,3),(10,3),(12,3),(16,5),(20,5),(25,5)]:
    pc = np.zeros(len(y)); ph = np.zeros(len(y)); used=[]
    for tr, te in cv.split(Zv, y, groups=g):
        i2, p2 = fit_card(Zv[tr], y[tr], k_target=kt, max_pts=mp)
        used.append(len(i2))
        pc[te] = Zv[te][:, i2] @ p2
        ph[te] = Zv[te][:, i2] @ np.sign(p2)
    print(f"{kt:13d} {mp:11d} {np.mean(used):16.1f} {roc_auc_score(y,pc):12.3f} {roc_auc_score(y,ph):9.3f}")

# ---------- final scorecard fitted on all data ----------
idx, pts = fit_card(Zv, y, k_target=20, max_pts=5)
raw = Zv[:, idx] @ pts
lo_s, hi_s = pts[pts<0].sum(), pts[pts>0].sum()
score = (raw - lo_s) / (hi_s - lo_s) * 100
print(f"\n=== final scorecard: {len(idx)} items, raw range {lo_s} to {hi_s} -> 0-100 ===")
order = np.argsort(-np.abs(pts))
for j in order:
    print(f"{pts[j]:+3d}  {names[idx[j]]}")
print(f"\nscore AUC (in-sample, for calibration only): {roc_auc_score(y, score):.3f}")
print("\nobserved high-risk prevalence by score band:")
band = pd.cut(score, [-0.1,20,40,60,80,100], labels=['0-20','21-40','41-60','61-80','81-100'])
t = pd.DataFrame({'band':band,'y':y}).groupby('band', observed=True)['y'].agg(['size','mean'])
for b,r in t.iterrows(): print(f"  {str(b):7s} n={int(r['size']):5d}   high risk {r['mean']:5.1%}")
np.save('/home/claude/mpox/score.npy', score)
