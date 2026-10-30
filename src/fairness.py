import sys; sys.path.insert(0,'src')
import pandas as pd, numpy as np
from common import load, design, rf, CAT, NUM
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score

h = load(); g = h['dlocality'].astype(str)
y = (h['safe'] <= 1).astype(int).values
h['indig'] = np.where(h['nethnicity'].astype(str)=='Baka','Baka (Indigenous)','Other (Bantu groups)')
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)

def oof(cat, num):
    X, prep = design(h, cat, num)
    return cross_val_predict(rf(prep), X, y, cv=cv, groups=g, method='predict_proba')[:,1]

def report(p, title, keys):
    print(f"\n--- {title} ---")
    yh = (p >= 0.5).astype(int)
    print(f"overall: AUC {roc_auc_score(y,p):.3f}  F1 {f1_score(y,yh):.3f}")
    for k in keys:
        print(f"{'  group':34s} {'n':>5s} {'prev':>6s} {'AUC':>6s} {'F1':>6s} {'TPR':>6s} {'FPR':>6s} {'mean p':>7s}")
        for v, idx in h.groupby(h[k]).groups.items():
            i = h.index.get_indexer(idx)
            yy, pp, hh = y[i], p[i], yh[i]
            if len(set(yy)) < 2: continue
            tpr = hh[yy==1].mean(); fpr = hh[yy==0].mean()
            print(f"  {str(v):32s} {len(i):5d} {yy.mean():6.3f} {roc_auc_score(yy,pp):6.3f} "
                  f"{f1_score(yy,hh):6.3f} {tpr:6.3f} {fpr:6.3f} {pp.mean():7.3f}")

p_full = oof(CAT, NUM)
report(p_full, "A. full model (all predictors)", ['indig','egender'])

# B. drop attribute-based predictors that would be unacceptable to act on
CAT_b = [c for c in CAT if c != 'nethnicity']
report(oof(CAT_b, NUM), "B. ethnicity removed", ['indig','egender'])

CAT_c = [c for c in CAT if c not in ('nethnicity','egender')]
NUM_c = [c for c in NUM if c not in ('primates','rats','bathunt')]
report(oof(CAT_c, NUM_c), "C. ethnicity, sex and species hunted removed", ['indig','egender'])
