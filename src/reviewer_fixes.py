import sys, json; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from common import load, design, rf, CAT, NUM
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
out = {}

# ---------- 1. leave-one-area-out ----------
h = load(); y = (h['safe'] <= 1).astype(int).values
X, prep = design(h)
print("=== 1. Leave-one-study-area-out validation ===")
print(f"{'held-out area':14s} {'n test':>7s} {'prevalence':>11s} {'AUC':>7s}")
loao = []
for area in sorted(h['cstudyarea'].dropna().unique()):
    te = (h['cstudyarea'] == area).values; tr = ~te
    m = rf(prep).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    a = roc_auc_score(y[te], p)
    loao.append([str(area), int(te.sum()), round(float(y[te].mean()), 3), round(float(a), 3)])
    print(f"{str(area):14s} {te.sum():7d} {y[te].mean():11.3f} {a:7.3f}")
print(f"{'mean':14s} {'':7s} {'':11s} {np.mean([r[3] for r in loao]):7.3f}")
out['loao'] = loao

# ---------- 2. scorecard without the two uninterpretable items ----------
CATc = [c for c in CAT if c not in ('nethnicity','egender')]
NUMc = [c for c in NUM if c not in ('primates','rats','bathunt')]
def sweep(cat, num, tag):
    Xi, pi = design(h, cat, num); pi.fit(Xi)
    names = list(pi.get_feature_names_out()); Z = pd.DataFrame(pi.transform(Xi), columns=names)
    for c in [c for c in names if c.startswith('n__')]: Z[c] = (Z[c] > 0).astype(int)
    Z = Z.loc[:, Z.var() > 0]; names = list(Z.columns); Zv = Z.values
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0); g = h['dlocality'].astype(str).values
    def fit_card(Zt, yt, k=20, mp=5):
        lo, hi = 1e-4, 5.0
        for _ in range(30):
            C = (lo*hi)**0.5
            mm = LogisticRegression(penalty='l1', solver='liblinear', C=C, class_weight='balanced', max_iter=4000).fit(Zt, yt)
            if int((mm.coef_[0] != 0).sum()) > k: hi = C
            else: lo = C
        co = mm.coef_[0]; idx = np.where(co != 0)[0]
        pts = np.round(co[idx]/np.abs(co[idx]).max()*mp).astype(int)
        keep = pts != 0; return idx[keep], pts[keep]
    pc = np.zeros(len(y)); used = []
    for tr, te in cv.split(Zv, y, groups=g):
        i2, p2 = fit_card(Zv[tr], y[tr]); used.append(len(i2)); pc[te] = Zv[te][:, i2] @ p2
    a = roc_auc_score(y, pc)
    idx, pts = fit_card(Zv, y)
    print(f"\n{tag}: out-of-fold AUC {a:.3f}, mean {np.mean(used):.1f} items")
    card = []
    for j in np.argsort(-np.abs(pts)):
        card.append([int(pts[j]), names[idx[j]]]); print(f"  {pts[j]:+3d}  {names[idx[j]]}")
    return round(float(a), 3), card
print("\n=== 2. Does dropping the two uninterpretable items cost anything? ===")
a_full, card_full = sweep(CATc, NUMc, 'with all candidate items')
CATd = [c for c in CATc if c not in ('q45havedogsathome','lage')]
a_drop, card_drop = sweep(CATd, NUMc, 'with dogs-at-home and age band removed from the pool')
print(f"\ncost of removing them: {a_full - a_drop:+.3f} AUC")
out['card_full'] = {'auc': a_full, 'card': card_full}
out['card_drop'] = {'auc': a_drop, 'card': card_drop}

# ---------- 3. who was excluded ----------
print("\n=== 3. The 111 excluded non-handlers vs the 2,262 analytic sample ===")
full = load(handlers_only=False)
exc = full[full['q17wildmeat_carcasses'] != 'Yes']
inc = full[full['q17wildmeat_carcasses'] == 'Yes']
rows = []
for v, lab in [('egender','Female'), ('cstudyarea', None), ('nethnicity','Baka'),
               ('hprimaryoccupationstatus','Temporary'), ('q1ever_hunt','Yes')]:
    if lab:
        a = (exc[v].astype(str) == lab).mean(); b = (inc[v].astype(str) == lab).mean()
        rows.append([f'{v} = {lab}', round(float(b)*100,1), round(float(a)*100,1)])
        print(f"  {v+' = '+lab:42s} included {b:6.1%}   excluded {a:6.1%}")
    else:
        for k in sorted(inc[v].dropna().unique()):
            a = (exc[v].astype(str) == str(k)).mean(); b = (inc[v].astype(str) == str(k)).mean()
            rows.append([f'{v} = {k}', round(float(b)*100,1), round(float(a)*100,1)])
            print(f"  {v+' = '+str(k):42s} included {b:6.1%}   excluded {a:6.1%}")
print(f"\n  mean safe-practice score      included {inc['safe'].mean():.2f}   excluded {exc['safe'].mean():.2f}")
out['excluded'] = rows
json.dump(out, open('results/reviewer_fixes.json','w'), indent=1)
print('\nsaved results/reviewer_fixes.json')
