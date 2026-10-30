import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

h = pd.read_csv('data/hunter_food_preparer_Dataverse.tab', sep='\t', low_memory=False)
OUT = ['soap','washsurface','leave_dead','dont_eat_lesions']
h['safe'] = h[OUT].apply(pd.to_numeric, errors='coerce').sum(axis=1)
y = (h['safe'] <= 1).astype(int)
groups = h['dlocality'].astype(str)          # 57 villages -> cluster-aware CV

# ---- predictors, grouped by theme. Nothing downstream of the outcome. ----
cat = ['egender','lage','nethnicity','fprimaryoccupation','isecondaryoccupation',
       'hprimaryoccupationstatus','gprimaryoccupationpaidorunpaid',
       'cstudyarea','q51obtainwaterfordrinking','q44types_ownlivestock',
       'q45havedogsathome','q1ever_hunt','q17wildmeat_carcasses',
       'q27frequently_eatwildmeat_rainys','q28frequently_eatwildmeat_drysea',
       'q34eat_rawmeat','q16have_sufferedanyinjuries','q48know_illnesses_someonehassuff']
num = ['opeopleliveinyourhousehold','q8howmanypeople_huntinhh',
       'hunter_PO','farmer','hunt_food','hunt_income','hunt_both',
       'butcher_wildmeat','cook_wildmeat','primates','rats','bathunt',
       'concerned','feardis','inj','bite','claws','nails','treatment']
cat = [c for c in cat if c in h.columns]
num = [c for c in num if c in h.columns]
X = pd.concat([h[cat].astype(str), h[num].apply(pd.to_numeric, errors='coerce')], axis=1)
print(f"n = {len(X)}   predictors: {len(cat)} categorical + {len(num)} numeric   villages: {groups.nunique()}")
print(f"outcome prevalence (high risk) = {y.mean():.3f}")

prep = ColumnTransformer([
  ('c', Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                  ('o', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=20))]), cat),
  ('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num)])

models = {
 'logistic regression': LogisticRegression(max_iter=4000, class_weight='balanced'),
 'random forest':       RandomForestClassifier(n_estimators=600, min_samples_leaf=5,
                                               class_weight='balanced', random_state=0, n_jobs=-1),
 'gradient boosting':   HistGradientBoostingClassifier(random_state=0, max_iter=300,
                                                       learning_rate=0.06),
}
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
print("\n=== 5-fold CV, grouped by village (no village in both train and test) ===")
for name, m in models.items():
    pipe = Pipeline([('p', prep), ('m', m)])
    auc = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring='roc_auc')
    f1  = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring='f1')
    print(f"{name:20s} AUC {auc.mean():.3f} (+/-{auc.std():.3f})   F1 {f1.mean():.3f}")
