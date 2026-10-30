import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

h = pd.read_csv('data/hunter_food_preparer_Dataverse.tab', sep='\t', low_memory=False)

items = ['soap','washsurface','leave_dead','dont_eat_lesions']
S = h[items].apply(pd.to_numeric, errors='coerce')
h['safe_score'] = S.sum(axis=1)                 # 0-4, higher = safer
h['high_risk']  = (h['safe_score'] <= 1).astype(int)   # 0 or 1 safe practices
print("composite safe_score distribution (0=least safe .. 4=safest):")
print(h['safe_score'].value_counts().sort_index().to_string())
print(f"\nhigh_risk prevalence: {h['high_risk'].mean():.3f}  (n={h['high_risk'].sum()})")

preds = ['egender','lage','nethnicity','fprimaryoccupation','isecondaryoccupation',
         'hprimaryoccupationstatus','gprimaryoccupationpaidorunpaid','hunter_PO',
         'cstudyarea','opeopleliveinyourhousehold','hunt_income']
preds = [p for p in preds if p in h.columns]
X = h[preds].astype(str)
y = h['high_risk']

prep = ColumnTransformer([('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), preds)])

models = {
 'logistic regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
 'random forest':       RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                               class_weight='balanced', random_state=0, n_jobs=-1),
 'gradient boosting':   HistGradientBoostingClassifier(random_state=0),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print("\n=== 5-fold cross-validation, outcome = high_risk ===")
for name, m in models.items():
    pipe = Pipeline([('prep', prep), ('clf', m)])
    auc = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    f1  = cross_val_score(pipe, X, y, cv=cv, scoring='f1')
    print(f"{name:20s} AUC {auc.mean():.3f} (+/-{auc.std():.3f})   F1 {f1.mean():.3f}")
print(f"\nbaseline (always predict majority) accuracy = {max(y.mean(), 1-y.mean()):.3f}")
