"""Exposure assessment + prediction of mpox-relevant exposure intensity.

Outcome is MEASURED from the survey (annual contact-event counts derived from
reported frequencies), never simulated. Variables used to build the exposure
metric are excluded from the predictor set to avoid circularity.
"""
import json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

PATH = 'data/hunter_food_preparer_Dataverse.tab'
RES  = 'results/'
RNG  = np.random.default_rng(20261030)

# --- frequency -> events per 6-month season -------------------------------
FREQ = {'Daily':182,'Weekly':26,'Monthly':6,'Every 6 months':1,
        'Once a year':0.5,'Less than once a year':0.25,'Never':0}
AMB  = 'Every months'          # ambiguous label; bounded 2 (bi-annual-ish) .. 3
# --- blood-staining ordinal -> proportion of handling events with blood ---
BLOOD_MAIN = {'Always':1.0,'Often':0.6,'Sometimes':0.3,'Rarely':0.1,'Never':0.0}
BLOOD_LOW  = {'Always':1.0,'Often':0.5,'Sometimes':0.2,'Rarely':0.05,'Never':0.0}
BLOOD_HIGH = {'Always':1.0,'Often':0.75,'Sometimes':0.4,'Rarely':0.15,'Never':0.0}

def seasonal(series, amb):
    m = dict(FREQ); m[AMB] = amb
    return series.map(m)

def build(amb=2.0, blood=BLOOD_MAIN):
    h = pd.read_csv(PATH, sep='\t', low_memory=False)
    h = h[h['q17wildmeat_carcasses'] == 'Yes'].reset_index(drop=True)
    h['area'] = h['cstudyarea'].str.replace('��', 'e', regex=False)
    rain = seasonal(h['q27frequently_eatwildmeat_rainys'], amb)
    dry  = seasonal(h['q28frequently_eatwildmeat_drysea'], amb)
    h['E_consume'] = rain.fillna(0) + dry.fillna(0)          # events / person-year
    butch = pd.to_numeric(h['butcher_wildmeat'], errors='coerce').fillna(0)
    h['E_butcher'] = h['E_consume'] * butch                  # carcass-contact events
    h['p_blood']   = h['q22often_clothesgetbloodstained'].map(blood)
    h['E_blood']   = h['E_butcher'] * h['p_blood'].fillna(0) # blood-contact events
    h['E_total']   = h['E_consume'] + h['E_butcher'] + h['E_blood']
    q = h['q38do_findviscerawithlesions']
    h['P4_lesion'] = q.isin(['Remove the lesion and eat the rest',
                             'Eat the viscera whole but well cooked',
                             'Eat the viscera normally']).astype(int)
    return h

# paper-1 predictor set MINUS everything that feeds the exposure metric
LEAK = ['q27frequently_eatwildmeat_rainys','q28frequently_eatwildmeat_drysea',
        'butcher_wildmeat','q22often_clothesgetbloodstained',
        'q38do_findviscerawithlesions','q17wildmeat_carcasses','dont_eat_lesions']
CAT = ['egender','lage','nethnicity','fprimaryoccupation','isecondaryoccupation',
       'hprimaryoccupationstatus','gprimaryoccupationpaidorunpaid','cstudyarea',
       'q51obtainwaterfordrinking','q44types_ownlivestock','q45havedogsathome',
       'q1ever_hunt','q34eat_rawmeat','q16have_sufferedanyinjuries',
       'q48know_illnesses_someonehassuff']
NUM = ['opeopleliveinyourhousehold','q8howmanypeople_huntinhh','hunter_PO','farmer',
       'hunt_food','hunt_income','hunt_both','cook_wildmeat','primates','rats',
       'bathunt','concerned','feardis','inj','bite','claws','nails','treatment',
       'soap','washsurface','leave_dead']

def design(h):
    cat = [c for c in CAT if c in h.columns and c not in LEAK]
    num = [c for c in NUM if c in h.columns and c not in LEAK]
    X = h[cat + num].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    for c in cat:
        X[c] = X[c].astype(str)
    pre = ColumnTransformer([
        ('c', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=20), cat),
        ('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num)])
    return X, pre, cat, num

MODELS = {
 'Random forest': lambda: RandomForestRegressor(n_estimators=600, min_samples_leaf=5,
                                                random_state=0, n_jobs=-1),
 'Gradient boosting': lambda: HistGradientBoostingRegressor(random_state=0),
 'Ridge regression': lambda: Ridge(alpha=1.0, random_state=None),
}

def loao(h, target='E_blood'):
    X, pre, cat, num = design(h)
    X = X.drop(columns=['cstudyarea'])
    pre = ColumnTransformer([
        ('c', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=20),
         [c for c in cat if c != 'cstudyarea']),
        ('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num)])
    y = np.log1p(h[target].values)
    res = {}
    for a in sorted(h['area'].unique()):
        tr, te = h['area'] != a, h['area'] == a
        pipe = Pipeline([('p', pre), ('m', MODELS['Random forest']())])
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[te])
        res[a] = {'n': int(te.sum()), 'r2': round(float(r2_score(y[te], p)), 3),
                  'spearman': round(float(spearmanr(y[te], p).statistic), 3)}
    res['mean_spearman'] = round(float(np.mean([v['spearman'] for k, v in res.items()
                                                if isinstance(v, dict)])), 3)
    return res

def q(x, p):
    return float(np.nanpercentile(x, p))

def profile(h):
    d = {}
    for k in ['E_consume', 'E_butcher', 'E_blood', 'E_total']:
        v = h[k].values
        d[k] = {'median': round(q(v,50),1), 'iqr': [round(q(v,25),1), round(q(v,75),1)],
                'mean': round(float(np.nanmean(v)),1), 'p90': round(q(v,90),1),
                'p99': round(q(v,99),1), 'max': round(float(np.nanmax(v)),1)}
    d['P4_lesion_pct'] = round(100*float(h['P4_lesion'].mean()), 1)
    by = {}
    for a, gg in h.groupby('area'):
        by[a] = {'n': len(gg),
                 'E_blood_median': round(float(gg['E_blood'].median()),1),
                 'E_blood_mean': round(float(gg['E_blood'].mean()),1),
                 'E_total_median': round(float(gg['E_total'].median()),1),
                 'P4_lesion_pct': round(100*float(gg['P4_lesion'].mean()),1)}
    d['by_area'] = by
    return d

def boot_spearman(y, pred, groups, B=2000):
    """village-cluster bootstrap CI for Spearman correlation"""
    gs = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in gs}
    vals = []
    for _ in range(B):
        pick = RNG.choice(gs, size=len(gs), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        r = spearmanr(y[ii], pred[ii]).statistic
        if np.isfinite(r):
            vals.append(r)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]

def evaluate2(h, target):
    X, pre, cat, num = design(h)
    y = np.log1p(h[target].values)
    g = h['dlocality'].astype(str).values
    cv = GroupKFold(n_splits=5)
    out = {}
    for name, mk in MODELS.items():
        pipe = Pipeline([('p', pre), ('m', mk())])
        pred = cross_val_predict(pipe, X, y, cv=cv, groups=g, n_jobs=1)
        out[name] = {'r2': round(float(r2_score(y, pred)), 3),
                     'spearman': round(float(spearmanr(y, pred).statistic), 3),
                     'spearman_ci': boot_spearman(y, pred, g),
                     'mae_log': round(float(mean_absolute_error(y, pred)), 3),
                     'mae_events': round(float(np.mean(np.abs(np.expm1(y) - np.expm1(pred)))), 1)}
    out['_n_features'] = int(Pipeline([('p', pre)]).fit_transform(X).shape[1])
    out['_n_villages'] = int(len(np.unique(g)))
    return out

if __name__ == '__main__':
    out = {'frequency_map': {**FREQ, AMB: '2 (primary) / 3 (sensitivity)'},
           'blood_map': BLOOD_MAIN,
           'blood_map_low': BLOOD_LOW, 'blood_map_high': BLOOD_HIGH}
    h = build()
    out['n'] = len(h)
    out['profile_main'] = profile(h)
    out['sensitivity'] = {
        'amb_3':      profile(build(amb=3.0))['E_blood'],
        'blood_low':  profile(build(blood=BLOOD_LOW))['E_blood'],
        'blood_high': profile(build(blood=BLOOD_HIGH))['E_blood'],
    }
    out['prediction'] = {t: evaluate2(h, t) for t in ['E_blood', 'E_total', 'E_consume']}
    out['loao'] = {t: loao(h, t) for t in ['E_blood', 'E_total']}
    json.dump(out, open(RES + 'exposure_model.json', 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ['n', 'prediction', 'loao']}, indent=1))
