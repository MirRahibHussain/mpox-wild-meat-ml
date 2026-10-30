import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

PATH = 'data/hunter_food_preparer_Dataverse.tab'
OUT  = ['soap','washsurface','leave_dead','dont_eat_lesions']
CAT = ['egender','lage','nethnicity','fprimaryoccupation','isecondaryoccupation',
       'hprimaryoccupationstatus','gprimaryoccupationpaidorunpaid','cstudyarea',
       'q51obtainwaterfordrinking','q44types_ownlivestock','q45havedogsathome',
       'q1ever_hunt','q27frequently_eatwildmeat_rainys',
       'q28frequently_eatwildmeat_drysea','q34eat_rawmeat',
       'q16have_sufferedanyinjuries','q48know_illnesses_someonehassuff']
NUM = ['opeopleliveinyourhousehold','q8howmanypeople_huntinhh','hunter_PO','farmer',
       'hunt_food','hunt_income','hunt_both','butcher_wildmeat','cook_wildmeat',
       'primates','rats','bathunt','concerned','feardis','inj','bite','claws','nails','treatment']

def load(handlers_only=True):
    h = pd.read_csv(PATH, sep='\t', low_memory=False)
    h['safe'] = h[OUT].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    if handlers_only:
        # the four practices are only defined for people who handle carcasses
        h = h[h['q17wildmeat_carcasses'] == 'Yes'].reset_index(drop=True)
    return h

def design(h, cat=None, num=None):
    cat = CAT if cat is None else cat
    num = NUM if num is None else num
    cat = [c for c in cat if c in h.columns]; num = [c for c in num if c in h.columns]
    X = pd.concat([h[cat].astype(str), h[num].apply(pd.to_numeric, errors='coerce')], axis=1)
    prep = ColumnTransformer([
      ('c', Pipeline([('i',SimpleImputer(strategy='most_frequent')),
                      ('o',OneHotEncoder(handle_unknown='ignore',sparse_output=False,min_frequency=20))]), cat),
      ('n', Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]), num)])
    return X, prep

def rf(prep):
    return Pipeline([('p',prep),('m',RandomForestClassifier(n_estimators=600,min_samples_leaf=5,
                     class_weight='balanced',random_state=0,n_jobs=-1))])
def lr(prep):
    return Pipeline([('p',prep),('m',LogisticRegression(max_iter=4000,class_weight='balanced'))])
