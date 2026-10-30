import pandas as pd, numpy as np
pd.set_option('display.width', 200)
h = pd.read_csv('data/hunter_food_preparer_Dataverse.tab', sep='\t', low_memory=False)

risk_items = {           # 1 = safer behaviour
 'soap':'washes hands with soap',
 'washsurface':'washes cutting surface',
 'surface_soap':'washes surface with soap',
 'leave_dead':'leaves animals found dead',
 'dont_eat_lesions':'avoids animals with lesions',
 'wash_handle':'washes hands after handling',
 'protect':'uses protection',
 'concerned':'concerned about disease',
}
print("=== candidate risk items ===")
for c,lab in risk_items.items():
    if c in h.columns:
        s = pd.to_numeric(h[c], errors='coerce')
        print(f"{c:20s} n={s.notna().sum():5d}  mean={s.mean():.3f}  ({lab})")

cand_pred = ['egender','lage','nethnicity','fprimaryoccupation','isecondaryoccupation',
             'hprimaryoccupationstatus','gprimaryoccupationpaidorunpaid','hunter_PO']
print("\n=== candidate predictors ===")
for c in cand_pred:
    if c in h.columns:
        print(f"{c:35s} missing={h[c].isna().sum():5d}  levels={h[c].nunique()}")
    else:
        print(f"{c:35s} MISSING")

# look for village / area / education columns
print("\n=== other likely predictors in the file ===")
pat = r'village|site|area|region|district|educat|school|literac|household|income|train|aware|know'
print([c for c in h.columns if pd.Series([c]).str.contains(pat, case=False, regex=True).iloc[0]][:40])
