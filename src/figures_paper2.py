import json, html, os
os.makedirs('figures', exist_ok=True)
R = json.load(open('results/exposure_model.json'))
P = R['profile_main']; BY = P['by_area']; AREAS = ['Bengbis','Lomie','Mintom','Yokadouma']
LBL  = {'Bengbis':'Bengbis','Lomie':'Lomié','Mintom':'Mintom','Yokadouma':'Yokadouma'}
INK, MID, LITE = '#1a1a1a', '#5f6b6d', '#c9d1d3'
ACC, ACC2, WARN = '#2b6cb0', '#4a9d9c', '#b7791f'
F = "'Helvetica Neue',Helvetica,Arial,sans-serif"

def wrap(svg, w, h, name):
    open(f'figures/{name}.svg','w').write(svg)
    open(f'figures/{name}.html','w').write(
        f'<html><head><meta charset="utf-8"><style>html,body{{margin:0;padding:0;'
        f'background:#fff}}</style></head><body>{svg}</body></html>')
    os.system(f'wkhtmltoimage --width 2280 --disable-smart-width --quality 100 '
              f'figures/{name}.html figures/{name}.png >/dev/null 2>&1')
    print(name, 'ok')

def txt(x,y,s,sz=13,col=INK,anc='start',w='400',st=''):
    return (f'<text x="{x}" y="{y}" font-family="{F}" font-size="{sz}" fill="{col}" '
            f'text-anchor="{anc}" font-weight="{w}"{st}>{html.escape(str(s))}</text>')

# ---------------- Figure 1: the four Codex stages, and where we stop -------
W,H = 960, 430
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
     f'<rect width="{W}" height="{H}" fill="#fff"/>']
stages = [('1. Hazard\nidentification','Complete','MPXV and its contact\nroutes are established',ACC2),
          ('2. Hazard\ncharacterisation','Not available','No human dose–response\nfunction exists',WARN),
          ('3. Exposure\nassessment','This study','67.2 contact events\nper person-year (median)',ACC),
          ('4. Risk\ncharacterisation','Blocked','Requires all four stages',MID)]
bw, gap, x0, by = 186, 44, 30, 120
for i,(name,status,note,col) in enumerate(stages):
    x = x0 + i*(bw+gap)
    done = status in ('Complete','This study')
    s.append(f'<rect x="{x}" y="{by}" width="{bw}" height="104" rx="7" fill="{"#fff" if not done else col+"14"}" '
             f'stroke="{col}" stroke-width="{2.4 if status=="This study" else 1.3}" '
             f'{"stroke-dasharray=\'5 4\'" if not done else ""}/>')
    for j,line in enumerate(name.split('\n')):
        s.append(txt(x+bw/2, by+30+j*18, line, 15, INK, 'middle', '600'))
    s.append(txt(x+bw/2, by+84, status, 13, col, 'middle', '700'))
    for j,line in enumerate(note.split('\n')):
        s.append(txt(x+bw/2, by+134+j*17, line, 12.5, MID, 'middle'))
    if i < 3:
        ax = x+bw+8
        s.append(f'<path d="M{ax} {by+52} L{ax+gap-16} {by+52}" stroke="{LITE}" stroke-width="2" '
                 f'marker-end="url(#ar)"/>')
s.append(f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
         f'markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{LITE}"/></marker></defs>')
s.append(txt(30, 48, 'Codex Alimentarius microbiological risk assessment framework', 17, INK, 'start','600'))
s.append(txt(30, 72, 'Status of each stage for monkeypox virus in the wild meat chain', 13.5, MID))
# missing-parameter callout
s.append(f'<rect x="30" y="300" width="{W-60}" height="98" rx="7" fill="#fafbfb" stroke="{LITE}"/>')
s.append(txt(46, 326, 'Four parameters stand between stage 3 and stage 4:', 13.5, INK,'start','600'))
for j,p in enumerate(['active-infection prevalence in the hunted taxa',
                      'viral load in contacted tissue',
                      'transfer efficiency to mucosa or broken skin',
                      'a validated human dose–response function']):
    cx = 46 + (j%2)*450; cy = 352 + (j//2)*22
    s.append(f'<circle cx="{cx+4}" cy="{cy-4}" r="2.6" fill="{WARN}"/>')
    s.append(txt(cx+16, cy, p, 12.5, MID))
s.append('</svg>')
wrap('\n'.join(s), W, H, 'p2fig1')

# ---------------- Figure 2: exposure and the lesion pathway by area --------
W,H = 900, 430
mx = max(BY[a]['E_blood_mean'] for a in AREAS)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
     f'<rect width="{W}" height="{H}" fill="#fff"/>']
s.append(txt(30, 40, 'Blood-contact exposure and the lesioned-viscera pathway, by study area', 17, INK,'start','600'))
s.append(txt(30, 63, 'Left: blood-contact events per person-year. Right: respondents consuming viscera with visible lesions.', 12.5, MID))
# panel A
ax0, ay0, aw, ah = 42, 142, 360, 196
s.append(f'<line x1="{ax0}" y1="{ay0+ah}" x2="{ax0+aw}" y2="{ay0+ah}" stroke="{LITE}"/>')
for v in [0,10,20,30,40]:
    y = ay0+ah - v/45*ah
    s.append(f'<line x1="{ax0}" y1="{y:.1f}" x2="{ax0+aw}" y2="{y:.1f}" stroke="#eef1f2"/>')
    s.append(txt(ax0-8, y+4, v, 11.5, MID, 'end'))
bw2 = 44
for i,a in enumerate(AREAS):
    cx = ax0 + 34 + i*(aw-60)/4
    mean, med = BY[a]['E_blood_mean'], BY[a]['E_blood_median']
    hm = mean/45*ah
    s.append(f'<rect x="{cx-bw2/2:.1f}" y="{ay0+ah-hm:.1f}" width="{bw2}" height="{hm:.1f}" fill="{ACC}" opacity="0.78" rx="2"/>')
    s.append(txt(cx, ay0+ah-hm-8, f'{mean:.1f}', 11.5, ACC, 'middle','600'))
    ym = ay0+ah - med/45*ah
    s.append(f'<line x1="{cx-bw2/2-5:.1f}" y1="{ym:.1f}" x2="{cx+bw2/2+5:.1f}" y2="{ym:.1f}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt(cx, ay0+ah+18, LBL[a], 12, INK, 'middle'))
s.append(txt(ax0-8, ay0-10, 'events / person-year', 11.5, MID, 'start'))
s.append(txt(ax0+4, ay0+ah+40, '— median', 11.5, INK)); s.append(txt(ax0+76, ay0+ah+40, 'bar = mean', 11.5, ACC))
s.append(txt(ax0-8, ay0-32, 'A  Blood-contact events', 13.5, INK,'start','600'))
# panel B
bx0 = 510; s.append(txt(bx0, ay0-32, 'B  Consumption of lesioned viscera', 13.5, INK,'start','600'))
s.append(f'<line x1="{bx0}" y1="{ay0+ah}" x2="{bx0+340}" y2="{ay0+ah}" stroke="{LITE}"/>')
for v in [0,20,40,60,80]:
    y = ay0+ah - v/80*ah
    s.append(f'<line x1="{bx0}" y1="{y:.1f}" x2="{bx0+340}" y2="{y:.1f}" stroke="#eef1f2"/>')
    s.append(txt(bx0-8, y+4, f'{v}%', 11.5, MID, 'end'))
for i,a in enumerate(sorted(AREAS, key=lambda z: BY[z]['P4_lesion_pct'])):
    cx = bx0 + 36 + i*280/4
    p = BY[a]['P4_lesion_pct']; hp = p/80*ah
    s.append(f'<rect x="{cx-bw2/2:.1f}" y="{ay0+ah-hp:.1f}" width="{bw2}" height="{hp:.1f}" fill="{WARN}" opacity="0.82" rx="2"/>')
    s.append(txt(cx, ay0+ah-hp-8, f'{p:.1f}%', 11.5, WARN, 'middle','600'))
    s.append(txt(cx, ay0+ah+18, LBL[a], 12, INK, 'middle'))
s.append(txt(bx0, ay0+ah+40, '5.1-fold range across four areas of the same region', 11.5, MID))
s.append('</svg>')
wrap('\n'.join(s), W, H, 'p2fig2')

# ---------------- Figure 3: transferability ------------------------------
W,H = 900, 395
L = R['loao']
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
     f'<rect width="{W}" height="{H}" fill="#fff"/>']
s.append(txt(30, 40, 'Exposure intensity does not transfer to an unseen area', 17, INK,'start','600'))
s.append(txt(30, 63, 'Rank correlation between observed and predicted exposure. Pooled uses village-grouped cross-validation;', 12.5, MID))
s.append(txt(30, 80, 'each area bar holds that entire area out and removes area from the predictor set.', 12.5, MID))
gx0, gy0, gw, gh = 70, 120, 780, 170
zero = gy0 + gh*0.80
s.append(f'<line x1="{gx0}" y1="{zero}" x2="{gx0+gw}" y2="{zero}" stroke="{INK}" stroke-width="1.2"/>')
for v in [-0.1,0.0,0.1,0.2,0.3,0.4]:
    y = zero - v/0.45*(gh*0.78)
    if abs(v)>1e-9: s.append(f'<line x1="{gx0}" y1="{y:.1f}" x2="{gx0+gw}" y2="{y:.1f}" stroke="#eef1f2"/>')
    s.append(txt(gx0-10, y+4, f'{v:.1f}', 11.5, MID, 'end'))
bars = [('Pooled CV\n(blood contact)', R['prediction']['E_blood']['Random forest']['spearman'], ACC),
        ('Pooled CV\n(composite)', R['prediction']['E_total']['Random forest']['spearman'], ACC)]
for a in AREAS:
    bars.append((f'{LBL[a]}\nheld out', L['E_blood'][a]['spearman'], WARN))
bw3 = 62
for i,(lab,v,col) in enumerate(bars):
    cx = gx0 + 70 + i*(gw-140)/5
    hv = v/0.45*(gh*0.78)
    y = zero - hv if v>=0 else zero
    s.append(f'<rect x="{cx-bw3/2:.1f}" y="{y:.1f}" width="{bw3}" height="{abs(hv):.1f}" fill="{col}" opacity="0.8" rx="2"/>')
    s.append(txt(cx, (zero-hv-9) if v>=0 else (zero-hv+18), f'{v:.3f}', 11.5, col, 'middle','600'))
    for j,line in enumerate(lab.split('\n')):
        s.append(txt(cx, gy0+gh+26+j*16, line, 11.5, INK, 'middle'))
s.append(txt(gx0-10, gy0-10, "Spearman ρ", 11.5, MID, 'start'))
s.append(txt(gx0, gy0+gh+76, 'R² was negative for every held-out area (−0.032 to −0.377): worse than predicting the training-set mean.', 12, MID))
s.append('</svg>')
wrap('\n'.join(s), W, H, 'p2fig3')
