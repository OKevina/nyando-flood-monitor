"""Registration & product checks for hand-labelled patches.

A) Shift-contrast: roll the flood mask across the SAR image. If labels are
   registered to the pixels beneath them, contrast (outside - inside, dB)
   should peak at shift 0.
B) Product check: compare hand vs dated VV-dB histograms, speckle
   (local std), and correlation with the orbit-median baseline rasters.
"""
import os, re, csv
import numpy as np
import rasterio
from rasterio.windows import from_bounds

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(ROOT, 'data', 'Nyando_Dataset')
SCENES = os.path.join(ROOT, 'data', 'scenes')
DATED = re.compile(r'^Nyando_(\d{4}-\d{2}-\d{2})_\d+$')


def load_ids(split):
    with open(os.path.join(DS, f'nyando_split_{split}.csv')) as f:
        return [r[0] for r in csv.reader(f) if r]


def vv_db(pid):
    with rasterio.open(os.path.join(DS, 'S1Hand', f'{pid}_S1Hand.tif')) as s:
        vv = s.read(1).astype(np.float32)
        b = s.bounds
    return 10.0 * np.log10(np.where(vv > 0, vv, np.nan)), b


def flood(pid):
    with rasterio.open(os.path.join(DS, 'LabelHand', f'{pid}_LabelHand.tif')) as s:
        return s.read(1) == 1


hand_ids = sorted({p for sp in ('train', 'val', 'test')
                   for p in load_ids(sp) if not DATED.match(p)})
hand_flood = [p for p in hand_ids]
# keep only patches whose label has flood
hand_flood = [p for p in hand_flood if flood(p).any()]
print(f'hand patches with flood: {len(hand_flood)}')

# cache SAR/labels once
data = {}
for p in hand_flood:
    data[p] = vv_db(p) + (flood(p),)

# ------------------------------------------------- A) shift contrast
print('\n=== A) Contrast (outside - inside, dB) vs label shift ===')
shifts = [0, 2, 4, 8, 16, 32, 64]
acc = {s: [] for s in shifts}
for pid, (vv, b, t) in data.items():
    valid = np.isfinite(vv)
    for s in shifts:
        f = np.roll(t, s, axis=1)
        # exclude wrapped columns
        if s:
            f[:, -s:] = False
        inside = vv[f & valid]
        outside = vv[(~f) & valid]
        if len(inside) > 50:
            acc[s].append(np.nanmean(outside) - np.nanmean(inside))
for s in shifts:
    v = np.array(acc[s])
    print(f'shift {s:3d}px: mean contrast {v.mean():+.2f} dB  (n={len(v)})')

# ------------------------------------------------- B) product checks
print('\n=== B1) VV-dB histogram percentiles ===')
def pct_summary(ids, label):
    vals = []
    for p in ids:
        vv, _ = vv_db(p)
        vals.append(vv[np.isfinite(vv)])
    v = np.concatenate(vals)
    print(f'{label:12s} p5={np.percentile(v,5):6.1f} p25={np.percentile(v,25):6.1f} '
          f'p50={np.percentile(v,50):6.1f} p75={np.percentile(v,75):6.1f} '
          f'p95={np.percentile(v,95):6.1f}')
pct_summary(hand_ids[:60], 'hand')
pct_summary([p for p in load_ids('test') if DATED.match(p)][:200], 'dated')

print('\n=== B2) Speckle: local std of VV-dB (5x5) ===')
def speckle(ids, n=40):
    out = []
    for p in ids[:n]:
        vv, _ = vv_db(p)
        v = np.nan_to_num(vv, nan=np.nanmedian(vv))
        # local std via uniform filter
        from scipy.ndimage import uniform_filter
        m = uniform_filter(v, size=5)
        m2 = uniform_filter(v * v, size=5)
        out.append(np.sqrt(np.maximum(m2 - m * m, 0)).mean())
    return np.mean(out)
print(f'hand  mean local-std: {speckle(hand_ids):.2f} dB')
print(f'dated mean local-std: {speckle([p for p in load_ids("test") if DATED.match(p)]):.2f} dB')

print('\n=== B3) Correlation with orbit-median baseline rasters ===')
meds = {}
for orb in ['130', '152', '']:
    f = os.path.join(SCENES, 'temporal_median_db' + (f'_{orb}' if orb else '') + '.tif')
    if os.path.exists(f):
        meds[orb or 'all'] = rasterio.open(f)
res = {k: [] for k in meds}
for pid, (vv, b, t) in list(data.items()):
    for name, src in meds.items():
        win = from_bounds(*b, transform=src.transform)
        r0, c0 = int(round(win.row_off)), int(round(win.col_off))
        r1, c1 = r0 + 256, c0 + 256
        if r1 > src.height or c1 > src.width:
            continue
        med = src.read(1, window=((r0, r1), (c0, c1)))
        if med.shape != (256, 256):
            continue
        m = np.isfinite(vv) & np.isfinite(med)
        if m.sum() > 1000:
            res[name].append(np.corrcoef(vv[m], med[m])[0, 1])
for name, v in res.items():
    v = np.array(v)
    print(f'median[{name:4s}]: mean r={v.mean():.3f}  median r={np.median(v):.3f}  n={len(v)}')
for src in meds.values():
    src.close()
