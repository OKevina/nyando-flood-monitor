"""Label-validity checks for the hand-labelled subset — run locally.

  A) Does hand-label flood sit on low VV backscatter (dark = water-like)?
     Compare median VV-dB inside vs outside flood, hand vs dated labels.
  B) Same tiles in both sets: for patch locations present as both a
     hand-labelled and dated patch, how much do the labels agree?
"""
import os, re, csv
import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(ROOT, 'data', 'Nyando_Dataset')
DATED = re.compile(r'^Nyando_(\d{4}-\d{2}-\d{2})_\d+$')


def load_ids(split):
    with open(os.path.join(DS, f'nyando_split_{split}.csv')) as f:
        return [r[0] for r in csv.reader(f) if r]


def sar_vv_db(pid):
    with rasterio.open(os.path.join(DS, 'S1Hand', f'{pid}_S1Hand.tif')) as s:
        vv = s.read(1).astype(np.float32)
    # patches store linear sigma0 (all values >= 0, median ~0.1) -> convert to dB
    return 10.0 * np.log10(np.where(vv > 0, vv, np.nan))


def label(pid):
    with rasterio.open(os.path.join(DS, 'LabelHand', f'{pid}_LabelHand.tif')) as s:
        return s.read(1)


def bounds(pid):
    with rasterio.open(os.path.join(DS, 'S1Hand', f'{pid}_S1Hand.tif')) as s:
        return tuple(round(v, 6) for v in s.bounds)


# ---------------------------------------------------------------- A
print('=== A. VV-dB inside vs outside flood labels ===')
for name, ids in [
    ('hand (all splits)', sorted({p for sp in ('train','val','test')
                                  for p in load_ids(sp) if not DATED.match(p)})),
    ('dated test (sample 200)', [p for p in load_ids('test')
                                 if DATED.match(p)][:200]),
]:
    din, dout = [], []
    for pid in ids:
        m = label(pid)
        t = m == 1
        if not t.any():
            continue
        vv = sar_vv_db(pid)
        valid = np.isfinite(vv)
        din.append(np.nanmedian(vv[t & valid]))
        dout.append(np.nanmedian(vv[~t & valid]))
    din, dout = np.array(din), np.array(dout)
    print(f'{name:22s} patches_with_flood={len(din):3d}  '
          f'median VV-dB inside={np.median(din):6.1f}  '
          f'outside={np.median(dout):6.1f}  '
          f'(inside-outside={np.median(din)-np.median(dout):+.1f} dB)')

# ---------------------------------------------------------------- B
print('\n=== B. Same tile in hand + dated sets: label agreement ===')
hand_by_loc, dated_by_loc = {}, {}
for sp in ('train', 'val', 'test'):
    for p in load_ids(sp):
        (dated_by_loc if DATED.match(p) else hand_by_loc).setdefault(bounds(p), []).append(p)

shared = sorted(set(hand_by_loc) & set(dated_by_loc))
print(f'shared locations: {len(shared)}')
rows = []
for loc in shared:
    hp = hand_by_loc[loc][0]
    hm = label(hp) == 1
    for dp in dated_by_loc[loc]:
        dm = label(dp) == 1
        inter = (hm & dm).sum()
        union = (hm | dm).sum()
        rows.append((hp, dp, int(hm.sum()), int(dm.sum()),
                     float(inter / max(union, 1)),
                     float(inter / max(hm.sum(), 1))))
ious = np.array([r[4] for r in rows])
rec = np.array([r[5] for r in rows])
print(f'hand-vs-dated label IoU: mean={ious.mean():.3f} median={np.median(ious):.3f} '
      f'frac>0.1={np.mean(ious > 0.1):.2f}')
print(f'hand flood covered by dated flood (recall): mean={rec.mean():.3f}')
print('\nworst-agreement examples (hand flood px, dated flood px, IoU):')
for r in sorted(rows, key=lambda r: r[4])[:8]:
    print(f'  {r[0]:14s} vs {r[1]:24s} hand={r[2]:6d} dated={r[3]:6d} IoU={r[4]:.3f}')
