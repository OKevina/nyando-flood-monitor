"""Hand-labelled subset diagnostics — run locally, no retraining needed.

Checks, in order:
  1. Tile overlap across splits (spatial leakage): same patch bounds in
     train/val/test?
  2. Hand-label geometry: are flood regions near-rectangular polygons?
  3. Permanent-water overlap: how much hand-label flood falls inside the
     permanent_water_mask (label-convention conflict check).
  4. Tolerant metrics + FP rate on flood-free hand patches, using the
     locally saved fine-tuned checkpoint (CPU).
"""
import os, re, sys, csv
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(ROOT, 'data', 'Nyando_Dataset')
SCENES = os.path.join(ROOT, 'data', 'scenes')
DATED = re.compile(r'^Nyando_(\d{4}-\d{2}-\d{2})_\d+$')


def load_ids(split):
    path = os.path.join(DS, f'nyando_split_{split}.csv')
    with open(path) as f:
        return [r[0] for r in csv.reader(f) if r]


def patch_bounds(pid):
    with rasterio.open(os.path.join(DS, 'S1Hand', f'{pid}_S1Hand.tif')) as s:
        return tuple(round(v, 6) for v in s.bounds)


def label_array(pid):
    with rasterio.open(os.path.join(DS, 'LabelHand', f'{pid}_LabelHand.tif')) as s:
        return s.read(1)


# ---------------------------------------------------------------- 1. tiles
def tile_overlap():
    print('=== 1. Spatial overlap across splits (same bounds = same tile) ===')
    locs = {}
    for split in ['train', 'val', 'test']:
        ids = load_ids(split)
        locs[split] = {}
        for pid in ids:
            locs[split].setdefault(patch_bounds(pid), []).append(pid)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        shared = set(locs[a]) & set(locs[b])
        print(f'{a}∩{b}: {len(shared)} shared tile locations '
              f'({len(locs[a])} vs {len(locs[b])} unique tiles)')
    all_loc = set().union(*[set(v) for v in locs.values()])
    print(f'total unique tiles: {len(all_loc)}; '
          f'multi-date tiles: {sum(1 for k in all_loc if sum(k in locs[s] for s in locs) > 1)}')


# ------------------------------------------------------- 2. label geometry
def hand_geometry():
    print('\n=== 2. Hand-label geometry (are flood regions rectangular?) ===')
    ids = [p for p in load_ids('test') + load_ids('val') + load_ids('train')
           if not DATED.match(p)]
    ids = sorted(set(ids))
    rect_px = tot_px = 0
    comp_sizes, fill_ratios = [], []
    for pid in ids:
        m = label_array(pid)
        t = m == 1
        if not t.any():
            continue
        lab, n = ndimage.label(t)
        for i in range(1, n + 1):
            comp = lab == i
            sz = int(comp.sum())
            if sz < 20:
                continue
            ys, xs = np.where(comp)
            bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            fill = sz / bbox_area
            comp_sizes.append(sz)
            fill_ratios.append(fill)
            tot_px += sz
            if fill > 0.8:
                rect_px += sz
    fr = np.array(fill_ratios)
    print(f'hand patches with flood: examined {len(comp_sizes)} components >=20px')
    print(f'fill-ratio (area/bbox): median={np.median(fr):.2f} '
          f'mean={fr.mean():.2f}  frac>0.8 (rect-like)={rect_px}/{tot_px} px '
          f'({rect_px/max(tot_px,1):.1%})')
    print(f'component size: median={np.median(comp_sizes):.0f}px '
          f'max={max(comp_sizes)}')


# ------------------------------------------------- 3. permanent-water overlap
def perm_overlap():
    print('\n=== 3. Permanent-water overlap ===')
    perm_path = os.path.join(SCENES, 'permanent_water_mask.tif')
    with rasterio.open(perm_path) as perm:
        ptransform = perm.transform
        for group_name, ids in [
            ('hand test', [p for p in load_ids('test') if not DATED.match(p)]),
            ('dated test (sample)', [p for p in load_ids('test')
                                     if DATED.match(p)][:60]),
        ]:
            fp_in_perm = fp_tot = perm_tot = npix = 0
            for pid in ids:
                with rasterio.open(os.path.join(DS, 'S1Hand', f'{pid}_S1Hand.tif')) as s:
                    b = s.bounds
                win = from_bounds(*b, transform=ptransform)
                # window may be off by a pixel; round to ints and clip
                row0 = max(0, int(round(win.row_off))); col0 = max(0, int(round(win.col_off)))
                row1 = min(perm.height, int(round(win.row_off + win.height)))
                col1 = min(perm.width, int(round(win.col_off + win.width)))
                pm = perm.read(1, window=((row0, row1), (col0, col1)))
                if pm.shape != (256, 256):
                    pm = ndimage.zoom(pm.astype(np.float32),
                                      (256 / pm.shape[0], 256 / pm.shape[1]),
                                      order=0).astype(np.uint8)
                m = label_array(pid)
                t = m == 1
                fp_tot += int(t.sum()); npix += t.size
                perm_tot += int((pm == 1).sum())
                fp_in_perm += int((t & (pm == 1)).sum())
            print(f'{group_name:22s} n={len(ids):3d}  '
                  f'flood px inside perm-water: {fp_in_perm}/{fp_tot} '
                  f'({fp_in_perm/max(fp_tot,1):.1%})   '
                  f'perm-water covers {perm_tot/max(npix,1):.1%} of patch area')


# ------------------------------------------------------- 4. tolerant metrics
def tolerant_metrics():
    print('\n=== 4. Tolerant metrics on hand-labelled test patches ===')
    try:
        import torch
        sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
        from importlib import import_module
        inf = import_module('05_inference')
        UNet = inf.UNet
    except Exception as e:
        print(f'skipping (torch/model import failed): {e}')
        return
    ckpts = [os.path.join(ROOT, '..', 'Downloads', 'nyando_finetuned_model.pth'),
             os.path.join(os.path.expanduser('~'), 'Downloads', 'nyando_finetuned_model.pth'),
             os.path.join(DS, 'nyando_finetuned_model.pth'),
             os.path.join(ROOT, 'models', 'nyando_finetuned_model.pth')]
    ckpt = next((c for c in ckpts if os.path.exists(c)), None)
    if ckpt is None:
        print('no local checkpoint found; skipping')
        return
    device = 'cpu'
    model = UNet(in_channels=2, out_channels=1)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ids = [p for p in load_ids('test') if not DATED.match(p)]

    def predict(pid):
        with rasterio.open(os.path.join(DS, 'S1Hand', f'{pid}_S1Hand.tif')) as s:
            sar = s.read().astype(np.float32)
        sar = np.nan_to_num(sar)
        sar = (sar - sar.mean(axis=(1, 2), keepdims=True)) / \
              (sar.std(axis=(1, 2), keepdims=True) + 1e-8)
        x = torch.from_numpy(np.ascontiguousarray(sar)).unsqueeze(0)
        with torch.no_grad():
            # UNet.forward already ends in Sigmoid — do NOT apply again
            return model(x).squeeze().numpy() > 0.5

    for tol in [0, 3, 8]:
        tp = fp = fn = nfp = npx = 0
        for pid in ids:
            p = predict(pid)
            t = label_array(pid) == 1
            if not t.any():
                nfp += int(p.sum()); npx += p.size
                continue
            if tol:
                td = ndimage.binary_dilation(t, iterations=tol)
                pd = ndimage.binary_dilation(p, iterations=tol)
            else:
                td, pd = t, p
            tp += int((pd & t).sum()); fp += int((pd & ~t).sum())
            fn += int((~pd & t).sum())
        iou = tp / (tp + fp + fn + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        print(f'tol={tol}: IoU={iou:.3f} recall={rec:.3f} '
              f'| FP px on flood-free patches={nfp} ({nfp/max(npx,1):.4f} of pixels)')


if __name__ == '__main__':
    tile_overlap()
    hand_geometry()
    perm_overlap()
    tolerant_metrics()
