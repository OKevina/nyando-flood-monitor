import os
import re
import random
import csv
from collections import defaultdict

# Pipeline patches look like "Nyando_2024-04-17_1"; hand-labelled ones like
# "Nyando_17" carry no date, so each is treated as its own group.
_DATE_RE = re.compile(r"^Nyando_(\d{4}-\d{2}-\d{2})_\d+$")


def _group_key(patch_id):
    m = _DATE_RE.match(patch_id)
    return m.group(1) if m else patch_id


def generate_splits(dataset_dir, train_ratio=0.6, val_ratio=0.2):
    s1_dir = os.path.join(dataset_dir, "S1Hand")

    if not os.path.exists(s1_dir):
        print(f"Error: {s1_dir} does not exist. Run the pipeline first.")
        return

    # Get all patch filenames (without extensions)
    patches = []
    for f in os.listdir(s1_dir):
        if f.endswith("_S1Hand.tif"):
            base_name = f.replace("_S1Hand.tif", "")
            patches.append(base_name)

    if not patches:
        print("No patches found. Pipeline may have failed or filtered everything.")
        return

    # Group patches by acquisition date so overlapping patches from the same
    # scene never leak across train/val/test.
    groups = defaultdict(list)
    for p in patches:
        groups[_group_key(p)].append(p)

    print(f"Found {len(patches)} valid flood patches across {len(groups)} groups.")

    n_total = len(patches)
    targets = {
        "train": n_total * train_ratio,
        "val": n_total * val_ratio,
        "test": n_total * (1 - train_ratio - val_ratio),
    }
    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    # Phase 1: assign multi-patch groups (real dates) first — they must stay
    # together, so place each where it least overshoots its target.
    random.seed(42)
    blocks = sorted((m for m in groups.values() if len(m) > 1), key=len, reverse=True)
    for members in blocks:
        # least overshoot of target; break ties toward the least-filled split
        split = min(splits, key=lambda s: (
            (counts[s] + len(members)) / targets[s],
            counts[s] / targets[s],
        ))
        splits[split].extend(members)
        counts[split] += len(members)

    # Phase 2: deal single-patch groups to whichever split is furthest below
    # target, fine-tuning the sizes.
    singles = [m[0] for m in groups.values() if len(m) == 1]
    random.shuffle(singles)
    for p in singles:
        split = min(splits, key=lambda s: counts[s] / targets[s])
        splits[split].append(p)
        counts[split] += 1

    for name in ("train", "val", "test"):
        random.shuffle(splits[name])

    print(f"Split sizes -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print("Groups per split -> " + ", ".join(
        f"{name}: {len({_group_key(p) for p in ps})}" for name, ps in splits.items()))

    # Write to CSV in Sen1Floods11 format
    def write_csv(split_name, patch_list):
        out_path = os.path.join(dataset_dir, f"nyando_split_{split_name}.csv")
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for p in patch_list:
                writer.writerow([p])
        print(f"Saved {out_path}")

    write_csv("train", splits["train"])
    write_csv("val", splits["val"])
    write_csv("test", splits["test"])

    print("Dataset split generation complete!")


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "data", "Nyando_Dataset")
    generate_splits(dataset_dir)
