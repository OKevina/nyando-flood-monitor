import os
import random
import csv

def generate_splits(dataset_dir, train_ratio=0.6, val_ratio=0.2):
    s1_dir = os.path.join(dataset_dir, "S1Hand")
    
    if not os.path.exists(s1_dir):
        print(f"Error: {s1_dir} does not exist. Run the pipeline first.")
        return
        
    # Get all patch filenames (without extensions)
    patches = []
    for f in os.listdir(s1_dir):
        if f.endswith("_S1Hand.tif"):
            # e.g., "Nyando_2024-04-17_1_S1Hand.tif" -> "Nyando_2024-04-17_1"
            base_name = f.replace("_S1Hand.tif", "")
            patches.append(base_name)
            
    if not patches:
        print("No patches found. Pipeline may have failed or filtered everything.")
        return
        
    print(f"Found {len(patches)} valid flood patches. Shuffling and splitting...")
    
    # Shuffle predictably for reproducibility
    random.seed(42)
    random.shuffle(patches)
    
    n_total = len(patches)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_patches = patches[:n_train]
    val_patches = patches[n_train:n_train+n_val]
    test_patches = patches[n_train+n_val:]
    
    print(f"Split sizes -> Train: {len(train_patches)}, Val: {len(val_patches)}, Test: {len(test_patches)}")
    
    # Write to CSV in Sen1Floods11 format
    def write_csv(split_name, patch_list):
        out_path = os.path.join(dataset_dir, f"nyando_split_{split_name}.csv")
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Sen1Floods11 format: just the ID, but no headers
            for p in patch_list:
                writer.writerow([p])
        print(f"Saved {out_path}")
        
    write_csv("train", train_patches)
    write_csv("val", val_patches)
    write_csv("test", test_patches)
    
    print("Dataset split generation complete!")

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "data", "Nyando_Dataset")
    generate_splits(dataset_dir)
