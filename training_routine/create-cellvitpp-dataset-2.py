import numpy as np
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create NPY from NPZ.")
    
    parser.add_argument("--output_dir", required=True)
    
    args = parser.parse_args()
    
    for npz_path in tqdm(Path(args.output_dir).rglob("*.npz")):
        npz = np.load(npz_path)
        inst_map = npz["inst_map"]
        type_map = npz["type_map"]
        np.save(str(npz_path).replace(".npz", ".npy"), 
                {"inst_map": inst_map, "type_map": type_map},
                allow_pickle=True)