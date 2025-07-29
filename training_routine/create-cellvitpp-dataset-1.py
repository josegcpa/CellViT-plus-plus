import numpy as np
import yaml
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path
from skimage import io
from tqdm import trange

YAML_BASE_FILE = """
logging:
  mode: offline
  project: cellvit++
  notes: cellvit++
  log_comment: cellvit++
  wandb_dir: ./logs_local
  log_dir: ./logs_local
  level: Debug

sweep:
  method: bayes
  name: 
  project: classpose_benchmarking
  metric:
    goal: maximize
    name: AUROC/Validation
  run_cap: 100


random_seed: 19

gpu: 0

data:
  dataset: SegmentationDataset
  dataset_path: 
  normalize_stains_train: false
  normalize_stains_val: false
  num_classes: 
  train_filelist: 
  val_filelist: 
  label_map:
    

cellvit_path: ./checkpoints/CellViT-Virchow-x40-AMP.pth

model:
  parameters:
    hidden_dim:
      values: [128, 256, 512]

training:
  cache_cell_dataset: true
  batch_size: 256
  epochs: 50
  drop_rate: 0.1
  optimizer: AdamW
  optimizer_hyperparameter:
    betas: [0.85, 0.9]
    parameters:
      lr:
        min: 0.00001
        max: 0.01
      weight_decay:
        min: 0.00001
        max: 0.001
  early_stopping_patience: 20
  scheduler:
    parameters:
      scheduler_type:
        values: [constant, exponential]
  mixed_precision: true
  eval_every: 1
"""
    
def get_centers(inst_map: np.ndarray, type_map: np.ndarray) -> np.ndarray:
  x, y = np.where(inst_map > 0)
  v = inst_map[x, y].astype(int)
  c = type_map[x, y].astype(int)
  us = np.unique(v).astype(int)
  centers = []
  for u in us:
    centers.append([
          np.mean(y[v == u]).astype(int), 
          np.mean(x[v == u]).astype(int), 
          c[v == u].max() - 1
        ])
  return np.array(centers)
    

yaml_template = yaml.safe_load(YAML_BASE_FILE)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create CellViT++ dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory.")
    parser.add_argument("--sweep_name", type=str, required=True, help="Name of the sweep.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory.")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # define output paths
    output_dir = Path(args.output_dir)
    output_dirs = {
        "train_images": output_dir / "train" / "images",
        "train_labels": output_dir / "train" / "labels",
        "test_images": output_dir / "test" / "images",
        "test_labels": output_dir / "test" / "labels",
        "split": output_dir / "splits" / "fold_0",
        "train_configs": output_dir / "train_configs" / "ViT256",
        }
    for k in output_dirs:
        output_dirs[k].mkdir(exist_ok=True, parents=True)
    
    dataset_name = data_dir.name
    
    train_images = data_dir / "train" / "images.npy"
    train_labels = data_dir / "train" / "labels.npy"
    test_images = data_dir / "test" / "images.npy"
    test_labels = data_dir / "test" / "labels.npy"
    
    train_images = np.load(train_images, allow_pickle=True)
    train_labels = np.load(train_labels, allow_pickle=True)
    max_class = int(train_labels[..., 1].max())
    print("Number of classes:", max_class)
    ids = {"train": [], "test": []}
    idx = 0
    for i in trange(len(train_images)):
        curr_name = f"{dataset_name}_{idx}"
        io.imsave(output_dirs["train_images"] / f"{curr_name}.png", train_images[i])
        inst_map, type_map = train_labels[i][..., 0], train_labels[i][..., 1]
        inst_map = inst_map * np.where(type_map > 0, 1, 0)
        np.savez(
            output_dirs["train_labels"] / f"{curr_name}", 
            **{"inst_map": inst_map, "type_map": type_map}
        )
        
        centers = get_centers(inst_map, type_map)
        if len(centers) > 1:
          centers = np.stack(centers, axis=0)
          centers = "\n".join([",".join(map(str, c)) for c in centers])
        else:
          centers = ""
        with open(output_dirs["train_labels"] / f"{curr_name}.csv", "w") as f:
            f.write(centers)
            
        ids["train"].append(curr_name)
        idx += 1
    
    test_images = np.load(test_images, allow_pickle=True)
    test_labels = np.load(test_labels, allow_pickle=True)
    for i in trange(len(test_images)):
        curr_name = f"{dataset_name}_{idx}"
        io.imsave(output_dirs["test_images"] / f"{curr_name}.png", test_images[i])
        inst_map, type_map = test_labels[i][..., 0], test_labels[i][..., 1]
        inst_map = inst_map * np.where(type_map > 0, 1, 0)
        np.savez(
            output_dirs["test_labels"] / f"{curr_name}", 
            **{"inst_map": inst_map, "type_map": type_map}
        )
        
        centers = get_centers(inst_map, type_map)
        if len(centers) > 1:
          centers = np.stack(centers, axis=0)
          centers = "\n".join([",".join(map(str, c)) for c in centers])
        else:
          centers = ""
        with open(output_dirs["test_labels"] / f"{curr_name}.csv", "w") as f:
            f.write(centers)

        ids["test"].append(curr_name)
        idx += 1
    
    train_fold, val_fold = train_test_split(
        ids["train"],
        train_size=0.8,
        random_state=42,
    )
    

    with open(str(output_dirs["split"]/"train.csv"), "w") as f:
        for i in train_fold:
            f.write(f"{i}\n")
    with open(str(output_dirs["split"]/"val.csv"), "w") as f:
        for i in val_fold:
            f.write(f"{i}\n")
            
    yaml_template["data"]["dataset_path"] = str(output_dir.absolute())
    yaml_template["data"]["train_filelist"] = str(
        (output_dirs["split"] / "train.csv").absolute())
    yaml_template["data"]["val_filelist"] = str(
        (output_dirs["split"] / "val.csv").absolute())
    yaml_template["data"]["label_map"] = {
        i: f"Class {i}" for i in range(1, max_class + 1)
    }
    yaml_template["data"]["num_classes"] = int(max_class)
    yaml_template["sweep"]["name"] = args.sweep_name
    yaml_template["logging"]["name"] = args.sweep_name
    
    with open(output_dirs["train_configs"] / "fold_0.yaml", "w") as f:
        yaml.dump(yaml_template, f)
