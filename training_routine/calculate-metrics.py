import json
import numpy as np
import yaml
import os
import pandas as pd
from pathlib import Path
from skimage import draw
from pq_metrics.compute_pq_metrics import compute_multiclass_pq_metrics

def cell_pred_dict_to_prediction_array(
    cell_pred_dict: dict, 
    original_images: dict[str, np.ndarray]
    )->tuple[list[np.ndarray], list[np.ndarray]]:

    all_labels = []
    all_predictions = []
    for key in cell_pred_dict:
        assert key in original_images
        shape = original_images[key].shape[:2]
        pred = np.zeros((*shape, 2), dtype=np.uint8)

        for cell_id, cell_info in cell_pred_dict[key].items():
            contour = np.array(cell_info["contour"])
            rr, cc = draw.polygon(contour[:, 1], contour[:, 0])
            rr = np.clip(rr, 0, shape[0] - 1)
            cc = np.clip(cc, 0, shape[1] - 1)
            pred[rr, cc, 0] = cell_id
            pred[rr, cc, 1] = cell_info["type"] + 1

        # io.imsave("pred.png", np.uint8(255 * color.label2rgb(pred[..., 0], bg_label=0)))
        # io.imsave("label.png", np.uint8(255 * color.label2rgb(original_images[key][..., 0], bg_label=0)))
        all_predictions.append(pred.astype(int))
        all_labels.append(original_images[key].astype(int))

    return all_predictions, all_labels

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    path = Path(args.logdir)
    test_path = path / "test_results"
    cell_pred_dict_path = test_path / "cell_pred_dict.json"
    confusion_matrix_path = test_path / "confusion_matrix_summary.csv_matrix.csv"
    config_path = path / "config.yaml"

    with open(cell_pred_dict_path, "r") as f:
        cell_pred_dict = json.load(f)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(confusion_matrix_path, "r") as f:
        confusion_matrix = [list(map(int, x.split(","))) for x in f.readlines()]
        confusion_matrix = np.array(confusion_matrix)

    support = np.sum(confusion_matrix, axis=1)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / support
    f1 = 2 * precision * recall / (precision + recall)
    df_cellvit = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    df_cellvit.index = [i for i in range(1, len(precision) + 1)]
    df_cellvit.loc["avg", :] = df_cellvit.mean(axis=0)
    df_cellvit.loc["avg", "support"] = np.sum(support)
    df_cellvit.to_csv(os.path.join(args.output_path, "metrics_cellvit.csv"))

    test_path = Path(config["data"]["dataset_path"]) / "test" / "labels"
    test_images = {
        p.name.replace(".npy", ""): np.load(p, allow_pickle=True) 
        for p in test_path.glob("*.npy")}

    n_classes = 0
    for k in test_images:
        test_images[k] = test_images[k].item()
        n_classes = max(n_classes, test_images[k]["type_map"].max())
        test_images[k] = np.stack(
            [test_images[k]["inst_map"], 
             test_images[k]["type_map"]], axis=-1)

    n_classes = int(n_classes)

    predictions, labels = cell_pred_dict_to_prediction_array(
        cell_pred_dict,
        test_images,
    )

    results = compute_multiclass_pq_metrics(
        labels,
        predictions,
        match_iou=0.5,
        nr_classes=n_classes,
    ).fillna(0)

    results.to_csv(os.path.join(args.output_path, "metrics.csv"))
    np.save(os.path.join(args.output_path, "predictions.npy"), predictions)
    np.save(os.path.join(args.output_path, "labels.npy"), labels)
