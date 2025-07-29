#!/usr/bin/env python
"""
Compute Panoptic Quality (PQ) metrics between ground truth and predicted masks.

This script calculates:
- PQ (Panoptic Quality): Combination of DQ and SQ (PQ = DQ * SQ)
- DQ (Detection/Recognition Quality): F1-score measuring how well instances are detected
- SQ (Segmentation Quality): IoU measuring the quality of segmentation for matched instances
- TP (True Positives): Number of correctly detected instances
- FP (False Positives): Number of predicted instances without ground truth matches
- FN (False Negatives): Number of ground truth instances without predicted matches

Usage:
    python compute_pq_metrics.py --gt_path /path/to/ground_truth_masks --pred_path /path/to/predicted_masks
    
    Optional arguments:
    --match_iou MATCH_IOU   IoU threshold for matching instances (default: 0.5)
    --output OUTPUT         Path to save results as CSV (default: None)
    --binary                If set, treat masks as binary instance segmentation without classes
    --nr_classes NR_CLASSES Number of classes for multi-class PQ calculation (default: 6)
"""

import argparse
import glob
import numpy as np
import os
import pandas as pd
from itertools import product
from tqdm import tqdm
from logging import getLogger

logger = getLogger(__name__)

# Use the existing implementation from the CoNIC metrics
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .stats_utils import get_pq, get_multi_pq_info
from .utils import remap_label


def load_masks(path: str) -> np.ndarray:
    """
    Load masks from a directory or file in npy or npz format.

    Args:
        path (str): Path to the directory or file.
    Returns:
        np.ndarray: Loaded masks.
    Raises:
        ValueError: If no .npy or .npz files are found in the directory.
    """
    if os.path.isdir(path):
        # If a directory, load all .npy or .npz files
        mask_files = sorted(glob.glob(os.path.join(path, "*.np[yz]")))
        if not mask_files:
            raise ValueError(f"No .npy or .npz files found in {path}")

        # Load the first file to determine format
        first_mask = np.load(mask_files[0], allow_pickle=True)
        if isinstance(first_mask, np.ndarray):
            # Single array per file format
            return [np.load(f, allow_pickle=True) for f in mask_files]
        else:
            # npz format with multiple arrays
            return [np.load(f, allow_pickle=True)["arr_0"] for f in mask_files]
    else:
        # If a single file
        if path.endswith(".npy"):
            return np.load(path, allow_pickle=True)
        elif path.endswith(".npz"):
            return np.load(path, allow_pickle=True)["arr_0"]
        else:
            raise ValueError(f"Unsupported file format: {path}")


def check_and_coherce_if_necessary(masks, expected_shape_length):
    """
    Check if masks have the expected shape and coherce if necessary.

    Args:
        masks (np.ndarray | list[np.ndarray]): Masks to check.
        expected_shape_length (int): Expected number of dimensions for an individual
            mask.

    Returns:
        np.ndarray: Checked and coherced masks.
    Raises:
        ValueError: If masks does not have the expected shape or a shape cohercible
            to that shape.
    """
    if isinstance(masks, np.ndarray) and masks.dtype == "object":
        return list(masks)

    if isinstance(masks, list):
        return masks

    if len(masks.shape) == expected_shape_length:
        masks = masks[None]
    elif len(masks.shape) != (expected_shape_length + 1):
        raise ValueError(
            f"Masks have {len(masks.shape)} dimensions, expected {expected_shape_length}"
        )
    return masks

def filter_out_unlabelled_cells(gt_masks: np.ndarray | list[np.ndarray], pred_masks: np.ndarray | list[np.ndarray], min_iou: float = 0.5) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    """
    Filter out unlabelled cells from the ground truth and predicted masks.
    
    Args:
        gt_masks (np.ndarray | list[np.ndarray]): Ground truth masks.
        pred_masks (np.ndarray | list[np.ndarray]): Predicted masks.
        min_iou (float, optional): IoU threshold for matching. Defaults to 0.5.
    
    Returns:
        tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]: Filtered ground truth and predicted masks.
    """
    
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        intersection = np.sum(a * b)
        union = np.sum(a) + np.sum(b) - intersection
        if union == 0:
            return 0
        return intersection / union
    
    for i in range(len(gt_masks)):
        gt_mask, pred_mask = gt_masks[i], pred_masks[i]
        gt_instances, pred_instances = remap_label(gt_mask[..., 0]), remap_label(pred_mask[..., 0])
        
        # Check if there are any instances to process
        gt_max = gt_instances.max()
        pred_max = pred_instances.max()
        
        # Skip processing if there are no instances in either mask
        if gt_max <= 0 or pred_max <= 0:
            # Still update the masks even if skipping processing
            gt_masks[i] = gt_mask
            pred_masks[i] = pred_mask
            continue
            
        # Skip processing if all instances are labeled
        gt_instance_count = gt_max  
        
        gt_labeled_instances = np.unique(gt_instances * (gt_mask[..., 1] > 0))
        gt_labeled_count = len(gt_labeled_instances[gt_labeled_instances > 0])
        
        if gt_instance_count == gt_labeled_count:
            gt_masks[i] = gt_mask
            pred_masks[i] = pred_mask
            continue
            
        dsc_matrix = np.zeros((gt_max - 1, pred_max - 1))
        for gt_id, pred_id in product(range(1, gt_max), range(1, pred_max)):
            dsc_matrix[gt_id - 1, pred_id - 1] = iou(gt_instances == gt_id, pred_instances == pred_id)
        gt_has_label = np.unique(gt_instances * (gt_mask[..., 1] > 0))
        gt_has_label = gt_has_label[gt_has_label > 0]
        gt_ids, pred_ids = np.where(dsc_matrix > min_iou)
        gt_ids, pred_ids = gt_ids + 1, pred_ids + 1
        remove_gt = []
        remove_pred = []
        for gt_id, pred_id in zip(gt_ids, pred_ids):
            if gt_id not in gt_has_label:
                remove_gt.append(gt_id)
                remove_pred.append(pred_id)
        remove_gt = np.unique(remove_gt)
        remove_pred = np.unique(remove_pred)
        gt_mask[np.isin(gt_instances, remove_gt)] = 0
        pred_mask[np.isin(pred_instances, remove_pred)] = 0
        gt_mask[..., 0] = remap_label(gt_mask[..., 0])
        pred_mask[..., 0] = remap_label(pred_mask[..., 0])
        
        gt_masks[i] = gt_mask
        pred_masks[i] = pred_mask
    
    return gt_masks, pred_masks

def compute_binary_pq_metrics(
    gt_masks: np.ndarray | list[np.ndarray],
    pred_masks: np.ndarray | list[np.ndarray],
    match_iou: float = 0.5,
) -> pd.DataFrame:
    """
    Compute binary PQ metrics for a batch of masks. Expects both input
    masks to have shapes HxW.

    Args:
        gt_masks (np.ndarray | list[np.ndarray]): Ground truth masks.
        pred_masks (np.ndarray | list[np.ndarray]): Predicted masks.
        match_iou (float, optional): IoU threshold for matching. Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame with PQ metrics.
    """
    results = []

    # The expected shape has to be 3 if we want to use
    gt_masks = check_and_coherce_if_necessary(gt_masks, 2)
    pred_masks = check_and_coherce_if_necessary(pred_masks, 2)

    for i in tqdm(range(len(gt_masks)), desc="Computing metrics"):
        gt = gt_masks[i]
        pred = pred_masks[i]

        # Ensure masks have proper instance IDs (contiguous)
        gt = remap_label(gt)
        pred = remap_label(pred)

        # Get PQ metrics
        pq_stats, counts, iou_sum = get_pq(
            gt, pred, match_iou=match_iou, remap=False
        )
        dq, sq, pq = pq_stats
        tp, fp, fn = counts

        results.append(
            {
                "image_id": i,
                "pq": pq,
                "dq": dq,
                "sq": sq,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
                "f1": (2 * tp) / (2 * tp + fp + fn),
                "iou_sum": iou_sum,
            }
        )

    return pd.DataFrame(results)


def compute_multiclass_pq_metrics(
    gt_masks: np.ndarray | list[np.ndarray],
    pred_masks: np.ndarray | list[np.ndarray],
    match_iou: float = 0.5,
    nr_classes: int = 6,
) -> pd.DataFrame:
    """
    Compute multi-class PQ metrics for a batch of masks. Expects both input
    masks to have shapes HxWx2, where the first channel is the instance
    segmentation mask and the second channel is the class mask.

    Args:
        gt_masks (np.ndarray | list[np.ndarray]): Ground truth masks.
        pred_masks (np.ndarray | list[np.ndarray]): Predicted masks.
        match_iou (float, optional): IoU threshold for matching. Defaults to 0.5.
        nr_classes (int, optional): Number of classes. Defaults to 6.

    Returns:
        pd.DataFrame: DataFrame with PQ metrics.
    """
    # Initialize arrays to store aggregated stats
    tp_per_class = np.zeros(nr_classes)
    fp_per_class = np.zeros(nr_classes)
    fn_per_class = np.zeros(nr_classes)
    iou_sum_per_class = np.zeros(nr_classes)

    gt_masks = check_and_coherce_if_necessary(gt_masks, 3)
    pred_masks = check_and_coherce_if_necessary(pred_masks, 3)
    
    gt_masks, pred_masks = filter_out_unlabelled_cells(gt_masks, pred_masks)

    for i in tqdm(range(len(gt_masks)), desc="Computing metrics"):
        gt = gt_masks[i]
        pred = pred_masks[i]

        # Get multi-class PQ info
        pq_info = get_multi_pq_info(
            gt, pred, nr_classes=nr_classes, match_iou=match_iou
        )

        # Aggregate stats for each class
        for class_idx in range(nr_classes):
            tp_per_class[class_idx] += pq_info[class_idx][0]
            fp_per_class[class_idx] += pq_info[class_idx][1]
            fn_per_class[class_idx] += pq_info[class_idx][2]
            iou_sum_per_class[class_idx] += pq_info[class_idx][3]

    # Calculate PQ metrics for each class
    results = []
    for class_idx in range(nr_classes):
        tp = tp_per_class[class_idx]
        fp = fp_per_class[class_idx]
        fn = fn_per_class[class_idx]
        iou_sum = iou_sum_per_class[class_idx]

        # Calculate DQ (Detection Quality)
        dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)

        # Calculate SQ (Segmentation Quality)
        sq = iou_sum / (tp + 1.0e-6)

        # Calculate PQ (Panoptic Quality)
        pq = dq * sq

        results.append(
            {
                "class_id": class_idx + 1,
                "pq": pq,
                "dq": dq,
                "sq": sq,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
                "f1": (2 * tp) / (2 * tp + fp + fn),
                "iou_sum": iou_sum,
            }
        )

    # Calculate average metrics across all classes
    avg_results = {
        "class_id": "avg",
        "pq": np.nanmean([r["pq"] for r in results]),
        "dq": np.nanmean([r["dq"] for r in results]),
        "sq": np.nanmean([r["sq"] for r in results]),
        "tp": np.nansum([r["tp"] for r in results]),
        "fp": np.nansum([r["fp"] for r in results]),
        "fn": np.nansum([r["fn"] for r in results]),
        "precision": np.nanmean([r["precision"] for r in results]),
        "recall": np.nanmean([r["recall"] for r in results]),
        "f1": np.nanmean([r["f1"] for r in results]),
        "iou_sum": np.nansum([r["iou_sum"] for r in results]),
    }

    results.append(avg_results)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compute PQ (Panoptic Quality) metrics between ground truth and predicted masks."
    )
    parser.add_argument(
        "--gt_path",
        required=True,
        help="Path to ground truth masks (directory or file)",
    )
    parser.add_argument(
        "--pred_path",
        required=True,
        help="Path to predicted masks (directory or file)",
    )
    parser.add_argument(
        "--match_iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching instances",
    )
    parser.add_argument("--output", help="Path to save results as CSV")
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Treat masks as binary instance segmentation without classes",
    )
    parser.add_argument(
        "--nr_classes",
        type=int,
        default=6,
        help="Number of classes for multi-class PQ calculation",
    )
    parser.add_argument(
        "--ignore_classes",
        type=int,
        default=None,
        nargs="+",
        help="Classes to ignore.",
    )

    args = parser.parse_args()

    logger.info(f"Loading ground truth masks from {args.gt_path}")
    gt_masks = load_masks(args.gt_path)

    logger.info(f"Loading predicted masks from {args.pred_path}")
    pred_masks = load_masks(args.pred_path)
    
    if args.ignore_classes:
        for i in args.ignore_classes:
            gt_masks[..., 1][gt_masks[..., 1] == i] = 0
            pred_masks[..., 1][pred_masks[..., 1] == i] = 0
    # Check that masks have the same shape
    if isinstance(gt_masks, list) and isinstance(pred_masks, list):
        if len(gt_masks) != len(pred_masks):
            raise ValueError(
                f"Number of ground truth masks ({len(gt_masks)}) doesn't match predicted masks ({len(pred_masks)})"
            )
    elif gt_masks.shape != pred_masks.shape:
        raise ValueError(
            f"Ground truth mask shape {gt_masks.shape} doesn't match predicted mask shape {pred_masks.shape}"
        )

    # Compute metrics
    if args.binary:
        logger.info(
            f"Computing binary PQ metrics with IoU threshold {args.match_iou}"
        )
        results = compute_binary_pq_metrics(
            gt_masks, pred_masks, match_iou=args.match_iou
        )
    else:
        logger.info(
            f"Computing multi-class PQ metrics with IoU threshold {args.match_iou} for {args.nr_classes} classes"
        )
        results = compute_multiclass_pq_metrics(
            gt_masks,
            pred_masks,
            match_iou=args.match_iou,
            nr_classes=args.nr_classes,
        )

    # Print results
    print("\nResults:")
    print(results.to_string(index=False))

    # Save results if output path is provided
    if args.output:
        results.to_csv(args.output, index=False)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
