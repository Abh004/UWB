"""
x7_2s_eval_presence.py
======================
Evaluates the trained dual-sensor presence CNN on the pure real-world holdout dataset.
Outputs:
- Overall Accuracy
- Per-class Precision, Recall, F1-Score
- Complete Confusion Matrix
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from x7_2s_train_presence import (
    X7DualPresenceDataset,
    PresenceCNN_X7_Dual,
    extract_presence_features,
    BATCH_SIZE
)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("Dual-Sensor Presence Model Offline Evaluation")
    print("==================================================")
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Test Dataset Path  : {args.data}")
    print(f"Model Weights Path : {args.model}")
    print("--------------------------------------------------")

    classes_json = os.path.join(os.path.dirname(args.model), "presence_classes.json")
    class_names = None
    if os.path.exists(classes_json):
        with open(classes_json, "r") as f:
            class_names = json.load(f)

    test_dataset = X7DualPresenceDataset(args.data, classes=class_names)
    if len(test_dataset) == 0:
        print(f"[ERROR] No data found in {args.data}.")
        return

    class_names = test_dataset.classes
    n_classes = len(class_names)

    model = PresenceCNN_X7_Dual(n_classes=n_classes).to(device)
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file {args.model} does not exist. Please train the model first.")
        return

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"[INFO] Successfully loaded model weights from {args.model}")

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=(device.type == "cuda")
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for raw_x, y in test_loader:
            raw_x = raw_x.to(device, non_blocking=True)
            feat_x = extract_presence_features(raw_x)
            outputs = model(feat_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            conf_matrix[t, p] += 1

    total_samples = len(all_targets)
    correct_samples = int(np.sum(all_preds == all_targets))
    overall_acc = correct_samples / total_samples if total_samples > 0 else 0

    print("\n==================================================")
    print("EVALUATION RESULTS (Holdout Real Dataset)")
    print("==================================================")
    print(f"Total Test Samples: {total_samples}")
    print(f"Correctly Predicted: {correct_samples}")
    print(f"Overall Accuracy  : {overall_acc * 100:.2f}%\n")

    print("-------------------------------------------------------------------------")
    print(f"{'Class Name':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-------------------------------------------------------------------------")

    for i, cname in enumerate(class_names):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        support = np.sum(conf_matrix[i, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{cname:<15} | {precision*100:8.1f}% | {recall*100:8.1f}% | {f1*100:8.1f}% | {support:<8}")

    print("-------------------------------------------------------------------------")
    print("\nCONFUSION MATRIX (Rows: Ground Truth, Cols: Predicted):")
    header = f"{'':<15} | " + " | ".join([f"{c:<10}" for c in class_names])
    print(header)
    print("-" * len(header))
    for i, cname in enumerate(class_names):
        row_str = f"{cname:<15} | " + " | ".join([f"{conf_matrix[i, j]:<10}" for j in range(n_classes)])
        print(row_str)
    print("==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Presence CNN Model")
    parser.add_argument("--data", default="human_dataset_2s_test_holdout", help="Holdout test dataset directory")
    parser.add_argument("--model", default=os.path.join("models", "best_model_2s_presence.pth"), help="Path to saved model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Inference batch size")
    args = parser.parse_args()
    run(args)
