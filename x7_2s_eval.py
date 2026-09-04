import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from pathlib import Path

from x7_2s_train import X7DualSensorDataset, GestureCNN_X7_Dual, N_CLASSES, GESTURE_NAMES, BATCH_SIZE

def run_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureCNN_X7_Dual(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(r"models\best_model_2s.pth", map_location=device))
    model.eval()

    dataset = X7DualSensorDataset("gesture_dataset_2s_new", augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    def evaluate(mask1=True, mask2=True):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in loader:
                if not mask1: x[:, 0] = 0
                if not mask2: x[:, 1] = 0
                preds = torch.argmax(model(x.to(device)), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        return np.array(all_labels), np.array(all_preds)

    # 1. Baseline (Both Sensors)
    labels, preds = evaluate(True, True)
    acc_baseline = np.mean(labels == preds)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(labels, preds, labels=range(N_CLASSES))
    
    # 2. LOFO Sensor 1 (Zero out Sensor 1)
    _, preds_s2_only = evaluate(False, True)
    acc_s2_only = np.mean(labels == preds_s2_only)
    
    # 3. LOFO Sensor 2 (Zero out Sensor 2)
    _, preds_s1_only = evaluate(True, False)
    acc_s1_only = np.mean(labels == preds_s1_only)
    
    print("\n--- RESULTS ---")
    print(f"Overall Accuracy (Both Sensors): {acc_baseline*100:.2f}%")
    print(f"Accuracy w/o Sensor 1 (Sensor 2 Only): {acc_s2_only*100:.2f}%")
    print(f"Accuracy w/o Sensor 2 (Sensor 1 Only): {acc_s1_only*100:.2f}%")
    
    print("\n--- CONFUSION MATRIX ---")
    print("Row: True Class | Col: Predicted Class")
    header = "    " + " ".join([f"{i:2d}" for i in range(N_CLASSES)])
    print(header)
    for i in range(N_CLASSES):
        row_str = f"{i:2d}: " + " ".join([f"{cm[i, j]:4d}" for j in range(N_CLASSES)])
        print(row_str)

if __name__ == "__main__":
    run_eval()
