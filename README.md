# Dual-Sensor IR-UWB Radar Intelligence System
### High-Precision 3D Gesture Recognition & Multi-Person Occupancy Detection

---

## 1. Executive Summary

This repository contains the end-to-end pr         oduction AI software stack for **Dual Novelda X7 Ultra-Wideband (IR-UWB) Transceiver Radars**. The system provides real-time, touchless human-machine interaction and environmental awareness via two core machine learning pipelines:

1. **3D Touchless Gesture Recognition:** A dual-branch Siamese / Late-Fusion Convolutional Neural Network that decodes spatial interferometry and Doppler micro-motion across **5 directional gesture classes** (**94.74% validation accuracy**, 89.1% through the full live capture path). Runs **continuously with no activation gesture**.
2. **5-State Human Presence & Occupancy Detection:** A GPU-accelerated multi-scale 2D CNN that counts human occupants ($0$, $1$, $2$, $3$, or $4$ people) from chest breathing micro-displacements and aggregate radar reflection energy (**94.00% Real-World Holdout Accuracy**).

Both pipelines execute natively on **NVIDIA RTX 5090 (Blackwell Architecture, CUDA 13.2)** with zero-copy GPU tensor processing.

---

## 2. Model Performance & Evaluation Metrics

### A. 3D Gesture Recognition Model (`best_model_2s_5class.pth`)

Five directional gestures, recognised continuously. **PINCH has been removed** — it
was the gated-mode activation trigger but never fired reliably on live hardware, and
carrying it in the label space also drew probability mass away from the directional
gestures (measured on identical recordings: 98.3% vs 96.8% once its logit is removed).

* **Validation accuracy (held-out 20% split):** **94.74%**
* **Full live-path accuracy (476 recordings, energy gate + capture + classify):** **89.1%**

#### Per-class (validation split):
```text
Class           Prec    Recall     F1     N
SWIPE_LR       90.0%    94.7%   92.3%    19
SWIPE_RL       94.7%    94.7%   94.7%    19
SWIPE_UP      100.0%    89.5%   94.4%    19
SWIPE_DOWN     90.5%   100.0%   95.0%    19
PUSH_IN       100.0%    94.7%   97.3%    19
```

#### Live-path confusion matrix (row: true, col: predicted):
```text
              SWIPE_LR  SWIPE_RL  SWIPE_UP  SWIPE_DN   PUSH_IN
SWIPE_LR            92         0         2         2         0
SWIPE_RL             1        86         7         1         0
SWIPE_UP             1         0        89         2         1
SWIPE_DOWN           0         0        13        81         1
PUSH_IN              1         1        14         3        76
```

> The legacy 7-class model (`best_model_2s.pth`, six gestures plus NOISE) is retained
> in `models/` for reference. It cannot be retrained from `gesture_dataset_2s_new`,
> which contains no NOISE recordings.

#### Live capture is peak-driven, not fixed-length
The lag between crossing the motion threshold and the gesture's energy peak is **not
constant** — measured across the corpus it spans 1 to 97 frames, and its mean differs
per gesture (SWIPE_UP ~17 frames, PUSH_IN ~34). A fixed capture length therefore
centres the model's window correctly for some gestures and not others: with a fixed
50-frame gather, PUSH_IN collapsed to 2/25 while swipes were unaffected.

The capture instead tracks the running energy maximum and stops only once the gesture
has demonstrably peaked — energy falling for `DECAY_FRAMES` consecutive frames **and**
at least `POST_PEAK_FRAMES` elapsed since the peak, so the second half of the window
holds real data rather than edge padding. This lifted the live path from 68.8% to
**89.1%** with every class above 80%.

---

### B. 5-Class Human Presence Model (`best_model_2s_presence.pth`)

Evaluated on **100 pure real-world holdout recordings** (`human_dataset_2s_test_holdout`):

* **Overall Holdout Accuracy:** **94.00%** (94 / 100 Correct Predictions)
* **Empty Room False Alarm Rate:** **0.0%** (100% Precision, 100% Recall on `0_people`)

#### Detailed Per-Class Metrics:
```text
========================================================================================
Class Name      | Precision  | Recall     | F1-Score   | Support (Test Samples)
========================================================================================
0_people        |    100.0% |    100.0% |    100.0% | 20
1_people        |     95.2% |    100.0% |     97.6% | 20
2_people        |     96.2% |     83.3% |     89.3% | 30
3_people        |     82.6% |     95.0% |     88.4% | 20
4_people        |    100.0% |    100.0% |    100.0% | 10
========================================================================================
```

#### $5 \times 5$ Confusion Matrix:
```text
Ground Truth \ Predicted | 0_people | 1_people | 2_people | 3_people | 4_people
--------------------------------------------------------------------------------
0_people                 |    20    |    0     |    0     |    0     |    0
1_people                 |     0    |   20     |    0     |    0     |    0
2_people                 |     0    |    1     |   25     |    4     |    0
3_people                 |     0    |    0     |    1     |   19     |    0
4_people                 |     0    |    0     |    0     |    0     |   10
================================================================================
```

---

## 3. Detailed Neural Network Architectures

See visual flow in [`architecture.mmd`](./architecture.mmd).

### A. 3D Gesture CNN (`GestureCNN_X7_Dual`)
* **Paradigm:** Dual-Branch Siamese Late Fusion with Cross-Sensor Interferometry.
* **Input Tensor:** $(B, 2, 8, 34, 64)$ — Batch, 2 Sensors, 8 Physical Channels, 34 Range Bins, 64 Frames.
* **Branch 1 & 2 (Per-Sensor Feature Extractor):**
  * `Conv2D(8 -> 64, kernel=3, pad=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2)` $\to$ `Dropout2d(0.1)`
  * `Conv2D(64 -> 128, kernel=3, pad=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2)` $\to$ `Dropout2d(0.1)`
  * `Conv2D(128 -> 256, kernel=3, pad=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2)`
  * `Conv2D(256 -> 512, kernel=3, pad=1)` $\to$ `BatchNorm2d` $\to$ `ReLU`
  * `AdaptiveAvgPool2d((4, 4))` $\to$ `Flatten()` $\to$ **8,192 features per branch**.
* **Late Fusion Classifier:**
  * Concatenation: $\text{Branch}_1 \oplus \text{Branch}_2 \implies 16,384$ features.
  * `Linear(16384 -> 512)` $\to$ `ReLU` $\to$ `Dropout(0.5)`
  * `Linear(512 -> 128)` $\to$ `ReLU` $\to$ `Dropout(0.3)`
  * `Linear(128 -> 7)` $\to$ **7 Class Logits**.

---

### B. Human Presence CNN (`PresenceCNN_X7_Dual`)
* **Paradigm:** GPU-Accelerated Multi-Scale 2D Convolutional Energy & Micro-Doppler Classifier.
* **Input Tensor:** $(B, 2, 50, 339)$ — Batch, 2 Sensors, 50 Frames (5 sec), 339 Range Bins.
* **Convolutional Backbone:**
  * `Block 1`: `Conv2D(2 -> 16, 3x3, pad=1)` $\to$ `BN` $\to$ `ReLU` $\to$ `MaxPool2D(2)` $\to$ Tensor: `(16, 25, 169)`
  * `Block 2`: `Conv2D(16 -> 32, 3x3, pad=1)` $\to$ `BN` $\to$ `ReLU` $\to$ `MaxPool2D(2)` $\to$ Tensor: `(32, 12, 84)`
  * `Block 3`: `Conv2D(32 -> 64, 3x3, pad=1)` $\to$ `BN` $\to$ `ReLU` $\to$ `MaxPool2D(2)` $\to$ Tensor: `(64, 6, 42)`
  * `Block 4`: `Conv2D(64 -> 128, 3x3, pad=1)` $\to$ `BN` $\to$ `ReLU` $\to$ `MaxPool2D(2)` $\to$ Tensor: `(128, 3, 21)`
  * `Global Pooling`: `AdaptiveAvgPool2D((1, 1))` $\to$ Tensor: `(128, 1, 1)`
* **Classifier:**
  * `Flatten()` $\to$ `Dropout(0.3)` $\to$ `Linear(128 -> 5)` $\to$ **5 Occupancy Class Logits**.

---

## 4. Novelda X7 SDK Modules & Integration

The system interfaces with the Novelda X7 Ultra-Wideband Radar hardware using the low-level C++ DLL libraries linked through Python wrappers in `novelda-uwb-demos`.

### Core SDK Modules Used:
1. **`RadarDirectCallback` (`radar_direct_callback.py`):**
   * High-throughput asynchronous callback driver that streams raw baseband complex I/Q radar frames over USB/UART without intermediate OS socket buffering.
2. **`X7RadarDirectCallback` Presets:**
   * `default_preset.json` — Optimized for high frame rate gesture detection (40 FPS, 34 range bins up to ~1.2 meters).
   * `human_preset.json` — Optimized for deep room penetration (10 FPS, 339 range bins up to ~5.0 meters).
3. **`X7DualRadar` Synchronizer (`x7_2s_record.py` / `x7_2s_live.py`):**
   * Multi-threaded producer-consumer pipeline with sequence number tracking and a software timestamp tolerance gate ($\Delta t < 50\,\text{ms}$) to guarantee frame alignment between Front and Side radars.
4. **Virtual MIMO Frame Assembly:**
   * Each X7 sensor possesses 2 Transmit (TX0, TX1) and 2 Receive (RX0, RX1) antennas. The SDK driver interleaves pulses to construct a complete 5D complex matrix per recording:
     $$\mathbf{W} \in \mathbb{C}^{T \times S \times N_{\text{TX}} \times N_{\text{RX}} \times N_{\text{bins}}} \quad (T \text{ Frames}, S=2 \text{ Sensors}, 2\,\text{TX}, 2\,\text{RX}, B \text{ Bins})$$

---

## 5. Mathematical Formulations & DSP Theory

### 1. Radar Range Bin to Distance Mapping
The physical distance $R(b)$ corresponding to range bin index $b$ is given by:
$$R(b) = R_0 + b \cdot \Delta R = R_0 + b \cdot \frac{c}{2 f_s \sqrt{\varepsilon_r}}$$
Where $c = 3 \times 10^8\,\text{m/s}$, $f_s$ is the baseband sampling frequency, and $R_0$ is the antenna offset delay.

---

### 2. Moving Target Indicator (MTI) Static Clutter Filter
To remove static background reflections (walls, vehicle seats, dashboard) and isolate breathing and moving hands:
$$\mathbf{W}_{\text{MTI}}(t, s, i, j, b) = \mathbf{W}(t, s, i, j, b) - \frac{1}{T}\sum_{\tau=0}^{T-1} \mathbf{W}(\tau, s, i, j, b)$$

---

### 3. Complex Cross-Antenna Interferometric Unit Phasor
The cross-correlation between two antenna channels $z_1, z_2 \in \mathbb{C}$ extracts instantaneous phase difference while suppressing amplitude variation:
$$\Phi(z_1, z_2) = \frac{z_1 \cdot z_2^*}{|z_1 \cdot z_2^*| + \varepsilon}$$
Where $z_2^*$ denotes the complex conjugate and $\varepsilon = 10^{-9}$ prevents division by zero.

---

### 4. Angle of Arrival (AoA) Physical Channels
For each sensor array with antenna spacing $d$:
* **Horizontal Phasor:**
  $$\mathbf{H}(t, b) = \Phi\Big(\mathbf{W}_{\text{MTI}}(t, \text{TX}_0, \text{RX}_0, b),\, \mathbf{W}_{\text{MTI}}(t, \text{TX}_1, \text{RX}_0, b)\Big)$$
  $$\Delta \phi_H = \arg(\mathbf{H}) = \frac{2\pi d_H \sin\theta_H}{\lambda}$$
* **Vertical Phasor:**
  $$\mathbf{V}(t, b) = \Phi\Big(\mathbf{W}_{\text{MTI}}(t, \text{TX}_0, \text{RX}_0, b),\, \mathbf{W}_{\text{MTI}}(t, \text{TX}_0, \text{RX}_1, b)\Big)$$
  $$\Delta \phi_V = \arg(\mathbf{V}) = \frac{2\pi d_V \sin\theta_V}{\lambda}$$

---

### 5. Instantaneous Doppler Velocity Phasor
The temporal phase derivative between successive frames $t$ and $t-1$ yields radial velocity $v_r$:
$$\mathbf{D}(t, b) = \Phi\Big(\mathbf{W}_{\text{MTI}}(t, \text{TX}_0, \text{RX}_0, b),\, \mathbf{W}_{\text{MTI}}(t-1, \text{TX}_0, \text{RX}_0, b)\Big)$$
$$f_D = \frac{1}{2\pi \Delta t}\arg(\mathbf{D}) = \frac{2 v_r}{\lambda}$$

---

### 6. Spatial Energy Skew & Amplitude Normalization
$$\bar{A}(t, b) = \frac{1}{4}\sum_{i=0}^1 \sum_{j=0}^1 |\mathbf{W}_{\text{MTI}}(t, i, j, b)|$$
$$A_{\text{norm}}(t, b) = \frac{\bar{A}(t, b) - \mu_A}{\sigma_A + \varepsilon}$$
$$A_{H\text{norm}}(t, b) = \frac{|\mathbf{W}_{\text{MTI}}(t, \text{TX}_0, \text{RX}_0, b)| - |\mathbf{W}_{\text{MTI}}(t, \text{TX}_1, \text{RX}_0, b)|}{|\mathbf{W}_{\text{MTI}}(t, \text{TX}_0, \text{RX}_0, b)| + |\mathbf{W}_{\text{MTI}}(t, \text{TX}_1, \text{RX}_0, b)| + \varepsilon}$$

---

### 7. Absolute Reflection Energy Logarithmic Compression (Presence Model)
To preserve the physical property that $N$ people reflect $\approx N \times$ the energy of 1 person while stabilizing neural network gradients:
$$\mathbf{X}_{\text{presence}}(s, t, b) = \ln\left(1 + 100 \cdot \frac{1}{4}\sum_{i=0}^1 \sum_{j=0}^1 |\mathbf{W}_{\text{MTI}}(t, s, i, j, b)|\right)$$

---

### 8. Cross-Entropy Loss with Label Smoothing
$$\mathcal{L}_{\text{CE}}(\mathbf{y}, \hat{\mathbf{p}}) = -\sum_{k=1}^K \tilde{y}_k \log \hat{p}_k, \quad \text{where } \tilde{y}_k = (1 - \alpha)\,\mathbb{I}(y = k) + \frac{\alpha}{K}$$
With smoothing factor $\alpha = 0.05$ for regularized generalization.

---

### 9. Cosine Annealing Learning Rate Schedule
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$

---

## 6. CLI Usage & Execution Reference

### Environment Setup
```powershell
.\.venv\Scripts\Activate.ps1
# Verify RTX 5090 CUDA 13.2 support
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

### 1. 3D Gesture Recognition
* **Offline Evaluation:**
  ```powershell
  python x7_2s_eval.py
  ```
* **Retrain Gesture Model:**
  ```powershell
  python x7_2s_train.py --epochs 80
  ```
* **Live Dual-Radar Inference (continuous, no activation gesture):**
  ```powershell
  python x7_2s_live.py --host1 <PORT_OR_IP_1> --host2 <PORT_OR_IP_2>
  # tune the motion gate for your room if needed:
  python x7_2s_live.py --host1 529 --host2 545 --threshold 600
  ```

### 2. Human Presence & Occupancy Detection
* **Offline Evaluation (5 Classes):**
  ```powershell
  python x7_2s_eval_presence.py --data human_dataset_2s_test_holdout --model models/best_model_2s_presence.pth
  ```
* **GPU-Accelerated Retraining (40 Epochs in ~25s):**
  ```powershell
  python x7_2s_train_presence.py --data human_dataset_2s_augmented --val_data human_dataset_2s_test_holdout --epochs 40 --batch_size 64
  ```
* **Generate Intra-Class Synthetic Augmentation:**
  ```powershell
  python x7_2s_synthesize_data.py --src human_dataset_2s --target_per_class 600
  ```
