# Dual-Sensor IR-UWB Radar Intelligence System
### Occupancy Detection and 3D Gesture Recognition with Dual Novelda X7 Radars

*Last updated: 2026-10-12. Engineering history in [`documentation/`](./documentation/).*

---

## 1. Overview

Two Novelda X7 IR-UWB radars (2 TX x 2 RX each) drive two pipelines:

| pipeline | what it does | status |
|---|---|---|
| **Occupancy (zone model)** | occupied / empty, front-row and back-row occupancy, from breathing and motion | **Proof of concept validated across days.** Occupied/empty and front row: AUC 1.000, 0% false alarm on a held-out day. Back row: indicator only. Per-person counting: not supported by this sensor geometry. |
| **3D gesture recognition** | 5 directional gestures + "no gesture", continuous, no activation gesture | **v4 "pause, then swipe", Nov 19.** Deployed `models/gesture_v4_pause.json` (11 sessions): 84.5% held-out accuracy, 91.1% precision with 61.3% of gestures recognised; left and right confused with each other 1/54 held out. Hold the hand still at the start, then swipe (Section 4.5). Left/right still needs more live-trigger recordings. `gesture_v3` (natural wind-up style) is kept for reference. |

**The proof of concept is recorded in a room with car-like seating** (2 front,
2 back, front row ~0.7-1.0 m from the radar), with four occupants (2 men,
2 women). The sensors are fixed **15 cm apart**, because the gesture pipeline
needs that mounting.

**Safety policy for occupancy:** a missed occupant is treated as worse than a
false alarm. Thresholds target 99% recall; the system never reports `EMPTY`
while warming up, while a sensor looks faulty, or while the setup appears to
have moved.

### What changed since the first version of this README
The earlier README reported a **5-class occupant-count CNN at 94% accuracy**.
That result did not survive testing: each occupancy level had been recorded on
its own day, and the same architecture that scored 99.7% on those recordings
scored **45%** on a new day (a classifier fed only range bins where nobody sits
also scored 99.7%). It had learnt the recording session, not the occupants. The
count CNN has been replaced by the zone model below; see
`documentation/2026-09-21.md`, `2026-09-24.md` and `2026-09-25.md`.

---

## 2. Hardware, Range Map and Acquisition

### 2.1 Radar configuration

| | gesture | occupancy |
|---|---|---|
| preset | `default_preset.json` | `human_preset.json`, FPS injected at runtime |
| frame rate | 40 FPS | **5 FPS** (`IterationsPerFrame` = 700 x 10 / FPS = 1400, +3 dB vs 10 FPS) |
| range bins used | first 34 delivered, physical bins 20-33 (+0.05 .. +0.98 m) used | all 112 (-1.38 .. +6.55 m) |
| window | 64 frames = 1.6 s | 30 s (three 10 s blocks) |

5 FPS is ~7x the Nyquist rate of breathing (0.20-0.33 Hz) and doubles the
integration per frame; it is not used for gesture. The other 14 preset settings
(TX power, masks, DC removal, ...) are identical. The two presets are
deliberately different: presence integrates 1400 iterations for breathing,
gestures need 40 FPS to follow a hand.

### 2.2 Range map (measured, not assumed)
`probe_range.py` reads the values the SDK itself reports:

$$R(b) = R_0 + b\,\Delta R,\qquad R_0 = -1.38\ \text{m},\quad \Delta R = 71.4\ \text{mm}$$

Bins 0-19 are negative range (behind the antenna, dominated by internal TX->RX
coupling); **bin 20 is the first physical bin** (+0.05 m). A code comment once
claimed 6.5 mm/bin -- wrong by 11x -- and everything is now derived from the
measured constants in `x7_physics.py` (`BIN_LENGTH_M`, `RANGE_OFFSET_M`,
`FIRST_PHYSICAL_BIN`).

Other constants: $f_c = 7.29$ GHz, $\lambda = 41.1$ mm, antenna baselines
$d_{TX} = d_{RX} = \lambda/2$.

**Each board is a one-axis array** (measured Sep 28). The TX and RX baselines
are parallel and equal, so the virtual elements TX1-RX0 and TX0-RX1 coincide
(phase difference 0.03 rad) and the 2x2 array is a 3-element line. Each board
measures angle along **one** axis -- the direction it is rotated to. There is
no separate vertical baseline (`x7_physics.py` still names `D_RX_V_M`
"vertical"; that is wrong). A board at roll $\alpha$ sees left-right motion in
proportion to $\cos\alpha$ and up-down motion in proportion to $\sin\alpha$, so
a gesture model is tied to the mounting it was recorded at. At the current
mounting the axis points mostly **up/down**: up/down swipes swing the angle by
2-5 rad, consistent in every session, while left/right is weak (Oct 5).
Left/right relies on weaker cues such as which board the hand passes first
(the boards are 15 cm apart), and on how the movement starts (Section 4.5).

### 2.3 Dual-sensor acquisition
* **Virtual MIMO frames:** each sensor's TX0/TX1 pulses are assembled into a
  $(2\,\text{TX} \times 2\,\text{RX} \times B)$ complex frame; the two sensors
  are paired by **SDK sequence number** (match within +-10). An earlier
  software timestamp gate compared integer ticks against 0.05 s and starved the
  queue; it was removed.
* **Recording tensor:** $\mathbf{W} \in \mathbb{C}^{T \times 2 \times 2 \times 2 \times B}$
  (frames, sensors, TX, RX, bins), saved as `.npy`.
* **Process isolation (`x7_acq_worker.py`):** the Novelda callback SDK holds the
  Python GIL while waiting for the next frame, which made inference in the same
  interpreter ~13,000x slower (1 ms -> 13 s, 0.01 cores busy). Acquisition
  therefore runs in its own process (`RadarProcess`, `kind="presence"` or
  `kind="gesture"`) and ships frames over a pipe; the gesture worker also
  reports the SDK's range map. The child watches its parent and exits if it dies, so a killed session
  cannot leave the radar locked.
* **The FT4222 USB bridge is exclusive-access.** A preflight in every radar
  script counts connected FT4222H interfaces (4 expected) and clears orphaned
  acquisition processes. The SDK's "check the DCNF0/1 jumpers" message almost
  always means an unplugged board or a stale process, not the jumpers.

---

## 3. Occupancy Detection -- Zone Model (`x7_presence_zone.py`)

### 3.1 Design
A fixed physics front end, **36 learned parameters**, and an asymmetric state
machine. numpy only -- no GPU, no neural network.

**Why so small:** v2 contains ~800 recordings but only ~16 independent
*sit-downs*. At $\lambda = 41$ mm a 1 cm shift of a torso rotates round-trip
phase by ~90 deg, so every sit-down has a unique, stable phase fingerprint. Every
higher-capacity model tried (count CNN, per-seat CNN) scored ~100% by
recognising the sit-down and chance on a new one. These features never see
absolute phase or absolute amplitude.

### 3.2 Pipeline
```
frames (5 Hz, 2 sensors x 4 virtual channels x 112 bins)
  -> 10 s blocks: quadratic detrend over time (MTI + drift removal), FFT
       breathing band 0.2-0.5 Hz, noise band 0.8-2.5 Hz
       reference floor: bins behind the antenna (no occupant possible)
  -> zones: front 0.35-1.05 m (bins 25-34, 80 channels)
            back  1.25-2.00 m (bins 37-47, 88 channels)   [shifted by calibration]
  -> per zone, over a 30 s window (3 blocks):
       eig    log(top whitened eigenvalue / Marchenko-Pastur edge)
       ncomp  log(1 + eigenvalues above the edge)
       snr    log(max breathing power / reference floor)
       br     log(max breathing power / own noise band)
       motion log(noise-band power / reference floor)
     + front/back coherence of the dominant components (leak rejection)   -> 11 features
  -> 3 logistic regressions: P(occupied), P(front), P(back)
  -> state machine (decision every 1 s)
```
The detrend matters: in a 10 s window the 0.2 Hz bin is two bins from DC, and
slow drift otherwise looks like breathing (an empty room showed 77 "sources"
before detrending, 1 after).

### 3.3 State machine and safety policy
* thresholds at **>= 99% out-of-fold recall**
* `OCCUPIED` is entered at once (probability over threshold, or per-frame motion)
  and **held 45 s** after the last evidence; `EMPTY` therefore appears ~60-75 s
  after everyone leaves (30 s window + hold)
* the hold is set by data: the longest run of missed windows in any sit-down was
  ~30 s (a lone back-seat occupant), so a shorter hold could flicker to `EMPTY`
* warm-up (first 30 s) shows `UNKNOWN`, or `OCCUPIED (motion)` if someone moves;
  never `EMPTY`
* **back row is shown as `Y` or `?`, never `-`**: at 99% recall it would flag
  ~54-67% of empty back rows, so only a confident-positive threshold (1% false
  alarm, 61% recall) is used, and it never drives `OCCUPIED`
* stalled frames -> `SENSOR FAULT`; health warnings -> `UNKNOWN` instead of `EMPTY`

### 3.4 Results

**Cross-day bridge test** -- trained on v2 (Sep 11-22, one occupancy level per
day), tested on v3 (Sep 25, every level on one day, same setup):

| model | occupied | front row | back row |
|---|---|---|---|
| **zone model** | AUC **1.000**, recall 100%, FA **0.0%** | AUC **1.000**, recall 97.9%, FA **0.0%** | AUC 0.954, FA 53.8% (indicator only) |
| count CNN (old approach), 5-class | 45.4% accuracy (balanced 55.3%) | -- | -- |
| count CNN, only bins > 3.55 m (control) | 31.5% | -- | -- |
| per-seat CNN | count 42.6% | seats 1/2: 81% / 81% | seats 3/4: 49% / 40% |

**Leave-one-day-out, final model (v2 + v3, 30 s window, 99% recall):**

| output | AUC | false alarm |
|---|---|---|
| occupied | 0.996 | 14.8% |
| front row | 0.998 | 5.5% |
| back row | 0.944 | 67.3% -> "Y or ?" at 1% FA, 61.1% recall |

**State machine on the held-out day** (replay, v2-only model, 30 s window):
missed-occupant **0.2%** of decisions; `EMPTY` after 39 s (10 s hold), 59 s
(30 s hold), 89 s (60 s hold). Compute: 0.01 ms per frame, 8 ms per decision
(desktop Python).

A no-seat control region (2.6-3.9 m) also separates occupied from empty. This is
physical -- occupants shadow the walls behind them and reflect into far bins,
reproduced on two separate empty days -- so for occupied/empty the evidence is
the transfer to a new day's empty trials, not the control.

### 3.5 Calibration, setup-change monitoring and self-test (`x7_presence_setup.py`)
Invariant by construction: sensor gain (+-6 dB), per-antenna phase, day-to-day
drift. **Not** invariant: where the rows sit relative to the fixed zones.
Simulated on the held-out day:

| change | occupied recall / FA | front recall / FA |
|---|---|---|
| none | 100% / 0% | 97.9% / 0% |
| radar 21 cm farther, old calibration | 93.3% / 0% | 79.2% / 0% |
| radar 21 cm farther, **recalibrated** | 100% / 0% | **97.9% / 0%** |
| radar 14 cm closer, old calibration | 100% / 0% | 100% / 11.5% |
| radar 14 cm closer, **recalibrated** | 100% / 0% | **100% / 0%** |

* **Calibration** finds each row's breathing peak (one person per seat) and
  shifts the trained zones by whole range bins; widths never change. Front
  offsets under 5 cm are treated as normal day-to-day variation; the back row
  follows the front unless they disagree by > 10 cm. It also sets the room's
  motion threshold and the receiver noise reference.
* **Health monitor (in `live`):** front-row breathing peak drifting > 10 cm over
  ~3 min of occupancy -> zones auto-align (rounded toward zero, max 30 cm) or
  `SETUP CHANGED, recalibrate`; receiver noise > 6 dB off calibration for 30 s
  -> `check the radar`; a calibration made for another model -> stale warning.
  Room-clutter fingerprints and wall-reflector lags were tested and could not
  tell a moved radar from an unchanged one; the front-row peak tracks a move to
  within a few cm.
* **`check`** (daily, ~4 min) scores one pass against the current calibration
  and recomputes it; it adopts a new calibration only if the setup moved and the
  new one passes a replay of the same recordings.
* Seats cannot be calibrated in pairs: with a front and a back occupant the
  back-row estimate locked onto front leakage (0.91 m vs 1.53 m); two people in
  one row read +2 to +21 cm off.

Current live calibration (2026-09-28): both rows **+29 cm** farther than in
training (front 0.99 m, back 1.79 m); self-test passed.

### 3.6 Limitations
* **No per-person count.** With sensors 15 cm apart, left/right separation is
  1.9-2.5 bins in the front row and 0.8-1.1 in the back; two people in one row
  merge into one return. One person produces 4-5 independent components in the
  breathing band, so source counting does not give a head count either.
* **Back row is weak:** ~11 dB below the front row (R^-4), and front-row
  breathing leaks into its bins. Sensors >= 50 cm apart would fix both but are
  not possible while the gesture mounting is shared.
* **Validated in one room** with four people. Other rooms, a real car cabin,
  more people, objects on seats and posture changes are untested; robustness
  can only be claimed for factors that were varied.
* v3 was recorded one scenario at a time, so it supports cross-day testing only
  (`check_confound.py`: `PARTIAL`). The next dataset should be one interleaved
  recorder run with re-seating before every trial.

---

## 4. 3D Gesture Recognition (v3, `x7_gesture_*.py`)

Rebuilt on Sep 29 after the old pipeline failed at the current mounting (the
deployed `m5_now` ensemble scored **7%** there, confidently wrong). Recording,
training and live all use one core module, so their trigger, windowing and
features cannot drift apart.

### 4.1 Classes
`SWIPE_LR`, `SWIPE_RL`, `SWIPE_UD` (up -> down), `SWIPE_DU` (down -> up),
`PUSH_IN`, and **`NO_GESTURE`**: non-gesture movements that trip the trigger
(reaching, scratching, adjusting in the seat). These teach the model to stay
silent. Vary them: one session's swipe-like non-gestures caused every
remaining false report (Sep 30).

### 4.2 Trigger and capture (`x7_gesture_core.Segmenter`)
* 40 FPS, **64-frame window** (1.6 s), peak-driven capture: the capture follows
  the running energy maximum and closes after 6 falling frames, 40 frames after
  the peak, or 110 frames; 20-frame cooldown.
* **Trigger energy:** moving energy in physical bins 20-33, divided by the
  noise floor behind the antenna (bins 0-15). The background adapts slowly
  (EMA, $\tau$ = 2 s) so a moving hand is not absorbed.
* **Trigger level:** 6x the idle level. The idle level comes from a 3 s
  warm-up (keep still), then is re-estimated every second from the 20th
  percentile of the last 30 s. It may only go **down**: a warm-up spoiled by
  movement recovers, and frequent gesturing cannot raise the trigger.
* While waiting, the recorder and live calibration show
  `energy N.Nx idle / trigger 6x`, or `radar sends NO FRAMES for N s`.
* **Recording uses the same automatic trigger as live** (default again since
  Oct 12). `--manual` arms each capture with **Enter** instead (added Oct 5);
  that captures only the deliberate stroke, which does **not** match what live
  detection captures, so it is not used for training the live model.
* **Direction check while recording:** each swipe is scored with the current
  model. If it looks like the opposite swipe (confidence >= 0.6), the recorder
  warns and `x` redoes it.

### 4.3 Features and model
* **Input per board** (2 boards x C x 14 bins x 64 frames), after removing the
  window mean:
  * `angle`: the board's four single-baseline pairs averaged coherently into one
    angle, as a unit phasor (Re, Im);
  * `amp`: MTI amplitude, z-scored within the window;
  * `doppler`: inter-frame phasor (optional, not used).
  * Default `angle+amp` (3 channels). Measured at the current mounting,
    `angle+amp` beat angle alone (53.9% vs 50.0%, with time reversal).
* **Model:** compact two-branch CNN, one branch per board (the boards are
  mounted differently), **113,606 parameters** (the old model had 11.6 M). A
  3-seed ensemble.
* **Training** (`x7_gesture_train.py`):
  * AdamW with a cosine schedule, class-weighted cross-entropy with label
    smoothing 0.05, 60 epochs;
  * augmentation: time shift, pace 0.8-1.25x, range shift ±1 bin (±7 cm),
    noise;
  * **time reversal:** a swipe played backwards is the opposite swipe (LR↔RL,
    UD↔DU), which doubles the data.
* **Evaluation:** leave-one-session-out only. The old random per-file split
  overstated accuracy by ~14 points.
* **Decision** (`GestureModel.decide`): a gesture is reported only if the
  prediction is not `NO_GESTURE`, its confidence clears the floor, and its
  **margin** (top minus second probability) clears the margin floor. The
  trainer picks both floors on the held-out predictions: the highest recall at
  `--target-precision`; if unreachable, the best precision with recall >= 15%.

### 4.4 Results (`models/gesture_v3.json`, trained 2026-09-30)
Data: `gesture_dataset_v3/`, 7 sessions (Sep 29: 4, Sep 30: 3), 378 captures,
all recorded with the automatic trigger.

| | value |
|---|---|
| held-out session accuracy | **82.3%** (per session 67.7-94.2%) |
| floors | confidence 0.65, margin 0.40 (`--target-precision 0.9`) |
| precision / gestures recognised | **90.2% / 76.6%** |
| natural left/right swipes, automatic trigger (`live_style_test/`, Oct 12) | **9/12** |
| cross-day | the Sep 29 model, frozen, scored 83.2% on Sep 30 |
| per gesture recognised | LR 83%, RL 72%, UD 72%, DU 71%, PUSH 85% |

What was measured to matter:

| lever | effect |
|---|---|
| more **sessions** | held-out 61.9 / 69.5 / 76.6 / 82.1% with 1 / 2 / 3 / 5 training sessions; not levelled off |
| more captures within a session | stops helping at ~8 per gesture |
| 5 models, averaging over time shifts, 120 epochs | within ±1 point: the model is not the limit |
| pretraining on the old corpus | hurt (41% -> 21%): other mountings contradict this one |
| session calibration (2 per class) | +13 points with 2 sessions; not re-measured with 7 (the base model already holds across days) |

**Seat moved** (simulated range shift of held-out captures):

| shift | 14 cm closer | 7 cm closer | none | 7 cm back | 14 cm back |
|---|---|---|---|---|---|
| gestures recognised | 69.2% | 73.6% | 76.6% | 74.9% | 67.6% |

A 5-10 cm move needs nothing; beyond that, record sessions at the new
positions.

### 4.5 Capture style decides what a swipe looks like (Oct 5-12)
* **Oct 5:** right→left was hard to detect live. Ten deliberate swipes recorded
  with the Enter trigger, hand still at both ends (`direction_test/`), were
  read as the opposite swipe 8/10 by `gesture_v3`. The September left/right
  labels were concluded to be reversed and relabelled. The corrected model
  (`_trial_corrected`, since deleted) scored 84.7% held out and 10/10 on `direction_test/`.
* **Oct 12:** live with the corrected model, left and right were swapped. Twelve
  natural swipes recorded with the automatic trigger (`live_style_test/`)
  settled it:

  | training labels | `live_style_test/` (live way) | `direction_test/` (Enter) |
  |---|---|---|
  | **September as recorded** | **10/12** | 1/10 |
  | September as recorded + Oct 5 sessions | 8/12 | 6/10 |
  | September relabelled + Oct 5 (`_trial_corrected`) | 0/12 | 10/10 |

* **Conclusion:** the September labels were right for natural swipes. A
  natural swipe captured the way live captures it (movement already under way,
  including the move into position) and an Enter-armed deliberate stroke have
  opposite signatures. The Oct 5 relabelling was undone
  (`models/_unswap_LRRL_20261012_log.json`). The Enter-style sessions G10051022
  and G10051151 were moved to `gesture_dataset_enter_style/`. Their labels are
  as performed; they are kept for a possible Enter-style live mode.
* **Rule:** record training data with the same trigger and the same natural
  movement as live use. Test a model on `live_style_test/`, not only on
  `direction_test/`.
* **Nov 19: "pause, then swipe" adopted.** Live left/right failed with
  `gesture_v3` although model and code were unchanged. The cue the model relies
  on -- which board the hand reaches first -- was reversed in 12 new natural
  swipes (`live_style_test_1119/`): left->right reached board 2 first in 77% of
  September swipes and 100% of Oct 12 ones, but in 0% of today's, the same as
  every Enter-armed swipe. The September captures held the *wind-up* (the hand
  travelling into position, the opposite way), so `gesture_v3` only works when
  the trigger catches a wind-up, which varies by person and day. The stroke itself
  is consistent, so live use now asks for a brief pause before the swipe.
  `gesture_dataset_v4_pause/` = September up/down/push/no-gesture + the Enter-style
  sessions + `direction_test/` + G11191210; September left/right are excluded.
  Held out, `gesture_v4_pause` confuses left and right 1/54 (`gesture_v3` on
  today's swipes: 4/12 wrong way), but left/right is often read as `PUSH_IN` and
  falls below the 0.80 floor: only 12 of its 33 swipes per direction come from
  the live trigger.

### 4.6 Limitations
* **Left vs right** is the weaker distinction at the current mounting, because
  each board's axis points mostly up/down (Section 2.2), and it depends on how
  the movement starts (Section 4.5). Up/down is strong.
* One room, one seat position, two days of training data. Hand distances 0.50-0.60 m are
  under-represented (50% recognised, measured on Sep 30).

### 4.7 Superseded gesture pipeline (`x7_2s_*.py`)
The old pipeline (10-channel input with a duplicate "vertical" AoA, 11.6 M
parameters, raw-energy trigger, one gesture per recording burst) and its
models `m5_now_seed{0,1,2}.pth` and `best_model_2s_5class.pth` are kept for
reference. They were trained at earlier mountings and do not work at the current
one. The Sep 10 advice to re-mount the boards "upright" was based on the
non-existent vertical baseline (see 2.2).

---

## 5. Mathematical Formulations & DSP Reference

### 5.1 Range mapping
$$R(b) = -1.38\ \text{m} + b \cdot 0.0714\ \text{m}$$

### 5.2 Static clutter removal
Gesture (window mean):
$$\mathbf{W}_{\text{MTI}}(t) = \mathbf{W}(t) - \frac{1}{T}\sum_{\tau}\mathbf{W}(\tau)$$
Occupancy (quadratic detrend per channel over each 10 s block):
$$\mathbf{x}_{\text{MTI}} = \big(\mathbf{I} - \mathbf{V}\mathbf{V}^{+}\big)\,\mathbf{x},\qquad \mathbf{V} = [\,\mathbf{1},\ \mathbf{t},\ \mathbf{t}^2\,]$$

### 5.3 Unit phasor and angle of arrival
$$\Phi(z_1, z_2) = \frac{z_1 z_2^*}{|z_1 z_2^*| + \varepsilon},\qquad \Delta\phi = \frac{2\pi d \sin\theta}{\lambda},\quad d = \lambda/2$$
The TX pair and the RX pair are parallel baselines of the same length, so
$\Phi(\text{TX}_0\text{RX}_0,\ \text{TX}_1\text{RX}_0)$ and
$\Phi(\text{TX}_0\text{RX}_0,\ \text{TX}_0\text{RX}_1)$ measure the **same**
angle (noise-cancelled correlation 0.91-0.98). v3 therefore averages the four
single-baseline pairs coherently:
$$z = v_{00}v_{10}^* + v_{01}v_{11}^* + v_{00}v_{01}^* + v_{10}v_{11}^*,\qquad \text{angle} = z/|z|$$

### 5.4 Doppler phasor and aliasing
$$\mathbf{D}(t) = \Phi\big(\mathbf{W}(t),\ \mathbf{W}(t-1)\big),\qquad v_{\max} = \frac{\lambda \cdot \text{FPS}}{4} = 0.41\ \text{m/s at 40 FPS}$$
Faster radial motion wraps; encoding (Re, Im) keeps the wrap continuous.

### 5.5 Path loss
Received power falls as $R^{-4}$: a 20 cm seat move at 0.6 m changes power by
~5 dB, and the back row (~1.5 m) is ~11 dB below the front row (~0.7 m). Any
feature reading absolute amplitude inherits this as nuisance; the occupancy
features use ratios and whitened statistics instead.

### 5.6 Breathing resolution
A window of length $T$ resolves $\Delta f = 1/T$: at 10 s the whole adult
breathing band (0.20-0.33 Hz) is ~1.3 bins, so people cannot be separated by
breathing rate; rates 0.05 Hz apart need $T \ge 20$ s.

### 5.7 Occupancy features
For zone snapshots $\mathbf{S} \in \mathbb{C}^{n \times p}$ (breathing-band FFT
bins x channels), each channel whitened by its own noise-band power,
$$\hat{\mathbf{C}} = \tfrac{1}{n}\,\mathbf{S}_w^{H}\mathbf{S}_w,\qquad \lambda_{\text{edge}} = \big(1 + \sqrt{p/n}\big)^2$$
Pure noise stays below the Marchenko-Pastur edge; `eig` = $\log(\lambda_1/\lambda_{\text{edge}})$,
`ncomp` = number of eigenvalues above it. Coherence between the zones' dominant
components, $|\langle \mathbf{a}, \mathbf{b} \rangle|^2 / (\|\mathbf{a}\|^2\|\mathbf{b}\|^2)$,
rejects front-row breathing that leaks into back-row bins.

### 5.8 Classifier and threshold
$$P(y=1 \mid \mathbf{f}) = \sigma\big(\mathbf{w}^\top \tfrac{\mathbf{f}-\boldsymbol{\mu}}{\boldsymbol{\sigma}} + b\big)$$
Class-balanced logistic regression (C = 0.5) per output; the threshold is the
1st percentile of out-of-fold positive scores (leave-one-day-out where possible),
i.e. 99% recall.

### 5.9 Gesture training
Class-weighted, label-smoothed cross-entropy with a cosine learning rate:
$$\mathcal{L} = -\sum_k \tilde{y}_k \log \hat{p}_k,\quad \tilde{y}_k = (1-\alpha)\,\mathbb{I}(y=k) + \alpha/K,\ \alpha = 0.05;\qquad \eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\big(1+\cos\tfrac{\pi t}{T_{\max}}\big)$$
Reporting rule, with sorted class probabilities $p_{(1)} \ge p_{(2)}$:
$$\text{report } \arg\max_k \hat p_k \iff \arg\max_k \hat p_k \ne \texttt{NO\_GESTURE},\ \ p_{(1)} \ge c,\ \ p_{(1)} - p_{(2)} \ge m$$

---

## 6. CLI Usage & Execution Reference

All commands run from the repository root. The radar host IDs used throughout
are `529` (sensor 1) and `545` (sensor 2). Seat numbering for presence:
**seats 1, 2 = front row, seats 3, 4 = back row.**

### Environment Setup
```powershell
.\.venv\Scripts\Activate.ps1
# Verify RTX 5090 CUDA 13.2 support
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```
If the NVIDIA driver is down, any PyTorch GPU call (and even a CPU `backward()`)
in `.venv` segfaults. Reboot; until then, CPU-only training works with the
system interpreter (same torch version, CPU build):
```powershell
C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe x7_2s_train_presence_seats.py --cpu
```
The presence zone model (`x7_presence_zone.py`, `x7_presence_setup.py`) is numpy
only and does not need the GPU at all.

### Hardware checks
* **What the radar reports (bin count, range offset, bin length):**
  ```powershell
  python probe_range.py
  ```
* **The FT4222 is exclusive-access.** Only one program may use the radar at a
  time -- never run live detection and the recorder together. Every radar
  script runs a preflight that (a) reports if fewer than 4 FT4222H USB
  interfaces are connected and (b) kills orphaned acquisition processes. The SDK
  error *"Could not find a FT4222 device configured in mode 0 ... DCNF0/1
  jumpers"* almost never means the jumpers: it means a board is unplugged or a
  stale process still holds the device (then replug the USB).
* **No frames although all 4 interfaces are present** (`SENSOR FAULT: no
  frames`), typically after a live run was force-killed: replug both USB
  cables. Stop live runs with Ctrl+C. If a stream fails, the run now stops
  with the SDK's error instead of waiting forever; `X7_WORKER_LOG=<file>`
  captures the radar worker's full output (Section 6.3).

---

### 1. Human Presence & Occupancy -- zone model (current pipeline)

Outputs **occupied / empty**, **front row** and **back row** (back row is shown
as `Y` or `?` only -- never `-` -- because with the sensors 15 cm apart it
cannot reliably say a back seat is empty). Validated leave-one-day-out and on a
held-out day; see `documentation/2026-09-25.md`.

#### Daily routine (~4 min, before every session)
```powershell
python x7_presence_setup.py check --host1 529 --host2 545
python x7_presence_zone.py live  --host1 529 --host2 545
```
`check` records empty 45 s, then one person each in seats 3, 4, 1, 2 (35 s),
scores it against the current calibration and computes a fresh one.
`>>> PASS -- go live` means the setup is unchanged. `>>> RECALIBRATED` means it
moved: the new calibration was replayed over the same recordings, passed, and
was adopted (the old one is kept as `models/presence_calib_prev.json`).
During `check`, `calibrate` and `selftest` the operator stays in **one place, at
least 2.5 m from the radar**.

#### Live detection
```powershell
python x7_presence_zone.py live --host1 529 --host2 545
python x7_presence_zone.py live --host1 529 --host2 545 --duration 120        # stop after 120 s
python x7_presence_zone.py live --host1 529 --host2 545 --exit-s 30           # shorter hold (riskier)
python x7_presence_zone.py live --host1 529 --host2 545 --no-auto-align       # report a moved setup, don't realign
python x7_presence_zone.py live --host1 529 --host2 545 --calib models/presence_calib.json
```
Status line: `OCCUPIED / EMPTY / UNKNOWN`, front `Y/-`, back `Y/?`,
probabilities, motion value vs threshold, and while occupied which evidence is
holding it (`held by ... clears in N s`). Warm-up is 30 s (someone arriving
shows `OCCUPIED (motion)` immediately); `EMPTY` is reported ~60-75 s after
everyone leaves (30 s window + 45 s hold -- a missed occupant is treated as worse
than a false alarm). Health warnings (setup moved, sensor signal changed, stale
calibration) turn `EMPTY` into `UNKNOWN`.

#### Calibration and self-test
```powershell
# calibrate only (~3 min): empty 40 s + one person per seat 30 s
python x7_presence_setup.py calibrate --host1 529 --host2 545

# quick self-test (~4 min): empty, seats 3, 4, 1, 2 -- no recalibration
python x7_presence_setup.py selftest --host1 529 --host2 545

# full self-test (~8.5 min): longer steps, all four seats, and occupied -> EMPTY clearing
python x7_presence_setup.py selftest --host1 529 --host2 545 --full --all-seats

# rerun only chosen steps (each failed report prints this line for you)
python x7_presence_setup.py selftest --host1 529 --host2 545 --steps 3 empty
```
Useful options: `--leave-s` (countdown to clear the seats, default 15),
`--empty-s`, `--seat-s`, `--calib`, `--report-dir` (default `selftest_reports/`).

**Offline** (runs the same procedures on recordings -- for testing the code, not
the setup; use a scratch `--calib` so the real calibration is not overwritten):
```powershell
python x7_presence_setup.py check     --from-files human_dataset_v3 --calib scratch_calib.json
python x7_presence_setup.py calibrate --from-files human_dataset_v3 --out scratch_calib.json
python x7_presence_setup.py selftest  --from-files human_dataset_v3 --calib scratch_calib.json
python test_presence_setup.py        # simulated radar moves / sensor fault: 10 checks
```

#### Training and evaluation
```powershell
# the command that produced the current models/presence_zone.json (36 parameters)
python x7_presence_zone.py train --data human_dataset_v2 human_dataset_v3 --window 3 --exit-s 45

# explore window lengths (in 10 s blocks) and skip the no-seat control
python x7_presence_zone.py train --data human_dataset_v2 human_dataset_v3 --windows 1 2 3 6 --no-control

# state-machine replay (latencies, missed-occupant rate) on recordings
python x7_presence_zone.py replay --model models/presence_zone_v2only.json --files "human_dataset_v3/*/*.npy" --runs 20 --run-len 18

# cross-day bridge test: train on v2, test on the v3 day (zone model, count CNN, seat CNN, controls)
python x7_presence_bridge_eval.py --zone-model models/presence_zone_v2only.json
```
After retraining, rerun `check` -- a calibration made with a different model is
flagged as stale.

#### Recording a presence dataset
```powershell
python x7_2s_record_presence.py --host1 529 --host2 545 --seats 4 --out human_dataset_v4 --trials 6 --block 1 --empty-trials 18
```
Answer `all` to the occupancy and arrangement prompts, and give a new subject
ID. Run the recorder **once** for the whole session (do not restart it per
scenario): it interleaves every scenario in reshuffled rounds, spreads the empty
trials through the session, and with `--block 1` prompts everyone to stand up and
re-seat before every trial -- each trial is then an independent sit-down.
Options: `--fps` (default 5), `--seconds` (default 10), `--countdown`,
`--countdown-empty` (default 15), `--preset`.

**Gate the dataset before training on it:**
```powershell
python check_confound.py --data human_dataset_v2 human_dataset_v3 human_dataset_v4
```
`NOT INTERPRETABLE` = occupancy is aliased with the recording session;
`PARTIAL` = only leave-one-day-out testing is valid; `usable` = classes are mixed
within sittings.

---

### 2. 3D Gesture Recognition (v3)

#### Live detection
```powershell
python x7_gesture_live.py --host1 529 --host2 545 --calibrate 0
python x7_gesture_live.py --host1 529 --host2 545 --calibrate 0 --log gesture_live_log   # save every capture
```
Uses `models/gesture_v4_pause.json` (check the `model ...` line in the header;
`--model models/gesture_v3.json` for the old natural-wind-up model).
See Section 4.5 for why the capture style matters. Keep still until "warming up"
finishes (3 s). For each gesture, **hold the hand still at the start for a
moment, then swipe** -- a swipe with a wind-up can be read as the opposite one.
The confidence and margin floors come from the model (0.80 / 0.00 for
`gesture_v4_pause`); override them with `--confidence` and `--margin`.
`--model PATH` selects another model. Output:
`>>> SWIPE_LR (97.3%) [4 ms]`, or `ignored: not a gesture / low confidence ...`.
Other options: `--trigger-x` (default 6), `--timeout` (s per calibration
prompt).

**Session calibration** is not needed on this setup (the base model holds
across days). `--calibrate N` asks for N captures per class, saves them to
`gesture_dataset_v4_pause/` as session `C…` (automatic trigger, as in live), and
fine-tunes the ensemble in a
separate process (~15 s; Ctrl+C skips). It raises the confidence floor to at
least 0.8, which cancels the margin floor's gain; `--no-save-calibration`
discards the captures.

#### Recording a session
```powershell
python x7_gesture_record.py --host1 529 --host2 545 --per-gesture 8 --no-gesture 20
```
After each GO, do the gesture **naturally, the way you do it in live use**.
The automatic trigger captures it exactly as live detection does.

* After each swipe the line `model sees ...` shows the current model's
  reading. `!! looks like the OPPOSITE swipe` means redo it with `x`.
  `--check-model PATH` chooses the model for this check; `--check-model ""`
  turns it off.
* `--manual` arms each capture with Enter (hand still at the start). It
  captures only the stroke and does not match live detection, so it is for
  experiments only.
* `--gestures SWIPE_LR SWIPE_RL` records only those gestures, e.g. a live-style
  test that stays out of the dataset:
  `--gestures SWIPE_LR SWIPE_RL --per-gesture 6 --no-gesture 0 --block 1 --out live_style_test`.
* Gestures come in shuffled blocks of 2 (`--block`), never the same label
  twice in a row. The session ID defaults to `G` + month/day/hour/minute.
* Press `x` at the next prompt to discard the last capture and redo it;
  Ctrl+C stops and keeps everything saved.
* A gesture that does not trigger within `--timeout` s (default 8) is asked
  again, up to 3 times.
* No calibration is needed before recording, only the warm-up.
* Record each session as a new sitting. Vary the seat position (±10 cm)
  between sessions, and the NO_GESTURE movements within one.

#### Training
```powershell
python x7_gesture_train.py --target-precision 0.9 --out models/_trial_new.json   # trial (~12 min)
python x7_gesture_train.py --target-precision 0.9                                # overwrites gesture_v4_pause
```
* Trains on every session in `gesture_dataset_v4_pause/`, prints the held-out
  accuracy per session, the confusion matrix and a floor table, then trains
  the final 3-model ensemble on all data.
* It **overwrites** `models/gesture_v4_pause.*`; use `--out` for a trial model.
* Options: `--channels {angle,angle+amp,angle+amp+doppler}`, `--no-reverse`,
  `--epochs` (60), `--seeds` (3), `--data`, `--pretrain-corpus DIR`
  (measured to hurt).
* The default `--target-precision` is still 0.95, which picks a floor too
  strict for live use (17-44% recall). Pass 0.9.
* Before promoting a trial model, score it on a new "pause, then swipe" session
  recorded with `--gestures SWIPE_LR SWIPE_RL --out <test dir>` and kept out of
  training (`live_style_test/` is the old wind-up style).
* To promote a trial: back up `models/gesture_v4_pause*`, then retrain without `--out`
  (or copy the trial files over the `gesture_v4_pause` names and edit the `weights`
  list inside the json to match).

#### Legacy (`x7_2s_*`, earlier mountings -- reference only)
```powershell
python x7_2s_live.py --host1 529 --host2 545                     # m5_now ensemble
python x7_2s_debug_live.py --host1 529 --host2 545 --label SWIPE_LR --session s3
python x7_2s_train.py --classes 5 --checkpoint best_model_2s_5class_new.pth --epochs 80
python x7_2s_eval.py --model models\best_model_2s_5class.pth --mode split
```

---

### 3. GUI, combined presence + gesture, and the packaged app (`X7App.exe`)

#### The window (`x7_gesture_gui.py`)
A Tk window that animates the recognised gesture with its full name
(`SWIPE_LR` -> "Left to Right"), the confidence and the last five detections.
During calibration it loops the gesture to perform next. In combined mode it
adds a cabin panel (status, front / back row) and a bar showing which radar
preset is running. It runs as a **separate process** fed JSON lines on stdin:
the radar SDK holds the GIL while waiting for frames, so an in-process
animation would stutter. Closing the window never stops recognition.
```powershell
python x7_gesture_live.py --host1 529 --host2 545 --gui       # gesture only, with window
python x7_gesture_gui.py --demo                               # window only, fake events, no radar
```

#### Combined presence + gesture (`x7_live_combined.py`)
Presence and gesture need incompatible presets (presence: 3000 iterations per
frame at 5 FPS to see breathing; gesture: 6 at 40 FPS), so the radar
alternates: presence until `OCCUPIED`, then gestures until 60 s without
movement (`--recheck-idle`), then a presence recheck.
```powershell
python x7_live_combined.py --host1 529 --host2 545            # with window (--no-gui for console only)
python x7_live_combined.py --simulate --speed 4               # replays human_dataset_v3 + gesture_dataset_v4_pause, no radar
```

#### Packaged app (`X7App.exe`)
The live tools as one Windows program that runs on a PC without Python and
without this repository (Nov 27, `documentation/2026-11-27.md`). Built with
PyInstaller as a **folder** (`dist\X7App\`, ~550 MB, of which torch ~360 MB),
not a single file: a one-file exe would unpack torch to a temp folder on every
launch.
```powershell
.\dist\X7App\X7App.exe                                        # menu (what a double-click shows)
.\dist\X7App\X7App.exe selfcheck                              # run first on a new PC
.\dist\X7App\X7App.exe combined --host1 529 --host2 545       # = x7_live_combined.py
.\dist\X7App\X7App.exe gesture  --host1 529 --host2 545       # = x7_gesture_live.py --gui
.\dist\X7App\X7App.exe presence-setup calibrate --host1 529 --host2 545   # = x7_presence_setup.py
.\dist\X7App\X7App.exe demo                                   # = x7_live_combined.py --simulate
.\dist\X7App\X7App.exe gui-demo                               # = x7_gesture_gui.py --demo
```
Every option of the underlying script is passed through (`<mode> --help`).
The same launcher runs from source: `python x7_app.py <mode> ...`.

* **Folder layout.** `X7App.exe` and `_internal\` (Python, CPU torch, the
  radar SDK in `sdk\`, the radar wrapper + flows + presets in `radar\`, the
  BA22 firmware in `firmware\`). Next to the exe, so they can be swapped and
  written: `models\` (deployed gesture and presence models,
  `presence_calib.json`), `gesture_dataset_v4_pause\` (mixed into gesture
  calibration; new calibration captures are saved there) and
  `human_dataset_v3\` (demo). The app works from that folder whatever the
  current directory, so keep it somewhere writable (not `Program Files`).
* **`selfcheck`** checks without streaming: SDK loads from the bundle, every
  Python node the flow files name imports, presets and firmware found, models
  load, worker processes start, folder writable, 4 FT4222H interfaces present.
* **Stopping.** Use **Ctrl+C**. Force-killing a live run (Task Manager,
  `Stop-Process -Force`) can leave the FT4222 stuck: the boards still
  enumerate but send no frames, from the app or from source, until both USB
  cables are replugged.
* **Worker log.** The radar worker's output (including the SDK's own
  messages) is invisible when the console output is redirected. Set
  `X7_WORKER_LOG` to capture it, for the app and for the Python scripts:
  ```powershell
  $env:X7_WORKER_LOG = "$env:TEMP\x7_worker.log"
  ```

#### Rebuilding the app
```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```
* Uses its own venv, `.venv-build`: Python **3.10.22** (standalone, from
  `uv python install 3.10`), **CPU-only** torch 2.13, numpy, psutil,
  PyInstaller 6.22 (`requirements-build.txt`). The CUDA torch in `.venv` would
  add ~2.5 GB for three 470 KB models.
* Python must be 3.10 (the SDK ships `PySignalFlow` for cp310 only) but **not
  3.10.0**, the system Python here: its `dis` module crashes PyInstaller
  (`IndexError: tuple index out of range` in `_get_const_info`).
* What goes into the bundle is in `x7_app.spec`; the build script then copies
  the six deployed model files and the two datasets next to the exe. Change
  the model list in `build_exe.ps1` when a new model is deployed.
* The build clears `PYTHONPATH`. This PC sets it to the Novelda `Utils`
  folder, and the SDK `bin` folder is on `PATH`; the app must not depend on
  either. `x7_paths.py` now locates the SDK, flows, presets and firmware for
  source and packaged runs alike (the recorders used hard-coded
  `C:\RR_02\...` paths).

---

### 4. Superseded presence tools (reference only)

The count-CNN presence pipeline below was found to learn the **recording
session, not the occupants**: every occupancy level had been recorded on its own
day, and the same architecture scored 99.7% on v2 but **45%** on a new day
(see `documentation/2026-09-21.md` and `2026-09-25.md`). The datasets its
defaults point to (`human_dataset_2s_*`) no longer exist. Kept for reference;
do not use for results.
```powershell
python x7_2s_live_presence.py --host1 529 --host2 545                        # count CNN, live
python x7_2s_train_presence.py --data <dir> --val_data <dir> --epochs 40      # count CNN training
python x7_2s_eval_presence.py --data <dir> --model models/best_model_2s_presence.pth
python x7_2s_synthesize_data.py --src <dir> --target_per_class 600            # synthetic augmentation
python x7_2s_train_presence_seats.py --data human_dataset_v2                  # per-seat CNN (chance across days)
```

---

## 7. Repository Map

**Shared**
| file | purpose |
|---|---|
| `x7_physics.py` | single source of truth: measured range map, wavelength, baselines, phasor/MTI helpers |
| `probe_range.py` | prints the bin count, range offset and bin length the radar reports |
| `x7_acq_worker.py` | radar in its own process; orphan watchdog; USB/stale-process preflight |

**Occupancy (current)**
| file | purpose |
|---|---|
| `x7_presence_zone.py` | zone model: features, training (`train`), `replay`, `live`, health monitor |
| `x7_presence_setup.py` | `check` (daily), `calibrate`, `selftest` -- live or `--from-files` |
| `x7_2s_record_presence.py` | interleaved, re-seating dataset recorder |
| `check_confound.py` | dataset gate: are occupancy levels aliased with sessions? |
| `x7_presence_bridge_eval.py` | train on one dataset, test on a different day, all models + controls |
| `test_presence_setup.py` | offline checks for calibration and the setup-change monitor |
| `models/presence_zone.json` | deployed zone model (v2 + v3, 30 s window, 45 s hold) |
| `models/presence_zone_v2only.json` | v2-only fit used for the cross-day bridge test |
| `models/presence_calib.json` | current live calibration (`_prev.json` = previous) |
| `human_dataset_v2/`, `human_dataset_v3/` | recordings (v2: one level per day, 799; v3: all levels on Sep 25, 108) |
| `selftest_reports/` | JSON reports from `check` and `selftest` |

**Gesture (current, v3)**
| file | purpose |
|---|---|
| `x7_gesture_core.py` | constants, trigger (`Segmenter`), Enter wait (`wait_for_enter`), features, model, `GestureModel` (decision rule) |
| `x7_gesture_record.py` | shuffled session recorder: automatic trigger (as live; `--manual` = Enter), direction check, `--gestures` |
| `x7_gesture_train.py` | leave-one-session-out training, floor selection, ensemble |
| `x7_gesture_live.py` | live recognition, optional session calibration |
| `models/gesture_v4_pause.json` + `gesture_v4_pause_seed{0,1,2}.pt` | **deployed model** (Nov 19, 11 sessions, "pause, then swipe", floors 0.80 / 0.00) |
| `models/gesture_v3.json` + `gesture_v3_seed{0,1,2}.pt` | previous model (Sep 30, natural wind-up style, floors 0.65 / 0.40) |
| `models/*_heldout.npz` | held-out predictions from each training run |
| `models/_unswap_LRRL_20261012_log.json` | rename log of the Oct 12 undo of the Oct 5 relabelling |
| `gesture_dataset_v4_pause/` | **current training set** ("pause, then swipe"; see Section 4.5 for what it contains); recorder default |
| `gesture_dataset_v3/` | September recordings (automatic trigger, natural wind-up), `LABEL_SESSION_ts_idx.npy`, 64 x 2 x 2 x 2 x 34 complex; 7 sessions |
| `gesture_dataset_enter_style/` | the two Oct 5 sessions (G10051022, G10051151), Enter-style captures; not trained on |
| `live_style_test/` | 12 natural LR/RL swipes, automatic trigger (Oct 12); wind-up style |
| `live_style_test_1119/` | 12 LR/RL swipes, automatic trigger (Nov 19), stroke signature; also in `gesture_dataset_v4_pause/` |
| `direction_test/` | 10 deliberate Enter-armed LR/RL swipes (Oct 5); test set for the Enter style |
| `gesture_live_rl/`, `gesture_live_corrected/` | logged live captures from the Oct 5 tests |
| `x7_2s_record.py` | still provides the gesture radar class used by `x7_acq_worker` |

**GUI, combined mode and the packaged app**
| file | purpose |
|---|---|
| `x7_gesture_gui.py` | the window (separate process, JSON lines on stdin); `--demo` |
| `x7_live_combined.py` | presence + gesture on the one dual-X7 board, alternating presets; `--simulate` |
| `x7_app.py` | launcher and menu; the entry point of `X7App.exe` |
| `x7_paths.py` | where the SDK, radar flows/presets and firmware are (source or bundle); replaces hard-coded `C:\RR_02\...` paths |
| `x7_selfcheck.py` | `selfcheck` mode: is this PC set up (SDK, flow nodes, models, workers, USB) |
| `x7_app.spec`, `build_exe.ps1`, `requirements-build.txt` | PyInstaller build of `dist\X7App\` (see Section 6.3) |
| `X7App_README.txt` | user notes shipped as `README.txt` next to the exe |

**Gesture (superseded):** `x7_2s_live.py`, `x7_2s_debug_live.py`, `x7_2s_train.py`,
`x7_2s_eval.py`, `models/m5_now_seed{0,1,2}.pth`, `models/best_model_2s_5class.pth`,
`debug_captures/` (315 windows from earlier mountings).

**Superseded (reference only):** `x7_2s_live_presence.py`, `x7_2s_train_presence.py`,
`x7_2s_eval_presence.py`, `x7_2s_synthesize_data.py`, `x7_2s_train_presence_seats.py`,
`models/presence_seats.pth`. `architecture.mmd` still describes the earlier design
(8 gesture channels, 50 ms timestamp gate, count-CNN presence) and has not been
updated.

---

## 8. Engineering Log

Dated entries in [`documentation/`](./documentation/), each with objective,
measurements, errors found (including our own), current state and lessons:

| date | topic |
|---|---|
| 2026-09-04 | full audit; augmentation noise, evaluation on training data, 339-bin padding |
| 2026-09-08 | live gesture debugging; PINCH removed; class-name inversion; 5-class ensemble |
| 2026-09-10 | range constant wrong by 11x; boards rolled 45 deg; re-mount; recorder rewrite |
| 2026-09-11 | presence audit; stale constants; the "3 people max" question |
| 2026-09-15 | first presence recordings; car seating; AoA channels (and the recording plan that caused the confound) |
| 2026-09-16 | 2-person recordings; later shown to be one block per sit-down |
| 2026-09-21 | the 99.7% that meant nothing; interleaved recorder; GIL fix; FTDI lock |
| 2026-09-22 | instrument stability; seat and breathing physics; scope: room PoC |
| 2026-09-24 | sit-down memorisation; energy and breathing tests; zone model built |
| 2026-09-25 | bridging session; cross-day validation; calibration, monitor and self-test |
| 2026-09-28 | first live calibration; faster `check` routine; README rewrite; gesture review: each board is a one-axis array |
| 2026-09-29 | gesture v3 pipeline (record/train/live); 4 sessions, 41% → 77% held-out; threshold, fine-tuning and trigger fixes |
| 2026-09-30 | gesture: holds across days (83% frozen model); 7 sessions, margin floor, 90% precision / 77% recall; seat-shift and SWIPE_RL analysis |
| 2026-10-05 | September left/right labels judged reversed and relabelled (undone Oct 12, see Section 4.5); manual Enter trigger; direction test |
| 2026-10-12 | Oct 5 relabelling found wrong and undone (live-style test 9/12 vs 0/12); recorder back to automatic trigger; presence recalibrated (+3 bins), empty step fails |
| 2026-11-19 | gesture left/right failing live: the captured board order had reversed (wind-up vs stroke); switched to "pause, then swipe", `gesture_v4_pause` deployed |
| 2026-11-27 | packaged app `X7App.exe` (PyInstaller); path, process and SDK-node fixes; radar stream failures no longer hang; worker log |

**Lessons that shaped the current design**
* Run the control before reporting the number: a classifier on empty range
  bins scoring 99.7% was the real result.
* Count independent samples, not files -- 799 recordings were ~16 sit-downs.
* Put invariance in the measurement (ratios, whitening, calibration), not in a
  model trained on too little variation.
* Verify the antenna model before interpreting angles: the "vertical"
  baseline never existed, and a one-line test (do the cross elements move?)
  showed it.
* For gestures, more sessions beat more repetitions, and model changes were
  within noise: vary what varies in real use (sitting, seat position,
  non-gestures).
* Test against ground truth that matches real use: ten deliberate,
  Enter-armed swipes were taken as ground truth and led to relabelling data
  that was right for natural swipes. Twelve natural swipes captured the live
  way showed the error (Oct 12).
* Encode the safety asymmetry in the state machine: recall targets, holds,
  `UNKNOWN` and `SENSOR FAULT` are what make "a missed occupant is worse" true.
