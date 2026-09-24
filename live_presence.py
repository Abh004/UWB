"""
DEPRECATED -- superseded by x7_2s_live_presence.py
==================================================
This is the original Keras/TensorFlow presence demo. It is retained for
reference only and is NOT part of the active pipeline:

  * It loads `presence_2d_cnn.h5`, which is not present in this repository.
  * It requires TensorFlow, which is no longer a project dependency
    (see requirements.txt -- the active stack is PyTorch).
  * It reads a single sensor and predates the dual-sensor MIMO format.

Use `x7_2s_live_presence.py` instead.
"""

import os
import sys
import argparse
import json
from collections import deque, Counter
import numpy as np

# Suppress TensorFlow logging to keep terminal output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

# Ensure X7 SDK Bin is on the system path
_X7_SDK_BIN = r"C:\RR_02\X7_SDK_0.6_x64-win\bin"
if _X7_SDK_BIN not in sys.path and os.path.isdir(_X7_SDK_BIN):
    sys.path.insert(0, _X7_SDK_BIN)

# Ensure the callback framework directory is on the path
_CALLBACK_DIR = r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect\X7RadarDirectCallback"
if _CALLBACK_DIR not in sys.path and os.path.isdir(_CALLBACK_DIR):
    sys.path.insert(0, _CALLBACK_DIR)

from radar_direct_callback import RadarDirectCallback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_SIZE      = 50   # Frames in the sliding window (must match training)
PRED_INTERVAL = 1        # Predict every 1 frame for maximum responsiveness
CONF_THRESHOLD = 0.6     # Reject predictions below 60% confidence
VOTING_WINDOW = 3        # Require only 3 consecutive/majority votes to stabilizee history length

STATE_CHANGE_COUNT = 3   # Consecutive agreeing votes required to flip state


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_buffer_keras(complex_buffer):
    """
    Extract 8 advanced physical channels from a window of complex radar data.
    Input: (Frames, TX, RX, Bins) -> expected (50, 2, 2, 339)
    Output: (Frames, Bins, 8) -> expected (50, 339, 8)
    """
    w = complex_buffer
    
    # 1. MTI Filter
    w_mti = w - np.mean(w, axis=0, keepdims=True)
    
    tx0_rx0 = w_mti[:, 0, 0, :]
    tx1_rx0 = w_mti[:, 1, 0, :]
    tx0_rx1 = w_mti[:, 0, 1, :]
    
    # Helper to calculate Unit Phasors
    def to_phasor(z1, z2):
        interf = z1 * np.conj(z2)
        return interf / (np.abs(interf) + 1e-9)
        
    # 2. Spatial Unit Phasors
    H_phasor = to_phasor(tx0_rx0, tx1_rx0)
    V_phasor = to_phasor(tx0_rx0, tx0_rx1)
    
    # 3. Temporal Unit Phasor (Doppler)
    D_phasor = np.zeros_like(tx0_rx0, dtype=np.complex64)
    D_phasor[1:] = to_phasor(tx0_rx0[1:], tx0_rx0[:-1])
    
    # 4. Amplitude Features
    amp = np.mean(np.abs(w_mti), axis=(1, 2))
    amp_norm = (amp - np.mean(amp)) / (np.std(amp) + 0.5)
    
    amp_H_diff = np.abs(tx0_rx0) - np.abs(tx1_rx0)
    amp_H_norm = amp_H_diff / (np.abs(tx0_rx0) + np.abs(tx1_rx0) + 0.5)
    
    # Stack into 8 bounded, physically explicit channels
    stacked = np.stack([
        amp_norm,
        np.real(H_phasor), np.imag(H_phasor),
        np.real(V_phasor), np.imag(V_phasor),
        np.real(D_phasor), np.imag(D_phasor),
        amp_H_norm
    ], axis=-1)
    
    return stacked.astype(np.float32)

# ---------------------------------------------------------------------------
# Live people counter
# ---------------------------------------------------------------------------

class RadarLiveCounter:
    def __init__(self, model_path="presence_2d_cnn.h5", window_size=50):
        self.window_size = window_size
        print(f"[*] Loading Keras model from {model_path}...")
        self.model = keras.models.load_model(model_path)

        # Model shape sanity check
        print("[*] Model input shape:", self.model.input_shape)
        expected = (None, WINDOW_SIZE, 339, 8)
        if self.model.input_shape != expected:
            print(
                f"[!] WARNING: Expected {expected}, "
                f"got {self.model.input_shape}"
            )

        # Sliding window buffer (now stores complex frames)
        self.buffer = deque(maxlen=self.window_size)
        self.frame_count = 0
        
        # State machine for robust output (majority voting)
        self.prediction_history = deque(maxlen=VOTING_WINDOW)
        self.current_state = 0 # 0=Empty, 1=Presence

        # Occupancy state machine
        self.pending_state = None
        self.pending_count = 0

        # Frame synchronization
        self.last_tx_data  = [False, False]
        self.virtual_frame = None

    # ------------------------------------------------------------------
    # Radar callback — called by RadarDirectCallback for every TX frame
    # ------------------------------------------------------------------

    def radar_callback(
        self,
        trx_vec,
        radar_data,
        sequence_num,
        timestamp,
        range_offset,
        bin_length,
    ):
        tx_idx = int(trx_vec[1])  # TX channel index: 0 or 1

        # Initialise virtual frame shape on first callback
        if self.virtual_frame is None:
            num_rx, _, num_bins = radar_data.shape
            # Ensure bins is 339 by padding/truncating if necessary
            self.virtual_frame = np.zeros(
                (2, num_rx, 339), dtype=np.complex64
            )

        # BBIQ → complex
        complex_data = (
            radar_data[:, 0, :] + 1j * radar_data[:, 1, :]
        ).astype(np.complex64)
        
        # Pad or truncate to 339
        if complex_data.shape[1] < 339:
            complex_data = np.pad(complex_data, ((0, 0), (0, 339 - complex_data.shape[1])))
        elif complex_data.shape[1] > 339:
            complex_data = complex_data[:, :339]

        self.virtual_frame[tx_idx, :, :] = complex_data
        self.last_tx_data[tx_idx]        = True

        # Only proceed once both TX channels are populated
        if all(self.last_tx_data):
            
            # Append the completed 2x2 MIMO frame to the buffer
            self.buffer.append(self.virtual_frame.copy())
            self.frame_count += 1

            # Reset TX sync flags for the next pair
            self.last_tx_data = [False, False]

            if self.frame_count % PRED_INTERVAL == 0:
                self.run_inference()

        return True

    # ------------------------------------------------------------------
    # Inference + stabilization + state machine
    # ------------------------------------------------------------------

    def run_inference(self):
        N = len(self.buffer)

        # Wait until the buffer is full to avoid using padded/cold data
        if N < WINDOW_SIZE:
            sys.stdout.write(f"\rCollecting frames... {N}/{WINDOW_SIZE}   ")
            sys.stdout.flush()
            return

        # window is (50, 2, 2, 339) complex64
        window = np.array(self.buffer, dtype=np.complex64)

        # Extract advanced radar features using the same function from training
        processed_window = preprocess_buffer_keras(window) # (50, 339, 8)

        # Motion metric (diagnostic): Sum of absolute MTI amplitude across the window
        mti_energy = np.max(np.sum(np.abs(window - np.mean(window, axis=0, keepdims=True)), axis=(1, 2, 3)))

        # The model now outputs a single sigmoid probability
        pred_prob = self.model.predict(np.expand_dims(processed_window, axis=0), verbose=0)[0][0]
        
        # Binary threshold at 0.5
        pred_class = 1 if pred_prob > 0.5 else 0
        confidence = pred_prob if pred_class == 1 else (1.0 - pred_prob)
        
        # Update voting history
        self.prediction_history.append(pred_class)
        
        # Determine stable state
        if len(self.prediction_history) == VOTING_WINDOW:
            counts = Counter(self.prediction_history)
            most_common = counts.most_common(1)[0][0]
            # Only change state if confident (e.g. majority > 50%)
            # Actually, most_common is already the majority
            self.current_state = most_common

        labels = ["Empty", "Presence"]
        
        sys.stdout.write(
            f"\r[FRAME {self.frame_count:5d}] "
            f"State: {labels[self.current_state]:<10} | "
            f"Conf: {confidence * 100:5.1f}% | "
            f"Energy: {mti_energy:6.1f}   "
        )
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live Radar Inference People Counter"
    )
    parser.add_argument(
        "preset_json",
        nargs="?",
        default="human_preset.json",
        help="Path to the setup preset JSON file",
    )
    parser.add_argument(
        "--model", type=str, default="presence_2d_cnn.h5", help="Path to Keras H5 model."
    )
    parser.add_argument(
        "--playback_file",
        type=str,
        help="Optional playback file for offline validation",
    )
    args = parser.parse_args()

    # Locate model file
    model_path = args.model
    if not os.path.exists(model_path):
        fallback_model = r"C:\RR_02\counting_2d_cnn.h5"
        if os.path.exists(fallback_model):
            model_path = fallback_model
        else:
            print(f"Error: Model file '{args.model}' not found.")
            sys.exit(1)

    # Locate preset file
    preset_path = args.preset_json
    if not os.path.exists(preset_path):
        fallback_preset = r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect\X7RadarDirectCallback\Presets\human_preset.json"
        if os.path.exists(fallback_preset):
            preset_path = fallback_preset
        else:
            print(f"Error: Preset file '{args.preset_json}' not found.")
            sys.exit(1)

    setup_json = preset_path

    # Playback override (offline validation)
    if args.playback_file:
        print(f"[*] Playback override: using '{args.playback_file}'")
        with open(preset_path, "r") as f:
            preset_data = json.load(f)

        preset_data["IsLive"]       = False
        preset_data["PlaybackFile"] = str(os.path.abspath(args.playback_file))
        preset_data["DoRecording"]  = False

        temp_preset_path = "temp_live_playback_preset.json"
        with open(temp_preset_path, "w") as f:
            json.dump(preset_data, f, indent=4)

        setup_json = temp_preset_path
        print(f"[*] Temporary preset saved to: {temp_preset_path}")

    # Startup banner
    print()
    print("=" * 38)
    print(" LIVE HUMAN OCCUPANCY DETECTION (2D CNN)")
    print("=" * 38)
    print(f"Window Size      : {WINDOW_SIZE} frames")
    print(f"Prediction Rate  : every {PRED_INTERVAL} frames")
    print(f"Confidence Gate  : {CONF_THRESHOLD}")
    print(f"Voting Window    : {VOTING_WINDOW} frames")
    print(f"State Change Req : {STATE_CHANGE_COUNT} votes")
    print("=" * 38)
    print()

    counter = RadarLiveCounter(model_path)

    print(f"[*] Starting inference with preset: {setup_json}")
    try:
        RadarDirectCallback().run_with_callback_preset(
            counter.radar_callback,
            setup_json,
        )
    except KeyboardInterrupt:
        print("\n[*] Exiting clean.")
    finally:
        if args.playback_file and os.path.exists("temp_live_playback_preset.json"):
            try:
                os.remove("temp_live_playback_preset.json")
            except Exception:
                pass


if __name__ == "__main__":
    main()