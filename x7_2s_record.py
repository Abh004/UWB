import os
os.add_dll_directory(r"C:\RR_02\bin") 

import sys
import json
import tempfile
import argparse
import time
import threading
import queue
from pathlib import Path
import numpy as np

_DEMO_DIR = r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect\X7RadarDirectCallback"
_DEMO_PARENT = r"C:\RR_02\novelda-uwb-demos\Demos"
for path in [_DEMO_PARENT, _DEMO_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from radar_direct_callback import RadarDirectCallback

# ── CONFIG ──────────────────────────────────────────────────────────────
RANGE_BINS = 34    
RADAR_FPS  = 40   
SYNC_TOLERANCE = 0.05  # seconds

GESTURE_CLASSES = {
    "1":  ("G01_SWIPE_LR", "Left -> Right swipe"),
    "2":  ("G02_SWIPE_RL", "Right -> Left swipe"),
    "3":  ("G03_SWIPE_UD", "Up -> Down swipe"),
    "4":  ("G04_SWIPE_DU", "Down -> Up swipe"),
    "5":  ("G05_PUSH_IN",  "Push inward"),
    "6":  ("G06_PINCH",    "Pinch to trigger"),
}

# ── RADAR SOURCE ────────────────────────────────────────────────────────
class X7DualRadar:
    def __init__(self, host1, host2, preset_json_path=None):
        self.frame_queue = queue.Queue(maxsize=400)
        self.range_bins = RANGE_BINS
        if preset_json_path is None:
            preset_json_path = r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect\X7RadarDirectCallback\Presets\default_preset.json"
        
        self.host1 = host1
        self.host2 = host2
        
        # Virtual frames for assembling TX0 and TX1 (TX, RX, BINS)
        self.virtual_frame1 = np.zeros((2, 2, self.range_bins), dtype=np.complex64)
        self.last_tx_data1 = [False, False]
        self.virtual_frame2 = np.zeros((2, 2, self.range_bins), dtype=np.complex64)
        self.last_tx_data2 = [False, False]
        
        # Buffer for software synchronization
        self.buffer_lock = threading.Lock()
        self.buffer1 = {} # seq_num -> frame
        self.buffer2 = {} # seq_num -> frame
        
        self.runner1 = RadarDirectCallback()
        self.runner2 = RadarDirectCallback()
        
        # Create custom presets with Host IDs
        self.preset1_path = self._create_port_preset(preset_json_path, host1)
        self.preset2_path = self._create_port_preset(preset_json_path, host2)
        
        self._thread1 = threading.Thread(target=self.runner1.run_with_callback_preset, 
                                        args=(self._frame_callback1, self.preset1_path), daemon=True)
        self._thread2 = threading.Thread(target=self.runner2.run_with_callback_preset, 
                                        args=(self._frame_callback2, self.preset2_path), daemon=True)
        
        self._thread1.start()
        print(f"[INFO] Sensor 1 (Host {host1}) stream started.")
        self._thread2.start()
        print(f"[INFO] Sensor 2 (Host {host2}) stream started.")

    def _create_port_preset(self, base_preset, host_id):
        with open(base_preset, 'r') as f:
            preset = json.load(f)
        
        if "ConnectionParameters" not in preset:
            preset["ConnectionParameters"] = {}
        preset["ConnectionParameters"]["HostInterfaceId"] = str(host_id)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(preset, temp_file)
        temp_file.close()
        return temp_file.name

    def _process_assembled_frame(self, frame, seq_num, sensor_idx):
        with self.buffer_lock:
            if sensor_idx == 1:
                self.buffer1[seq_num] = frame
            else:
                self.buffer2[seq_num] = frame
                
            # Try to synchronize
            matched_seq = None
            search_window = 10
            for offset in range(-search_window, search_window + 1):
                target_seq = seq_num + offset
                if target_seq in self.buffer1 and target_seq in self.buffer2:
                    matched_seq = target_seq
                    break
                    
            if matched_seq is not None:
                # Stack to shape (2 Sensors, 2 TX, 2 RX, Bins)
                combined_frame = np.stack([self.buffer1[matched_seq], self.buffer2[matched_seq]], axis=0)
                # Clean up older frames
                keys1 = list(self.buffer1.keys())
                keys2 = list(self.buffer2.keys())
                for k in keys1:
                    if k <= matched_seq: del self.buffer1[k]
                for k in keys2:
                    if k <= matched_seq: del self.buffer2[k]
                    
                if not self.frame_queue.full():
                    self.frame_queue.put(combined_frame)

    def _frame_callback1(self, trx_mask, data, seq_num, timestamp, range_offset, bin_length):
        try:
            tx_idx = int(trx_mask[1])
            radar_data = np.asarray(data, dtype=np.float32)
            if radar_data.shape[2] > self.range_bins:
                radar_data = radar_data[:, :, :self.range_bins]
                
            complex_data = (radar_data[:, 0, :] + 1j * radar_data[:, 1, :]).astype(np.complex64)
            self.virtual_frame1[tx_idx, :, :] = complex_data
            self.last_tx_data1[tx_idx] = True
            
            if all(self.last_tx_data1):
                self._process_assembled_frame(self.virtual_frame1.copy(), seq_num, 1)
                self.last_tx_data1 = [False, False]
        except Exception as e:
            pass
        return True

    def _frame_callback2(self, trx_mask, data, seq_num, timestamp, range_offset, bin_length):
        try:
            tx_idx = int(trx_mask[1])
            radar_data = np.asarray(data, dtype=np.float32)
            if radar_data.shape[2] > self.range_bins:
                radar_data = radar_data[:, :, :self.range_bins]
                
            complex_data = (radar_data[:, 0, :] + 1j * radar_data[:, 1, :]).astype(np.complex64)
            self.virtual_frame2[tx_idx, :, :] = complex_data
            self.last_tx_data2[tx_idx] = True
            
            if all(self.last_tx_data2):
                self._process_assembled_frame(self.virtual_frame2.copy(), seq_num, 2)
                self.last_tx_data2 = [False, False]
        except Exception as e:
            pass
        return True

    def get_frame(self):
        try: return self.frame_queue.get(timeout=2.0)
        except queue.Empty: return None

    def flush_queue(self):
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: break

    def close(self): 
        try:
            os.remove(self.preset1_path)
            os.remove(self.preset2_path)
        except:
            pass
        print("[INFO] X7 dual radar closed.")

# ── RECORDING LOGIC ─────────────────────────────────────────────────────
def record_trial(radar, duration_s, label):
    time.sleep(1.0)
    
    # Empty the buffer right before recording so we don't grab old frames
    radar.flush_queue()
    
    n_frames = int(duration_s * RADAR_FPS)
    # Shape: (Frames, 2 Sensors, 2 TX, 2 RX, Bins)
    frames = np.zeros((n_frames, 2, 2, 2, RANGE_BINS), dtype=np.complex64)
    collected = 0
    print(f"\n  ▶ Recording '{label}'...", end="")
    while collected < n_frames:
        frame = radar.get_frame() 
        if frame is not None:
            frames[collected] = frame
            collected += 1
            print(f"\r  ▶ Recording '{label}' ... {collected}/{n_frames}", end="", flush=True)
        else:
            print(f"\n[WARN] Timeout: No combined frame received. Check synchronization.", end="", flush=True)
            break
    return frames[:collected]

def run_gesture(args, radar):
    print("\nStarting dual-sensor gesture recording session...")
    for k, v in GESTURE_CLASSES.items():
        print(f"  [{k}] - {v[1]}")
    
    choice = input("\nEnter gesture ID (1-6): ").strip()
    if choice not in GESTURE_CLASSES: 
        print("Invalid choice.")
        return
        
    subject_id = input("Enter Subject ID (e.g., S01) [default S01]: ").strip()
    if not subject_id:
        subject_id = "S01"
        
    trials_str = input("Enter number of trials to record [default 1]: ").strip()
    try:
        n_trials = int(trials_str)
    except ValueError:
        n_trials = 1
        
    folder, desc = GESTURE_CLASSES[choice]
    out_dir = Path(args.out) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_trials):
        print(f"\n--- Trial {i+1} of {n_trials} ---")
        data = record_trial(radar, 2.5, desc)
        if len(data) > 0:
            filename = f"{subject_id}_{int(time.time())}_{i+1}.npy"
            np.save(out_dir / filename, data)
            print(f"\nSaved {len(data)} dual-sensor frames to {filename}.")

# ── ENTRY POINT ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host1", required=True, help="HostInterfaceId for Sensor 1 (e.g. 529)")
    parser.add_argument("--host2", required=True, help="HostInterfaceId for Sensor 2 (e.g. 545)")
    parser.add_argument("--mode", default="gesture", help="Run mode")
    parser.add_argument("--out", default="gesture_dataset_2s", help="Output directory")
    parser.add_argument("--subject", default="S01", help="Subject ID for data collection (e.g. S01, S02)")
    args = parser.parse_args()

    radar = X7DualRadar(args.host1, args.host2)
    try:
        if args.mode == "gesture":
            while True:
                run_gesture(args, radar)
                again = input("\nRecord another? (y/n): ").strip().lower()
                if again != 'y': break
    except KeyboardInterrupt:
        pass
    finally:
        radar.close()
