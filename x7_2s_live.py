import os
os.add_dll_directory(r"C:\RR_02\bin") 
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import collections

_DEMO_DIR = r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect\X7RadarDirectCallback"
_DEMO_PARENT = r"C:\RR_02\novelda-uwb-demos\Demos"
for path in [_DEMO_PARENT, _DEMO_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from x7_2s_record import X7DualRadar, RANGE_BINS, RADAR_FPS
from x7_2s_train import GestureCNN_X7_Dual, to_interferometry_tensor, N_CLASSES, GESTURE_NAMES

WINDOW_SIZE = 64
STATE_IDLE = 0
STATE_ACTIVE = 1

def run_live_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureCNN_X7_Dual(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    radar = X7DualRadar(args.host1, args.host2)
    time.sleep(2.0)
    radar.flush_queue()
    
    print("\n[INFO] Starting Live Inference...")
    print(f"[INFO] Classes: {GESTURE_NAMES}")
    print("\n" + "="*40)
    print(" 💤 IDLE - Waiting for PINCH gesture...")
    print("="*40)
    
    app_state = STATE_IDLE
    active_start_time = 0
    
    # Ring buffer for 150 frames to find the peak of a gesture
    frame_buffer = collections.deque(maxlen=150)
    frames_seen = 0
    
    radar_state = "IDLE"
    cooldown = 0
    
    try:
        while True:
            frame = radar.get_frame()
            if frame is None:
                continue
                
            frame_buffer.append(frame)
            frames_seen += 1
            if frames_seen < 150:
                continue
                
            if cooldown > 0:
                cooldown -= 1
                continue
            
            # 5-frame energy check
            recent_frames = list(frame_buffer)[-5:]
            recent_energy = np.sum(np.abs(recent_frames))
            
            if radar_state == "IDLE":
                if app_state == STATE_ACTIVE:
                    elapsed = time.time() - active_start_time
                    if elapsed > 5.0:
                        print("\n[TIMEOUT] 5 seconds passed with no gesture.")
                        print(" 💤 IDLE - Waiting for PINCH gesture...")
                        app_state = STATE_IDLE
                        continue
                        
                threshold = 800 if app_state == STATE_IDLE else 1000
                if recent_energy > threshold: # Motion detected
                    radar_state = "GATHERING"
                    # Wait 40 frames for PINCH, 50 frames for SWIPES (required for SWIPE_DU's late peak)
                    gather_frames = 40 if app_state == STATE_IDLE else 50
                    
            elif radar_state == "GATHERING":
                gather_frames -= 1
                if gather_frames <= 0:
                    buf_arr = np.stack(frame_buffer, axis=0)
                    energies = np.sum(np.abs(buf_arr), axis=(1, 2, 3, 4))
                    
                    if app_state == STATE_IDLE:
                        search_start = 100
                        search_end = 135
                    else:
                        search_start = 80
                        search_end = 150
                    
                    local_peak = np.argmax(energies[search_start:search_end])
                    peak_idx = search_start + local_peak
                    
                    # We want the peak to be at index 32
                    start_idx = peak_idx - 32
                    start_idx = max(0, min(150 - WINDOW_SIZE, start_idx))
                    
                    window = buf_arr[start_idx : start_idx + WINDOW_SIZE]
                    
                    t1 = to_interferometry_tensor(window[:, 0])
                    t2 = to_interferometry_tensor(window[:, 1])
                    x = torch.stack([t1, t2], dim=0).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits = model(x)
                        probs = F.softmax(logits, dim=1)[0]
                        pred_class = torch.argmax(probs).item()
                        confidence = probs[pred_class].item()
                        
                    if pred_class != 6: # Ignore NOISE prints
                        print(f"  [DEBUG] Energy: {recent_energy:.1f}, Pred: {GESTURE_NAMES[pred_class]} ({confidence*100:.1f}%)")
                    
                    if app_state == STATE_IDLE:
                        # Look for PINCH (Class index 5)
                        if pred_class == 5 and confidence >= 0.75:
                            print(f"\n[TRIGGER] PINCH detected! ({confidence*100:.1f}%)")
                            print(" 🟢 ACTIVE - Waiting for next gesture... (3s window)")
                            app_state = STATE_ACTIVE
                            active_start_time = time.time()
                            cooldown = 15 # extremely short cooldown after pinch to allow immediate gesture
                            radar_state = "IDLE"
                            continue
                        else:
                            cooldown = 5 # Extremely short cooldown if noise
                            radar_state = "IDLE"
                            continue
                            
                    elif app_state == STATE_ACTIVE:
                        # Ignore PINCH and NOISE in active state
                        if pred_class not in [5, 6]:
                            # Vertical swipes (UP/DOWN) are inherently harder with the 12-degree inward tilt.
                            # Lowering the confidence threshold for them accommodates the variance.
                            required_conf = 0.45 if pred_class in [2, 3] else 0.60
                            if confidence >= required_conf:
                                gesture_name = GESTURE_NAMES[pred_class]
                                print(f"\n>>> GESTURE RECOGNIZED: {gesture_name} ({confidence*100:.1f}%) <<<")
                                print("\n 💤 IDLE - Waiting for PINCH gesture...")
                                app_state = STATE_IDLE
                                cooldown = 20
                                radar_state = "IDLE"
                                continue
                            else:
                                cooldown = 5 # Extremely short cooldown if not recognized
                                radar_state = "IDLE"
                                continue
                        else:
                            cooldown = 5 # Extremely short cooldown if noise or pinch
                            radar_state = "IDLE"
                            continue
                            
                    radar_state = "IDLE"
                    cooldown = 40 # Fallback cooldown
                    
    except KeyboardInterrupt:
        print("\n[INFO] Live inference stopped.")
    finally:
        radar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r"models\best_model_2s.pth", help="Path to trained weights")
    parser.add_argument("--host1", required=True, help="HostInterfaceId for Sensor 1 (e.g. 529)")
    parser.add_argument("--host2", required=True, help="HostInterfaceId for Sensor 2 (e.g. 545)")
    args = parser.parse_args()
    
    run_live_inference(args)
