"""Print what the SDK actually reports for range_offset and bin_length.

Everything downstream has assumed 6.5 mm/bin from a code comment. This reads the
values the radar itself supplies, which is the only reliable source.
"""
import os
os.add_dll_directory(r"C:\RR_02\bin")
import sys, time, threading
import numpy as np

for p in [r"C:\RR_02\novelda-uwb-demos\Demos",
          r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect\X7RadarDirectCallback"]:
    if p not in sys.path:
        sys.path.insert(0, p)
from radar_direct_callback import RadarDirectCallback

PRESET = (r"C:\RR_02\novelda-uwb-demos\Demos\RadarDirect"
          r"\X7RadarDirectCallback\Presets\human_preset.json")
done = threading.Event()

def cb(trx_mask, data, seq_num, timestamp, range_offset, bin_length):
    if not done.is_set():
        a = np.asarray(data)
        n = a.shape[2]
        print("\n" + "=" * 58)
        print(" WHAT THE RADAR ACTUALLY REPORTS")
        print("=" * 58)
        print(f"  data shape        {a.shape}   (rx, iq, bins)")
        print(f"  range bins        {n}")
        print(f"  range_offset      {range_offset}")
        print(f"  bin_length        {bin_length}")
        try:
            bl = float(bin_length); ro = float(range_offset)
            print(f"\n  -> bin length     {bl*1000:.2f} mm")
            print(f"  -> coverage       {ro:.2f} m  to  {ro + (n-1)*bl:.2f} m")
        except Exception as e:
            print(f"  (could not convert: {e})")
        print("=" * 58)
        done.set()
    return not done.is_set()

r = RadarDirectCallback()
threading.Thread(target=r.run_with_callback_preset, args=(cb, PRESET), daemon=True).start()
if not done.wait(timeout=25):
    print("[ERROR] no frame within 25s")
time.sleep(0.5)
