"""
PC <-> ESP32-S3 EEG replay

Sends raw EEG samples from subject1_eeg.csv at 256 Hz and simultaneously
reads/prints ESP32 responses in this SAME terminal.

IMPORTANT:
- Do NOT open Arduino Serial Monitor.
- Close Arduino Serial Plotter too.
- This script owns COM9 while it is running.

Usage:
    pip install pandas pyserial
    python send_eeg_csv.py COM9

CSV format:
    time,value,event

Only the "value" column is sent.
The "event" column is NOT sent to the ESP32.
"""

import sys
import time
import threading
import pandas as pd
import serial
from serial import SerialException

CSV_FILE = "subject1_eeg.csv"
BAUD = 115200
FS = 256.0

if len(sys.argv) != 2:
    print("Usage: python send_eeg_csv.py COM_PORT")
    print("Example: python send_eeg_csv.py COM9")
    raise SystemExit(1)

PORT = sys.argv[1]

# ---------- Load CSV ----------
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"ERROR: Cannot find {CSV_FILE}")
    print("Put subject1_eeg.csv in the same folder as this script.")
    raise SystemExit(1)

required = {"time", "value", "event"}
missing = required - set(df.columns)

if missing:
    print(f"ERROR: CSV is missing columns: {sorted(missing)}")
    raise SystemExit(1)

values = pd.to_numeric(df["value"], errors="coerce").dropna().to_numpy()

print("=" * 60)
print("ESP32-S3 EEG CSV REPLAY")
print("=" * 60)
print(f"Samples       : {len(values)}")
print(f"Replay rate   : {FS:.0f} Hz")
print(f"Serial port   : {PORT}")
print(f"Baud rate     : {BAUD}")
print()
print("Arduino Serial Monitor must be CLOSED.")
print("This terminal will display the ESP32 output.")
print("=" * 60)

stop_reader = threading.Event()

def read_esp32():
    """Continuously print lines returned by the ESP32."""
    while not stop_reader.is_set():
        try:
            if ser.in_waiting:
                raw = ser.readline()
                if raw:
                    text = raw.decode("utf-8", errors="replace").rstrip()
                    if text:
                        print(f"\n[ESP32] {text}", flush=True)
            else:
                time.sleep(0.002)
        except Exception as e:
            if not stop_reader.is_set():
                print(f"\n[ESP32 reader error] {e}", flush=True)
            break

try:
    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=0.05,
        write_timeout=1
    )
except SerialException as e:
    print()
    print("ERROR: Could not open", PORT)
    print(e)
    print()
    print("Check that:")
    print("1. Arduino Serial Monitor is CLOSED.")
    print("2. Arduino Serial Plotter is CLOSED.")
    print("3. No other program is using", PORT)
    print("4. The ESP32-S3 is connected.")
    raise SystemExit(1)

time.sleep(2.0)
ser.reset_input_buffer()

reader = threading.Thread(target=read_esp32, daemon=True)
reader.start()

period = 1.0 / FS
next_time = time.perf_counter()

sent = 0
start_time = time.perf_counter()

try:
    print("\nStarting replay...\n", flush=True)

    for value in values:
        # pyserial requires BYTES, not a Python str.
        message = f"{float(value):.9f}\n".encode("ascii")
        ser.write(message)

        sent += 1

        next_time += period
        delay = next_time - time.perf_counter()

        if delay > 0:
            time.sleep(delay)

        if sent % 256 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"\r[SENDER] {sent}/{len(values)} samples "
                f"({sent / FS:.1f}s replay, {elapsed:.1f}s wall)",
                end="",
                flush=True
            )

    print("\n\n[SENDER] Replay complete.")
    print("[SENDER] Waiting for final ESP32 response...")

    # Allow the last 512-sample window to finish processing.
    time.sleep(3.0)

except KeyboardInterrupt:
    print("\n\n[SENDER] Stopped by user (Ctrl+C).")

except Exception as e:
    print(f"\n\n[SENDER ERROR] {e}")

finally:
    stop_reader.set()

    try:
        reader.join(timeout=1.0)
    except Exception:
        pass

    try:
        ser.close()
    except Exception:
        pass

print("[SENDER] Serial port closed.")
print("[SENDER] Done.")
