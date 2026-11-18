ESP32-S3 EXACT 18-FEATURE FLOW DETECTOR
==========================================

Source:
  build_advanced_dataset.py

Training settings:
  Fs = 256 Hz
  window = 512 samples
  non-overlapping windows

The ESP32 receives ONLY raw EEG values from the PC.
It does not receive event labels.

The sketch calculates:
  Mean
  Variance
  Std
  RMS
  Hjorth Activity
  Hjorth Mobility
  Hjorth Complexity
  Delta/Theta/Alpha/Beta/Gamma Welch band powers
  Engagement Index
  STFT Energy
  db4 Wavelet Energy
  Shannon Entropy
  Fractal Dimension
  Hurst Exponent

Then it calls the 100-tree Random Forest.

IMPORTANT NUMERICAL PARITY NOTE:
The formulas and parameters are ported from the Python training code.
The Welch/STFT calculations are implemented directly in C.
The db4 wavelet uses the same db4 analysis filters and symmetric
finite-boundary approach. Before using this as a scientific result,
compare ESP32 feature values against Python on the same 512-sample
window and confirm numerical tolerance.

Arduino:
  Put all files in one sketch folder:
    esp32_raw_eeg_flow.ino
    flow_rf_optimized.h

PC:
  Put subject1_eeg.csv beside send_eeg_csv.py.
  Install:
    pip install pandas pyserial

  Run:
    python send_eeg_csv.py COM7

  Replace COM7 with your ESP32-S3 port.

Serial Monitor:
  115200 baud.

The ESP32 prints all 18 features and the Random Forest prediction
for every 512-sample window.
