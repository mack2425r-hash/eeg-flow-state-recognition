# Cognitive Flow State Recognition

> **Repository Description:**  
> Cognitive Flow State Recognition using single-channel wearable EEG (BioAmp EXG Pill) and multi-domain feature fusion, deployed on ESP32-S3 microcontroller.

**Topics & Keywords:**  
`eeg` | `signal-processing` | `tinyml` | `esp32-s3` | `random-forest` | `bioamp` | `wearables` | `machine-learning`

---

## Directory Layout

```
eeg-flow-state-recognition/
├── hardware/
│   ├── esp32_s3_live_eeg_flow_oled/
│   │   ├── esp32_s3_live_eeg_flow_oled.ino      # ESP32-S3 firmware
│   │   └── flow_rf_optimized.h                  # Pre-trained Random Forest model in C
│   └── esp32_serial_stream/
│       ├── esp32_raw_eeg_flow.ino               # Raw acquisition firmware
│       └── send_eeg_csv.py                      # Python serial stream bridge
│
├── machine_learning/
│   ├── build_advanced_dataset.py                # 18-feature extraction pipeline
│   ├── result_02_random_forest.py               # Random forest classification
│   ├── result_03_balanced_rf.py                 # Balanced weighting
│   ├── result_04_feature_groups.py              # Domain performance comparisons
│   ├── result_05_confusion_matrix.py            # Confusion matrix generation
│   ├── result_10_realtime_performance.py        # Latency benchmarking
│   └── flow_random_forest_optimized.pkl         # Lightweight trained model
│
├── paper/
│   ├── main.tex                                 # Springer LNCS LaTeX manuscript
│   ├── llncs.cls                                # Document class
│   └── fig/                                     # Publication figures
│
├── data/
│   ├── sample_subject1_eeg.csv                  # Sample EEG recording (256 Hz)
│   └── sample_subject1_feedback.csv             # Self-report score logs
│
├── .gitignore                                   # Large file filter (omits >100MB files)
├── requirements.txt                             # Python dependencies
└── README.md
```

---

## System Overview

1. **Acquisition:** Single-channel frontal EEG ($FP_1 - FP_2$) recorded using the BioAmp EXG Pill analog front-end at 256 Hz.
2. **Feature Extraction:** 18 features computed per 2-second window across time, frequency, time-frequency, and non-linear domains.
3. **Classification:** Random Forest distinguishing flow from non-flow cognitive states (85.40% accuracy, 88.84% F1, 82.36% balanced accuracy).
4. **Embedded Inference:** Pre-trained model ported to an ESP32-S3 microcontroller (`flow_rf_optimized.h`) with 72.8 ms execution latency.

---

## Instructions

### Python Signal Processing & Evaluation
```bash
pip install -r requirements.txt
cd machine_learning
python result_02_random_forest.py
```

### Hardware Deployment
1. Open `hardware/esp32_s3_live_eeg_flow_oled/esp32_s3_live_eeg_flow_oled.ino` in Arduino IDE.
2. Select board: **ESP32-S3 Dev Module**.
3. Verify that `flow_rf_optimized.h` is in the same sketch folder.
4. Upload to the microcontroller.
