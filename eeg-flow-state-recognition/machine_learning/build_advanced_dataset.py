import os
import glob
import re

import numpy as np
import pandas as pd
import pywt

from scipy.signal import welch, stft
from scipy.stats import entropy


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FOLDER = r"D:\paper\dataset"

OUTPUT_FEATURE_TABLE = r"D:\paper\advanced_feature_table.csv"
OUTPUT_TRAINING_DATA = r"D:\paper\advanced_training_data.csv"
OUTPUT_LABEL_SUMMARY = r"D:\paper\label_summary.csv"


SAMPLING_RATE = 256

WINDOW_SIZE = 2 * SAMPLING_RATE
# 256 × 2 = 512 samples


TASKS = [
    "C1_TASK1",
    "C1_TASK2",
    "C1_TASK3",
    "C2_TASK1",
    "C2_TASK2",
    "C2_TASK3"
]


difficulty_map = {
    "Easy": 1,
    "Medium": 2,
    "Hard": 3
}


# ============================================================
# BAND POWER
# ============================================================

def bandpower(signal, fs, low, high):

    f, pxx = welch(
        signal,
        fs=fs,
        nperseg=min(len(signal), 256)
    )

    mask = (
        (f >= low) &
        (f <= high)
    )

    if not np.any(mask):
        return 0.0

    return np.trapz(
        pxx[mask],
        f[mask]
    )


# ============================================================
# ENGAGEMENT INDEX
# ============================================================

def engagement_index(alpha, beta, theta):

    return beta / (
        alpha + theta + 1e-10
    )


# ============================================================
# HJORTH PARAMETERS
# ============================================================

def hjorth_parameters(signal):

    d1 = np.diff(signal)
    d2 = np.diff(d1)

    var0 = np.var(signal)
    var1 = np.var(d1)
    var2 = np.var(d2)

    activity = var0

    mobility = np.sqrt(
        var1 / (var0 + 1e-10)
    )

    complexity = (
        np.sqrt(
            var2 / (var1 + 1e-10)
        )
        /
        (mobility + 1e-10)
    )

    return (
        activity,
        mobility,
        complexity
    )


# ============================================================
# STFT ENERGY
# ============================================================

def stft_energy(signal):

    _, _, Z = stft(
        signal,
        fs=SAMPLING_RATE,
        nperseg=128
    )

    return np.sum(
        np.abs(Z) ** 2
    )


# ============================================================
# WAVELET ENERGY
# ============================================================

def wavelet_energy(signal):

    coeffs = pywt.wavedec(
        signal,
        "db4",
        level=4
    )

    return sum(
        np.sum(c ** 2)
        for c in coeffs
    )


# ============================================================
# SHANNON ENTROPY
# ============================================================

def shannon_entropy(signal):

    hist, _ = np.histogram(
        signal,
        bins=50,
        density=True
    )

    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    return entropy(
        hist,
        base=2
    )


# ============================================================
# FRACTAL DIMENSION
# ============================================================

def fractal_dimension(signal):

    L = np.sum(
        np.abs(np.diff(signal))
    )

    d = np.max(
        np.abs(signal - signal[0])
    )

    n = len(signal)

    denominator = (
        np.log10(n)
        +
        np.log10(
            (d + 1e-10)
            /
            (L + 1e-10)
        )
    )

    if abs(denominator) < 1e-12:
        return 0.0

    return (
        np.log10(n)
        /
        denominator
    )


# ============================================================
# HURST EXPONENT
# ============================================================

def hurst_exponent(signal):

    signal = np.asarray(signal)

    N = len(signal)

    if N <= 1:
        return 0.0

    Y = np.cumsum(
        signal - np.mean(signal)
    )

    R = np.max(Y) - np.min(Y)

    S = np.std(signal)

    if S == 0:
        return 0.0

    return (
        np.log(
            (R / S) + 1e-10
        )
        /
        np.log(N)
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(signal):

    features = {}

    # --------------------------------------------------------
    # Time Domain
    # --------------------------------------------------------

    features["Mean"] = np.mean(signal)

    features["Variance"] = np.var(signal)

    features["Std"] = np.std(signal)

    features["RMS"] = np.sqrt(
        np.mean(signal ** 2)
    )

    activity, mobility, complexity = (
        hjorth_parameters(signal)
    )

    features["Hjorth_Activity"] = activity

    features["Hjorth_Mobility"] = mobility

    features["Hjorth_Complexity"] = complexity

    # --------------------------------------------------------
    # Frequency Domain
    # --------------------------------------------------------

    delta = bandpower(
        signal,
        SAMPLING_RATE,
        0.5,
        4
    )

    theta = bandpower(
        signal,
        SAMPLING_RATE,
        4,
        8
    )

    alpha = bandpower(
        signal,
        SAMPLING_RATE,
        8,
        13
    )

    beta = bandpower(
        signal,
        SAMPLING_RATE,
        13,
        30
    )

    gamma = bandpower(
        signal,
        SAMPLING_RATE,
        30,
        45
    )

    features["Delta"] = delta

    features["Theta"] = theta

    features["Alpha"] = alpha

    features["Beta"] = beta

    features["Gamma"] = gamma

    features["Engagement_Index"] = (
        engagement_index(
            alpha,
            beta,
            theta
        )
    )

    # --------------------------------------------------------
    # Time-Frequency Domain
    # --------------------------------------------------------

    features["STFT_Energy"] = (
        stft_energy(signal)
    )

    features["Wavelet_Energy"] = (
        wavelet_energy(signal)
    )

    # --------------------------------------------------------
    # Non-Linear Features
    # --------------------------------------------------------

    features["Shannon_Entropy"] = (
        shannon_entropy(signal)
    )

    features["Fractal_Dimension"] = (
        fractal_dimension(signal)
    )

    features["Hurst_Exponent"] = (
        hurst_exponent(signal)
    )

    return features


# ============================================================
# CREATE LABEL DICTIONARY
# ============================================================

def create_label_dictionary(feedback_df):

    feedback_df = feedback_df.copy()

    feedback_df["difficulty_score"] = (
        feedback_df["difficulty"]
        .astype(str)
        .str.strip()
        .map(difficulty_map)
    )

    # Check for invalid difficulty values

    if feedback_df["difficulty_score"].isna().any():

        print("\nWARNING: Unknown difficulty value found.")

        print(
            feedback_df[
                feedback_df["difficulty_score"].isna()
            ]
        )

    feedback_df["focus"] = pd.to_numeric(
        feedback_df["focus"],
        errors="coerce"
    )

    feedback_df["enjoyment"] = pd.to_numeric(
        feedback_df["enjoyment"],
        errors="coerce"
    )

    feedback_df["flow_score"] = (
        feedback_df["focus"]
        +
        feedback_df["enjoyment"]
        -
        feedback_df["difficulty_score"]
    )

    feedback_df["label"] = (
        feedback_df["flow_score"] >= 8
    ).astype(int)

    label_dict = dict(
        zip(
            feedback_df["task"],
            feedback_df["label"]
        )
    )

    return feedback_df, label_dict


# ============================================================
# EXTRACT SUBJECT NUMBER
# ============================================================

def get_subject_number(filename):

    match = re.search(
        r"subject(\d+)",
        filename.lower()
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("      BUILDING ADVANCED EEG FEATURE DATASET")
    print("=" * 70)


    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    eeg_files = sorted(
        glob.glob(
            os.path.join(
                DATASET_FOLDER,
                "*_eeg.csv"
            )
        )
    )

    feedback_files = sorted(
        glob.glob(
            os.path.join(
                DATASET_FOLDER,
                "*_feedback.csv"
            )
        )
    )


    print("\nEEG files found      :", len(eeg_files))

    print(
        "Feedback files found :",
        len(feedback_files)
    )


    if len(eeg_files) == 0:

        print("\nERROR: No EEG files found.")

        return


    # --------------------------------------------------------
    # Create feedback lookup
    # --------------------------------------------------------

    feedback_lookup = {}

    for feedback_file in feedback_files:

        filename = os.path.basename(
            feedback_file
        )

        subject_number = get_subject_number(
            filename
        )

        if subject_number is None:
            continue

        feedback_lookup[
            subject_number
        ] = feedback_file


    # --------------------------------------------------------
    # Store all extracted rows
    # --------------------------------------------------------

    all_rows = []

    label_summary = []


    # --------------------------------------------------------
    # Process every EEG file
    # --------------------------------------------------------

    for subject_index, eeg_file in enumerate(
        eeg_files,
        start=1
    ):

        filename = os.path.basename(
            eeg_file
        )

        subject_number = get_subject_number(
            filename
        )


        print("\n")
        print("-" * 70)

        print(
            f"Processing Subject {subject_number}"
        )

        print("-" * 70)


        # ----------------------------------------------------
        # Find corresponding feedback
        # ----------------------------------------------------

        if subject_number not in feedback_lookup:

            print(
                "WARNING: Feedback file not found."
            )

            continue


        feedback_file = feedback_lookup[
            subject_number
        ]


        # ----------------------------------------------------
        # Read feedback
        # ----------------------------------------------------

        feedback_df = pd.read_csv(
            feedback_file
        )


        feedback_df, label_dict = (
            create_label_dictionary(
                feedback_df
            )
        )


        # ----------------------------------------------------
        # Print labels
        # ----------------------------------------------------

        print("\nTask Labels:")

        for task in TASKS:

            label = label_dict.get(
                task,
                None
            )

            if label is None:

                print(
                    f"  {task}: MISSING"
                )

            else:

                state = (
                    "FLOW"
                    if label == 1
                    else
                    "NON-FLOW"
                )

                print(
                    f"  {task}: "
                    f"{state} "
                    f"(Label = {label})"
                )


        # ----------------------------------------------------
        # Save label summary
        # ----------------------------------------------------

        for _, row in feedback_df.iterrows():

            task = row["task"]

            if task not in TASKS:
                continue

            label_summary.append({

                "Subject":
                    f"S{subject_number}",

                "Task":
                    task,

                "Focus":
                    row["focus"],

                "Difficulty":
                    row["difficulty"],

                "Enjoyment":
                    row["enjoyment"],

                "Difficulty_Score":
                    row["difficulty_score"],

                "Flow_Score":
                    row["flow_score"],

                "Label":
                    row["label"]

            })


        # ----------------------------------------------------
        # Read EEG
        # ----------------------------------------------------

        print("\nReading EEG...")

        eeg_df = pd.read_csv(
            eeg_file
        )


        required_columns = {
            "time",
            "value",
            "event"
        }


        if not required_columns.issubset(
            eeg_df.columns
        ):

            print(
                "ERROR: EEG columns are incorrect."
            )

            print(
                "Found:",
                list(eeg_df.columns)
            )

            continue


        print(
            "Total EEG samples:",
            len(eeg_df)
        )


        # ----------------------------------------------------
        # Clean EEG signal
        # ----------------------------------------------------

        eeg_df["value"] = pd.to_numeric(
            eeg_df["value"],
            errors="coerce"
        )

        eeg_df = eeg_df.dropna(
            subset=["value"]
        )


        # Fill event labels forward.
        #
        # If event is already present on every sample,
        # this does not change anything.
        #
        # If events occur only at transition points,
        # forward filling allows the following samples
        # to belong to that task.

        eeg_df["event"] = (
            eeg_df["event"]
            .astype(str)
            .replace(
                {
                    "nan": np.nan,
                    "": np.nan
                }
            )
            .ffill()
        )


        # ----------------------------------------------------
        # Display available events
        # ----------------------------------------------------

        available_events = sorted(
            eeg_df["event"]
            .dropna()
            .unique()
        )


        print("\nAvailable Events:")

        print(
            available_events
        )


        # ----------------------------------------------------
        # Process each task
        # ----------------------------------------------------

        subject_window_count = 0


        for task in TASKS:

            # -----------------------------------------------
            # Check label
            # -----------------------------------------------

            if task not in label_dict:

                print(
                    f"\nWARNING: "
                    f"{task} has no feedback label."
                )

                continue


            label = label_dict[
                task
            ]


            # -----------------------------------------------
            # Extract task signal
            # -----------------------------------------------

            task_signal = eeg_df.loc[
                eeg_df["event"] == task,
                "value"
            ].values


            print(
                f"\n{task}: "
                f"{len(task_signal)} samples"
            )


            if len(task_signal) < WINDOW_SIZE:

                print(
                    "  Not enough samples "
                    "for one complete window."
                )

                continue


            # -----------------------------------------------
            # Number of complete windows
            # -----------------------------------------------

            total_windows = (
                len(task_signal)
                //
                WINDOW_SIZE
            )


            print(
                f"  Windows: {total_windows}"
            )


            # -----------------------------------------------
            # 2-second non-overlapping windows
            # -----------------------------------------------

            for window_number in range(
                total_windows
            ):

                start = (
                    window_number
                    *
                    WINDOW_SIZE
                )

                end = (
                    start
                    +
                    WINDOW_SIZE
                )


                segment = task_signal[
                    start:end
                ]


                # -------------------------------------------
                # Extract 18 features
                # -------------------------------------------

                features = extract_features(
                    segment
                )


                # -------------------------------------------
                # Add metadata
                # -------------------------------------------

                features["Subject"] = (
                    f"S{subject_number}"
                )

                features["Task"] = task

                features["Window"] = (
                    window_number + 1
                )

                features["Label"] = label


                all_rows.append(
                    features
                )


                subject_window_count += 1


        print(
            f"\nSubject {subject_number} "
            f"total windows:",
            subject_window_count
        )


    # ========================================================
    # COMBINE ALL SUBJECTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("COMBINING ALL SUBJECTS")
    print("=" * 70)


    if len(all_rows) == 0:

        print(
            "\nERROR: No feature rows generated."
        )

        return


    feature_table = pd.DataFrame(
        all_rows
    )


    # --------------------------------------------------------
    # Arrange columns
    # --------------------------------------------------------

    feature_columns = [

        "Mean",
        "Variance",
        "Std",
        "RMS",

        "Hjorth_Activity",
        "Hjorth_Mobility",
        "Hjorth_Complexity",

        "Delta",
        "Theta",
        "Alpha",
        "Beta",
        "Gamma",

        "Engagement_Index",

        "STFT_Energy",
        "Wavelet_Energy",

        "Shannon_Entropy",
        "Fractal_Dimension",
        "Hurst_Exponent"

    ]


    metadata_columns = [

        "Subject",
        "Task",
        "Window",
        "Label"

    ]


    feature_table = feature_table[
        feature_columns
        +
        metadata_columns
    ]


    # ========================================================
    # SAVE ADVANCED FEATURE TABLE
    # ========================================================

    feature_table.to_csv(
        OUTPUT_FEATURE_TABLE,
        index=False
    )


    # ========================================================
    # CREATE TRAINING DATA
    # ========================================================

    training_data = feature_table[
        feature_columns
        +
        ["Label"]
    ]


    training_data.to_csv(
        OUTPUT_TRAINING_DATA,
        index=False
    )


    # ========================================================
    # SAVE LABEL SUMMARY
    # ========================================================

    label_summary_df = pd.DataFrame(
        label_summary
    )


    label_summary_df.to_csv(
        OUTPUT_LABEL_SUMMARY,
        index=False
    )


    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DATASET GENERATION COMPLETED")
    print("=" * 70)


    print(
        "\nAdvanced Feature Table Shape:",
        feature_table.shape
    )


    print(
        "Training Dataset Shape:",
        training_data.shape
    )


    print(
        "\nNumber of EEG windows:",
        len(feature_table)
    )


    print(
        "Number of Features:",
        len(feature_columns)
    )


    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counts = (
        feature_table["Label"]
        .value_counts()
        .sort_index()
    )


    non_flow = class_counts.get(
        0,
        0
    )

    flow = class_counts.get(
        1,
        0
    )


    print("\n==============================================")
    print("CLASS DISTRIBUTION")
    print("==============================================")


    print(
        "Non-Flow:",
        non_flow
    )

    print(
        "Flow:",
        flow
    )


    print(
        "Total:",
        len(feature_table)
    )


    # --------------------------------------------------------
    # Subject distribution
    # --------------------------------------------------------

    print("\n==============================================")
    print("SUBJECT-WISE WINDOW COUNT")
    print("==============================================")


    subject_counts = (
        feature_table["Subject"]
        .value_counts()
        .sort_index(
            key=lambda x: x.str.extract(
                r"(\d+)"
            )[0].astype(int)
        )
    )


    print(
        subject_counts
    )


    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    print("\n==============================================")
    print("OUTPUT FILES")
    print("==============================================")


    print(
        "\n1.",
        OUTPUT_FEATURE_TABLE
    )

    print(
        "2.",
        OUTPUT_TRAINING_DATA
    )

    print(
        "3.",
        OUTPUT_LABEL_SUMMARY
    )


    print("\nDone.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()