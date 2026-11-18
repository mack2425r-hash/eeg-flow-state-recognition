import os
import glob
import pandas as pd


# ============================================================
# DATASET VALIDATION
# ============================================================

DATASET_FOLDER = r"D:\paper\dataset"


print("\n==============================================")
print("       EEG FLOW DATASET VALIDATION")
print("==============================================\n")


# ------------------------------------------------------------
# 1. Find EEG and Feedback files
# ------------------------------------------------------------

eeg_files = sorted(
    glob.glob(
        os.path.join(DATASET_FOLDER, "*_eeg.csv")
    )
)

feedback_files = sorted(
    glob.glob(
        os.path.join(DATASET_FOLDER, "*_feedback.csv")
    )
)


print("EEG files found      :", len(eeg_files))
print("Feedback files found :", len(feedback_files))


# ------------------------------------------------------------
# 2. Display filenames
# ------------------------------------------------------------

print("\n==============================================")
print("EEG FILES")
print("==============================================")

for file in eeg_files:
    print(os.path.basename(file))


print("\n==============================================")
print("FEEDBACK FILES")
print("==============================================")

for file in feedback_files:
    print(os.path.basename(file))


# ------------------------------------------------------------
# 3. Check whether every subject has both files
# ------------------------------------------------------------

eeg_subjects = set()

for file in eeg_files:

    filename = os.path.basename(file)

    subject = filename.replace(
        "_eeg.csv",
        ""
    )

    eeg_subjects.add(subject)


feedback_subjects = set()

for file in feedback_files:

    filename = os.path.basename(file)

    subject = filename.replace(
        "_feedback.csv",
        ""
    )

    feedback_subjects.add(subject)


print("\n==============================================")
print("SUBJECT MATCHING")
print("==============================================")


missing_feedback = (
    eeg_subjects - feedback_subjects
)

missing_eeg = (
    feedback_subjects - eeg_subjects
)


if len(missing_feedback) == 0:

    print("✓ Every EEG file has feedback.")

else:

    print("WARNING: Missing feedback for:")

    for subject in sorted(missing_feedback):
        print("   ", subject)


if len(missing_eeg) == 0:

    print("✓ Every feedback file has EEG.")

else:

    print("WARNING: Missing EEG for:")

    for subject in sorted(missing_eeg):
        print("   ", subject)


# ------------------------------------------------------------
# 4. Inspect first EEG file
# ------------------------------------------------------------

if len(eeg_files) > 0:

    first_eeg = eeg_files[0]

    print("\n==============================================")
    print("FIRST EEG FILE")
    print("==============================================")

    print(
        "File:",
        os.path.basename(first_eeg)
    )

    eeg = pd.read_csv(
        first_eeg,
        nrows=10
    )

    print("\nColumns:")

    print(
        list(eeg.columns)
    )

    print("\nFirst 10 rows:")

    print(eeg)


# ------------------------------------------------------------
# 5. Inspect first feedback file
# ------------------------------------------------------------

if len(feedback_files) > 0:

    first_feedback = feedback_files[0]

    print("\n==============================================")
    print("FIRST FEEDBACK FILE")
    print("==============================================")

    print(
        "File:",
        os.path.basename(first_feedback)
    )

    feedback = pd.read_csv(
        first_feedback
    )

    print("\nColumns:")

    print(
        list(feedback.columns)
    )

    print("\nFeedback data:")

    print(feedback)


# ------------------------------------------------------------
# 6. Final summary
# ------------------------------------------------------------

print("\n==============================================")
print("SUMMARY")
print("==============================================")

print(
    "Total EEG files      :",
    len(eeg_files)
)

print(
    "Total feedback files :",
    len(feedback_files)
)

print(
    "Total subjects       :",
    len(
        eeg_subjects.intersection(
            feedback_subjects
        )
    )
)

print("\nDataset validation completed.")