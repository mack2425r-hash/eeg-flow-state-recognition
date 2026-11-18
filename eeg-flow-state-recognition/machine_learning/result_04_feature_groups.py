import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = r"D:\paper\advanced_training_data.csv"

RANDOM_STATE = 42


# ============================================================
# FEATURE GROUPS
# ============================================================

TIME_FEATURES = [
    "Mean",
    "Variance",
    "Std",
    "RMS",
    "Hjorth_Activity",
    "Hjorth_Mobility",
    "Hjorth_Complexity"
]


FREQUENCY_FEATURES = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma",
    "Engagement_Index"
]


TIME_FREQUENCY_FEATURES = [
    "STFT_Energy",
    "Wavelet_Energy"
]


NONLINEAR_FEATURES = [
    "Shannon_Entropy",
    "Fractal_Dimension",
    "Hurst_Exponent"
]


ALL_FEATURES = (
    TIME_FEATURES
    +
    FREQUENCY_FEATURES
    +
    TIME_FREQUENCY_FEATURES
    +
    NONLINEAR_FEATURES
)


FEATURE_GROUPS = {

    "Time Domain":
        TIME_FEATURES,

    "Frequency Domain":
        FREQUENCY_FEATURES,

    "Time-Frequency":
        TIME_FREQUENCY_FEATURES,

    "Non-Linear":
        NONLINEAR_FEATURES,

    "All 18 Features":
        ALL_FEATURES

}


# ============================================================
# LOAD DATA
# ============================================================

print("\n")
print("=" * 70)
print("       FEATURE GROUP ANALYSIS")
print("       EEG FLOW / NON-FLOW CLASSIFICATION")
print("=" * 70)


data = pd.read_csv(DATA_FILE)


print(
    "\nDataset Shape:",
    data.shape
)


# ============================================================
# CLEAN DATA
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna()


y = data["Label"]


# ============================================================
# CREATE ONE COMMON TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(
    len(data)
)


train_indices, test_indices = train_test_split(

    indices,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


y_train = y.iloc[
    train_indices
]

y_test = y.iloc[
    test_indices
]


print("\nTraining samples:", len(train_indices))

print("Testing samples :", len(test_indices))


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# RUN EACH FEATURE GROUP
# ============================================================

for group_name, feature_list in FEATURE_GROUPS.items():

    print("\n")
    print("=" * 70)

    print(
        f"FEATURE GROUP: {group_name}"
    )

    print("=" * 70)

    print(
        "\nNumber of Features:",
        len(feature_list)
    )

    print(
        "Features:"
    )

    for feature in feature_list:

        print(
            "  -",
            feature
        )


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    X = data[
        feature_list
    ]


    X_train = X.iloc[
        train_indices
    ]

    X_test = X.iloc[
        test_indices
    ]


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=300,

        criterion="entropy",

        max_depth=None,

        max_features="sqrt",

        min_samples_split=2,

        min_samples_leaf=1,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1

    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.append({

        "Feature_Group":
            group_name,

        "Number_of_Features":
            len(feature_list),

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_Score":
            f1,

        "ROC_AUC":
            roc_auc

    })


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    r"D:\paper\feature_group_results.csv"
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL FEATURE GROUP COMPARISON")
print("=" * 70)

print()

print(
    results_df.to_string(
        index=False,
        formatters={

            "Accuracy":
                "{:.4f}".format,

            "Precision":
                "{:.4f}".format,

            "Recall":
                "{:.4f}".format,

            "F1_Score":
                "{:.4f}".format,

            "ROC_AUC":
                "{:.4f}".format

        }
    )
)


print("\n")
print(
    "Results saved to:"
)

print(
    OUTPUT_FILE
)


print("\nFeature group analysis completed.")