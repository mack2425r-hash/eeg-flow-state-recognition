import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = r"D:\paper\advanced_training_data.csv"

MODEL_FILE = r"D:\paper\flow_random_forest.pkl"

RESULT_FILE = r"D:\paper\rf_results.txt"

IMPORTANCE_FILE = r"D:\paper\feature_importance.csv"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n==============================================")
print("RANDOM FOREST - FLOW STATE CLASSIFICATION")
print("==============================================")

data = pd.read_csv(DATA_FILE)

print("\nDataset Shape:", data.shape)

print("\nColumns:")
print(list(data.columns))


# ============================================================
# 2. CHECK DATASET
# ============================================================

print("\n==============================================")
print("DATASET CHECK")
print("==============================================")

print("\nMissing Values:")
print(data.isnull().sum())

print("\nInfinite Values:")

numeric_data = data.select_dtypes(
    include=[np.number]
)

print(
    np.isinf(numeric_data).sum()
)


# Replace infinite values if any

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows containing invalid values

before = len(data)

data = data.dropna()

after = len(data)

print(
    "\nRows removed due to invalid values:",
    before - after
)


# ============================================================
# 3. FEATURES AND LABEL
# ============================================================

X = data.drop(
    "Label",
    axis=1
)

y = data["Label"]


print("\n==============================================")
print("FEATURE INFORMATION")
print("==============================================")

print("\nNumber of Features:", X.shape[1])

print("\nFeatures:")

for i, feature in enumerate(
    X.columns,
    start=1
):
    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print("\n==============================================")
print("CLASS DISTRIBUTION")
print("==============================================")

print(
    y.value_counts().sort_index()
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n==============================================")
print("TRAIN / TEST SPLIT")
print("==============================================")

print(
    "Training Samples:",
    len(X_train)
)

print(
    "Testing Samples :",
    len(X_test)
)


print("\nTraining Class Distribution:")

print(
    y_train.value_counts().sort_index()
)


print("\nTesting Class Distribution:")

print(
    y_test.value_counts().sort_index()
)


# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)


print("\n==============================================")
print("TRAINING RANDOM FOREST")
print("==============================================")

model.fit(
    X_train,
    y_train
)

print("\nModel trained successfully.")


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================

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


cm = confusion_matrix(
    y_test,
    y_pred
)


report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Non-Flow",
        "Flow"
    ],
    zero_division=0
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 55)
print("RESULT 2 - RANDOM FOREST PERFORMANCE")
print("=" * 55)


print(
    f"\nAccuracy  : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
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


print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================\n")

print(cm)


print("\nInterpretation:")

print(
    f"True Non-Flow : {cm[0][0]}"
)

print(
    f"False Flow    : {cm[0][1]}"
)

print(
    f"False Non-Flow: {cm[1][0]}"
)

print(
    f"True Flow     : {cm[1][1]}"
)


print("\n==============================================")
print("CLASSIFICATION REPORT")
print("==============================================\n")

print(report)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        model.feature_importances_

})


importance = importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


print("\n==============================================")
print("FEATURE IMPORTANCE")
print("==============================================\n")

print(
    importance.to_string(
        index=False
    )
)


importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# 12. SAVE RESULT TEXT
# ============================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "RANDOM FOREST FLOW STATE CLASSIFICATION\n"
    )

    file.write(
        "=======================================\n\n"
    )

    file.write(
        f"Dataset Samples : {len(data)}\n"
    )

    file.write(
        f"Features        : {X.shape[1]}\n"
    )

    file.write(
        f"Training        : {len(X_train)}\n"
    )

    file.write(
        f"Testing         : {len(X_test)}\n\n"
    )

    file.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    file.write(
        f"Precision : {precision:.4f}\n"
    )

    file.write(
        f"Recall    : {recall:.4f}\n"
    )

    file.write(
        f"F1-Score  : {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC   : {roc_auc:.4f}\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nClassification Report:\n"
    )

    file.write(
        report
    )


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("FILES GENERATED")
print("==============================================")

print(
    "\nModel:",
    MODEL_FILE
)

print(
    "Results:",
    RESULT_FILE
)

print(
    "Feature Importance:",
    IMPORTANCE_FILE
)

print("\nResult 2 completed successfully.")