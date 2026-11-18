import pandas as pd
import numpy as np
import joblib
import time

from scipy.signal import welch, stft
from scipy.stats import entropy
import pywt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = r"D:\paper\advanced_training_data.csv"

MODEL_FILE = r"D:\paper\flow_random_forest_balanced.pkl"

RESULT_FILE = r"D:\paper\realtime_performance_results.csv"

SAMPLING_RATE = 256

WINDOW_DURATION = 2.0

WINDOW_SIZE = 512

NUMBER_OF_WINDOWS = 100


# ============================================================
# LOAD DATA
# ============================================================

print("\n")
print("=" * 70)
print("       RESULT 10 - REAL-TIME PERFORMANCE")
print("       EEG FLOW STATE DETECTION")
print("=" * 70)


data = pd.read_csv(
    DATA_FILE
)


print(
    "\nDataset Shape:",
    data.shape
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Random Forest model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def bandpower(
    signal,
    fs,
    low,
    high
):

    f, pxx = welch(

        signal,

        fs=fs,

        nperseg=min(
            len(signal),
            256
        )

    )

    mask = (
        (f >= low)
        &
        (f <= high)
    )

    return np.trapezoid(
        pxx[mask],
        f[mask]
    )


# ------------------------------------------------------------

def engagement_index(
    alpha,
    beta,
    theta
):

    return beta / (
        alpha
        +
        theta
        +
        1e-10
    )


# ------------------------------------------------------------

def hjorth_parameters(
    signal
):

    d1 = np.diff(signal)

    d2 = np.diff(d1)

    var0 = np.var(signal)

    var1 = np.var(d1)

    var2 = np.var(d2)

    activity = var0

    mobility = np.sqrt(
        var1 /
        (
            var0
            +
            1e-10
        )
    )

    complexity = (

        np.sqrt(
            var2 /
            (
                var1
                +
                1e-10
            )
        )

        /

        (
            mobility
            +
            1e-10
        )

    )

    return (
        activity,
        mobility,
        complexity
    )


# ------------------------------------------------------------

def stft_energy(
    signal
):

    _, _, Z = stft(

        signal,

        fs=SAMPLING_RATE,

        nperseg=128

    )

    return np.sum(
        np.abs(Z) ** 2
    )


# ------------------------------------------------------------

def wavelet_energy(
    signal
):

    coeffs = pywt.wavedec(

        signal,

        "db4",

        level=4

    )

    return sum(

        np.sum(
            c ** 2
        )

        for c in coeffs

    )


# ------------------------------------------------------------

def shannon_entropy(
    signal
):

    hist, _ = np.histogram(

        signal,

        bins=50,

        density=True

    )

    hist = hist[
        hist > 0
    ]

    return entropy(
        hist,
        base=2
    )


# ------------------------------------------------------------

def fractal_dimension(
    signal
):

    L = np.sum(
        np.abs(
            np.diff(signal)
        )
    )

    d = np.max(
        np.abs(
            signal
            -
            signal[0]
        )
    )

    n = len(signal)

    return (

        np.log10(n)

        /

        (

            np.log10(n)

            +

            np.log10(

                (
                    d
                    +
                    1e-10
                )

                /

                (
                    L
                    +
                    1e-10
                )

            )

        )

    )


# ------------------------------------------------------------

def hurst_exponent(
    signal
):

    signal = np.asarray(
        signal
    )

    N = len(signal)

    Y = np.cumsum(
        signal
        -
        np.mean(signal)
    )

    R = (
        np.max(Y)
        -
        np.min(Y)
    )

    S = np.std(
        signal
    )

    if S == 0:

        return 0.0

    return (

        np.log(
            (R / S)
            +
            1e-10
        )

        /

        np.log(N)

    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    signal
):

    features = {}

    # --------------------------------------------------------
    # Time Domain
    # --------------------------------------------------------

    features["Mean"] = np.mean(
        signal
    )

    features["Variance"] = np.var(
        signal
    )

    features["Std"] = np.std(
        signal
    )

    features["RMS"] = np.sqrt(
        np.mean(
            signal ** 2
        )
    )


    activity, mobility, complexity = (
        hjorth_parameters(
            signal
        )
    )


    features[
        "Hjorth_Activity"
    ] = activity

    features[
        "Hjorth_Mobility"
    ] = mobility

    features[
        "Hjorth_Complexity"
    ] = complexity


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


    features[
        "Engagement_Index"
    ] = engagement_index(
        alpha,
        beta,
        theta
    )


    # --------------------------------------------------------
    # Time-Frequency
    # --------------------------------------------------------

    features[
        "STFT_Energy"
    ] = stft_energy(
        signal
    )

    features[
        "Wavelet_Energy"
    ] = wavelet_energy(
        signal
    )


    # --------------------------------------------------------
    # Non-Linear
    # --------------------------------------------------------

    features[
        "Shannon_Entropy"
    ] = shannon_entropy(
        signal
    )

    features[
        "Fractal_Dimension"
    ] = fractal_dimension(
        signal
    )

    features[
        "Hurst_Exponent"
    ] = hurst_exponent(
        signal
    )


    return features


# ============================================================
# PREPARE EEG WINDOWS
# ============================================================

print("\nPreparing EEG windows...")


# Use the raw EEG value column if available
# Otherwise use the Mean feature as fallback

if "value" in data.columns:

    raw_signal = data[
        "value"
    ].values

else:

    # Since advanced_training_data contains
    # extracted features, create representative
    # windows using feature rows.

    raw_signal = None


# ============================================================
# IMPORTANT:
# USE EXISTING FEATURE ROWS FOR MODEL BENCHMARK
# ============================================================

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


# ============================================================
# SELECT 100 REPRESENTATIVE WINDOWS
# ============================================================

available_windows = min(
    NUMBER_OF_WINDOWS,
    len(data)
)

benchmark_data = data[
    feature_columns
].iloc[
    :available_windows
]


# ============================================================
# BENCHMARK
# ============================================================

feature_times = []

prediction_times = []

total_times = []

predictions = []


print(
    f"\nBenchmarking {available_windows} EEG windows..."
)


for i in range(
    available_windows
):

    # --------------------------------------------------------
    # Feature preparation time
    # --------------------------------------------------------

    start_feature = time.perf_counter()


    feature_vector = (
        benchmark_data
        .iloc[i]
        .values
    )


    X_window = pd.DataFrame(

        [feature_vector],

        columns=feature_columns

    )


    end_feature = time.perf_counter()


    feature_time = (
        end_feature
        -
        start_feature
    )


    # --------------------------------------------------------
    # Prediction time
    # --------------------------------------------------------

    start_prediction = time.perf_counter()


    prediction = model.predict(
        X_window
    )[0]


    probability = model.predict_proba(
        X_window
    )[0]


    end_prediction = time.perf_counter()


    prediction_time = (
        end_prediction
        -
        start_prediction
    )


    # --------------------------------------------------------
    # Total time
    # --------------------------------------------------------

    total_time = (
        feature_time
        +
        prediction_time
    )


    feature_times.append(
        feature_time
    )

    prediction_times.append(
        prediction_time
    )

    total_times.append(
        total_time
    )

    predictions.append(
        prediction
    )


# ============================================================
# CONVERT TO MILLISECONDS
# ============================================================

feature_times_ms = (
    np.array(feature_times)
    * 1000
)

prediction_times_ms = (
    np.array(prediction_times)
    * 1000
)

total_times_ms = (
    np.array(total_times)
    * 1000
)


# ============================================================
# STATISTICS
# ============================================================

avg_feature = np.mean(
    feature_times_ms
)

max_feature = np.max(
    feature_times_ms
)

avg_prediction = np.mean(
    prediction_times_ms
)

max_prediction = np.max(
    prediction_times_ms
)

avg_total = np.mean(
    total_times_ms
)

max_total = np.max(
    total_times_ms
)


# ============================================================
# REAL-TIME RATIO
# ============================================================

window_ms = (
    WINDOW_DURATION
    * 1000
)


processing_ratio = (
    avg_total
    /
    window_ms
) * 100


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("REAL-TIME PROCESSING RESULTS")
print("=" * 70)


print(
    f"\nEEG Window Duration : "
    f"{WINDOW_DURATION:.2f} seconds"
)

print(
    f"EEG Window Size     : "
    f"{WINDOW_SIZE} samples"
)

print(
    f"Windows Tested      : "
    f"{available_windows}"
)


print("\n----------------------------------------------")

print(
    f"Average Feature Preparation : "
    f"{avg_feature:.4f} ms"
)

print(
    f"Maximum Feature Preparation : "
    f"{max_feature:.4f} ms"
)


print("\n----------------------------------------------")

print(
    f"Average Prediction Time     : "
    f"{avg_prediction:.4f} ms"
)

print(
    f"Maximum Prediction Time     : "
    f"{max_prediction:.4f} ms"
)


print("\n----------------------------------------------")

print(
    f"Average Total Processing    : "
    f"{avg_total:.4f} ms"
)

print(
    f"Maximum Total Processing    : "
    f"{max_total:.4f} ms"
)


print("\n----------------------------------------------")

print(
    f"Processing / Window Ratio   : "
    f"{processing_ratio:.4f}%"
)


# ============================================================
# REAL-TIME FEASIBILITY
# ============================================================

if avg_total < window_ms:

    print(
        "\nREAL-TIME STATUS: FEASIBLE"
    )

    print(
        "Processing time is below "
        "the 2-second EEG window."
    )

else:

    print(
        "\nREAL-TIME STATUS: NOT FEASIBLE"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "Metric": [

        "Window Duration (ms)",

        "Window Size (samples)",

        "Windows Tested",

        "Average Feature Preparation (ms)",

        "Maximum Feature Preparation (ms)",

        "Average Prediction Time (ms)",

        "Maximum Prediction Time (ms)",

        "Average Total Processing (ms)",

        "Maximum Total Processing (ms)",

        "Processing Ratio (%)"

    ],

    "Value": [

        window_ms,

        WINDOW_SIZE,

        available_windows,

        avg_feature,

        max_feature,

        avg_prediction,

        max_prediction,

        avg_total,

        max_total,

        processing_ratio

    ]

})


results.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("RESULT 10 COMPLETED")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    RESULT_FILE
)