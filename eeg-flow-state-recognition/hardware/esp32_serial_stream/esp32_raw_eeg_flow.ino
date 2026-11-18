/*
  ESP32-S3 LIVE/REPLAY EEG FLOW DETECTOR
  ======================================

  Training pipeline reproduced from build_advanced_dataset.py:

    Fs = 256 Hz
    Window = 512 samples (2 seconds)
    Non-overlapping windows

  Features, in exact training order:
    0  Mean
    1  Variance
    2  Std
    3  RMS
    4  Hjorth_Activity
    5  Hjorth_Mobility
    6  Hjorth_Complexity
    7  Delta
    8  Theta
    9  Alpha
    10 Beta
    11 Gamma
    12 Engagement_Index
    13 STFT_Energy
    14 Wavelet_Energy
    15 Shannon_Entropy
    16 Fractal_Dimension
    17 Hurst_Exponent

  Input:
    one raw EEG sample per line at 115200 baud

  The PC sender must send ONLY the CSV "value" column.
  Event/labels are never sent to the ESP32.
*/

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "flow_rf_optimized.h"

static const float FS = 256.0f;
static const int N = 512;

static float eeg[N];

static float re256[256], im256[256];
static float re128[128], im128[128];

static inline float sqf(float x) { return x * x; }

static void fft(float *re, float *im, int n) {
  // Bit reversal.
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }

  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * PI / (float)len;
    float wlen_r = cosf(ang);
    float wlen_i = sinf(ang);

    for (int i = 0; i < n; i += len) {
      float wr = 1.0f;
      float wi = 0.0f;

      for (int j = 0; j < len / 2; ++j) {
        int u = i + j;
        int v = i + j + len / 2;

        float vr = re[v] * wr - im[v] * wi;
        float vi = re[v] * wi + im[v] * wr;

        re[v] = re[u] - vr;
        im[v] = im[u] - vi;
        re[u] += vr;
        im[u] += vi;

        float nwr = wr * wlen_r - wi * wlen_i;
        wi = wr * wlen_i + wi * wlen_r;
        wr = nwr;
      }
    }
  }
}

/* -------------------- basic/time domain -------------------- */

static void basic_features(const float *x,
                            float &mean,
                            float &variance,
                            float &stddev,
                            float &rms) {
  double s = 0.0;
  double s2 = 0.0;

  for (int i = 0; i < N; ++i) {
    s += x[i];
    s2 += (double)x[i] * (double)x[i];
  }

  mean = (float)(s / N);
  variance = (float)((s2 / N) - (double)mean * mean);
  if (variance < 0.0f) variance = 0.0f;
  stddev = sqrtf(variance);
  rms = sqrtf((float)(s2 / N));
}

static void hjorth(const float *x, float &activity,
                   float &mobility, float &complexity) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0;
  float prev = x[0];

  for (int i = 0; i < N; ++i) {
    s0 += (double)x[i] * x[i];
  }

  float d1[511];
  for (int i = 0; i < N - 1; ++i) {
    d1[i] = x[i + 1] - x[i];
    s1 += (double)d1[i] * d1[i];
  }

  for (int i = 0; i < N - 2; ++i) {
    float d2 = d1[i + 1] - d1[i];
    s2 += (double)d2 * d2;
  }

  // np.var(signal), np.var(diff), np.var(diff(diff))
  // Calculate the exact population variance of each sequence.
  double mean0 = 0.0, mean1 = 0.0, mean2 = 0.0;

  for (int i = 0; i < N; ++i) mean0 += x[i];
  mean0 /= N;

  for (int i = 0; i < N - 1; ++i) mean1 += d1[i];
  mean1 /= (N - 1);

  for (int i = 0; i < N - 2; ++i) {
    float d2 = d1[i + 1] - d1[i];
    mean2 += d2;
  }
  mean2 /= (N - 2);

  double v0 = 0.0, v1 = 0.0, v2 = 0.0;

  for (int i = 0; i < N; ++i) v0 += sqf((float)(x[i] - mean0));
  for (int i = 0; i < N - 1; ++i) v1 += sqf(d1[i] - (float)mean1);
  for (int i = 0; i < N - 2; ++i) {
    float d2 = d1[i + 1] - d1[i];
    v2 += sqf(d2 - (float)mean2);
  }

  activity = (float)(v0 / N);

  mobility = sqrtf(
      (float)(v1 / (N - 1)) /
      (activity + 1e-10f)
  );

  complexity =
      (sqrtf(
          (float)(v2 / (N - 2)) /
          ((float)(v1 / (N - 1)) + 1e-10f)
       )
       /
       (mobility + 1e-10f));
}

/* -------------------- Welch band power -------------------- */

/*
  scipy.signal.welch:
    fs=256
    nperseg=min(len(signal),256) = 256
    default noverlap = nperseg//2 = 128
    default window = Hann (DFT-even / periodic)
    default detrend = constant
    scaling = density

  For a 512-sample window this gives three segments:
    [0..255], [128..383], [256..511]
*/
static float welch_bandpower(const float *x, float low, float high) {
  float average_psd[129];

  for (int k = 0; k <= 128; ++k) average_psd[k] = 0.0f;

  const float win_sum_sq = 96.0f;  // sum of periodic Hann^2 for N=256

  for (int seg = 0; seg < 3; ++seg) {
    int start = seg * 128;

    double mean = 0.0;
    for (int i = 0; i < 256; ++i) mean += x[start + i];
    mean /= 256.0;

    for (int i = 0; i < 256; ++i) {
      float w = 0.5f - 0.5f * cosf(2.0f * PI * i / 256.0f);
      re256[i] = ((float)(x[start + i] - mean)) * w;
      im256[i] = 0.0f;
    }

    fft(re256, im256, 256);

    for (int k = 0; k <= 128; ++k) {
      float p = (re256[k] * re256[k] + im256[k] * im256[k])
                / (FS * win_sum_sq);

      if (k != 0 && k != 128) p *= 2.0f;

      average_psd[k] += p / 3.0f;
    }
  }

  // Frequency spacing is exactly 1 Hz for N=256 at Fs=256.
  // scipy np.trapz over bins in the requested band.
  int k0 = (int)ceilf(low);
  int k1 = (int)floorf(high);

  if (k0 < 0) k0 = 0;
  if (k1 > 128) k1 = 128;
  if (k1 < k0) return 0.0f;

  float area = 0.0f;

  for (int k = k0; k <= k1; ++k) {
    float weight = 1.0f;
    if (k == k0 || k == k1) weight = 0.5f;
    area += average_psd[k] * weight;
  }

  return area;  // df = 1 Hz
}

/* -------------------- STFT energy -------------------- */

/*
  scipy.signal.stft:
    fs=256
    nperseg=128
    noverlap=64 (default)
    boundary='zeros'
    padded=True
    detrend='constant'
    scaling='spectrum'

  For N=512 this produces 9 frames with centers:
    0,64,128,...,512 samples.

  Each frame is zero-padded outside [0,511].
*/
static float stft_energy(const float *x) {
  float total = 0.0f;

  for (int frame = 0; frame < 9; ++frame) {
    int start = frame * 64 - 64;

    double mean = 0.0;
    int valid_count = 0;

    // scipy detrends the padded segment, including zeros.
    float seg[128];

    for (int i = 0; i < 128; ++i) {
      int idx = start + i;
      seg[i] = (idx >= 0 && idx < N) ? x[idx] : 0.0f;
      mean += seg[i];
      valid_count++;
    }

    mean /= 128.0;

    for (int i = 0; i < 128; ++i) {
      float w = 0.5f - 0.5f * cosf(2.0f * PI * i / 128.0f);
      re128[i] = (seg[i] - (float)mean) * w;
      im128[i] = 0.0f;
    }

    fft(re128, im128, 128);

    // scaling='spectrum' for STFT = FFT / sum(window), sum(window)=64.
    for (int k = 0; k <= 64; ++k) {
      float zr = re128[k] / 64.0f;
      float zi = im128[k] / 64.0f;
      total += zr * zr + zi * zi;
    }
  }

  return total;
}

/* -------------------- db4 wavelet energy -------------------- */

/*
  db4 decomposition filters from PyWavelets.
  The training code uses:
      pywt.wavedec(signal, "db4", level=4)
  and sums sum(c**2) over every coefficient array.

  Symmetric extension is used for the finite boundaries.
*/
static const float DB4_LO[8] = {
  -0.010597401785f,
   0.032883011667f,
   0.030841381836f,
  -0.187034811719f,
  -0.027983769417f,
   0.630880767930f,
   0.714846570553f,
   0.230377813309f
};

static const float DB4_HI[8] = {
  -0.230377813309f,
   0.714846570553f,
  -0.630880767930f,
  -0.027983769417f,
   0.187034811719f,
   0.030841381836f,
  -0.032883011667f,
  -0.010597401785f
};

/*
  Symmetric sample lookup.
  This implements the common "symmetric" mirror extension used by
  PyWavelets for finite-support filtering.
*/
static int sym_index(int i, int n) {
  if (n <= 1) return 0;

  while (i < 0 || i >= n) {
    if (i < 0) i = -i - 1;
    else i = 2 * n - 1 - i;
  }
  return i;
}

/*
  One db4 analysis stage.

  The implementation uses the PyWavelets db4 filters and symmetric
  extension. Coefficient lengths follow the standard symmetric
  extension rule.
*/
static int dwt_db4(const float *in, int in_n,
                   float *approx, float *detail) {
  const int out_n = (in_n + 8 - 1) / 2;

  for (int k = 0; k < out_n; ++k) {
    float a = 0.0f;
    float d = 0.0f;

    for (int j = 0; j < 8; ++j) {
      // Analysis filter bank, downsample by 2.
      int idx = 2 * k - 6 + j;
      int si = sym_index(idx, in_n);

      a += DB4_LO[j] * in[si];
      d += DB4_HI[j] * in[si];
    }

    approx[k] = a;
    detail[k] = d;
  }

  return out_n;
}

static float wavelet_energy(const float *x) {
  float a0[512];
  float a1[259], d1[259];
  float a2[132], d2[132];
  float a3[69],  d3[69];
  float a4[38],  d4[38];

  for (int i = 0; i < N; ++i) a0[i] = x[i];

  int n1 = dwt_db4(a0, 512, a1, d1);
  int n2 = dwt_db4(a1, n1, a2, d2);
  int n3 = dwt_db4(a2, n2, a3, d3);
  int n4 = dwt_db4(a3, n3, a4, d4);

  double e = 0.0;

  for (int i = 0; i < n4; ++i) {
    e += (double)a4[i] * a4[i];
    e += (double)d4[i] * d4[i];
  }

  for (int i = 0; i < n3; ++i) e += (double)d3[i] * d3[i];
  for (int i = 0; i < n2; ++i) e += (double)d2[i] * d2[i];
  for (int i = 0; i < n1; ++i) e += (double)d1[i] * d1[i];

  return (float)e;
}

/* -------------------- Shannon entropy -------------------- */

static float shannon_entropy(const float *x) {
  float xmin = x[0];
  float xmax = x[0];

  for (int i = 1; i < N; ++i) {
    if (x[i] < xmin) xmin = x[i];
    if (x[i] > xmax) xmax = x[i];
  }

  float hist[50];
  for (int i = 0; i < 50; ++i) hist[i] = 0.0f;

  if (xmax == xmin) return 0.0f;

  float width = (xmax - xmin) / 50.0f;

  for (int i = 0; i < N; ++i) {
    int b = (int)((x[i] - xmin) / width);
    if (b < 0) b = 0;
    if (b >= 50) b = 49;
    hist[b] += 1.0f;
  }

  // np.histogram(..., density=True) gives density values.
  // scipy.stats.entropy normalizes the supplied values internally,
  // so the resulting probability is simply count / total count.
  double H = 0.0;

  for (int i = 0; i < 50; ++i) {
    if (hist[i] > 0.0f) {
      double p = hist[i] / (double)N;
      H -= p * (log(p) / log(2.0));
    }
  }

  return (float)H;
}

/* -------------------- fractal dimension -------------------- */

static float fractal_dimension(const float *x) {
  double L = 0.0;

  for (int i = 0; i < N - 1; ++i) {
    L += fabs((double)x[i + 1] - x[i]);
  }

  float d = 0.0f;

  for (int i = 0; i < N; ++i) {
    float v = fabsf(x[i] - x[0]);
    if (v > d) d = v;
  }

  const double eps = 1e-10;
  double denominator =
      log10((double)N) +
      log10(((double)d + eps) / (L + eps));

  if (fabs(denominator) < 1e-12) return 0.0f;

  return (float)(log10((double)N) / denominator);
}

/* -------------------- Hurst exponent -------------------- */

static float hurst_exponent(const float *x) {
  double mean = 0.0;

  for (int i = 0; i < N; ++i) mean += x[i];
  mean /= N;

  double y = 0.0;
  double ymin = 0.0;
  double ymax = 0.0;

  for (int i = 0; i < N; ++i) {
    y += (double)x[i] - mean;
    if (i == 0) {
      ymin = y;
      ymax = y;
    } else {
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
  }

  double R = ymax - ymin;

  double var = 0.0;
  for (int i = 0; i < N; ++i) {
    double z = (double)x[i] - mean;
    var += z * z;
  }

  double S = sqrt(var / N);

  if (S == 0.0) return 0.0f;

  return (float)(
      log((R / S) + 1e-10) / log((double)N)
  );
}

/* -------------------- all 18 features -------------------- */

static void extractFeatures(const float *x, float f[18]) {
  basic_features(x, f[0], f[1], f[2], f[3]);

  hjorth(x, f[4], f[5], f[6]);

  f[7]  = welch_bandpower(x, 0.5f, 4.0f);
  f[8]  = welch_bandpower(x, 4.0f, 8.0f);
  f[9]  = welch_bandpower(x, 8.0f, 13.0f);
  f[10] = welch_bandpower(x, 13.0f, 30.0f);
  f[11] = welch_bandpower(x, 30.0f, 45.0f);

  f[12] = f[10] / (f[9] + f[8] + 1e-10f);

  f[13] = stft_energy(x);
  f[14] = wavelet_energy(x);

  f[15] = shannon_entropy(x);
  f[16] = fractal_dimension(x);
  f[17] = hurst_exponent(x);
}

/* -------------------- serial input -------------------- */

static bool readSample(float &sample) {
  if (!Serial.available()) return false;

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.length() == 0) return false;

  char *endptr = nullptr;
  float v = strtof(line.c_str(), &endptr);

  if (endptr == line.c_str()) return false;

  sample = v;
  return true;
}

static void printFeatures(const float f[18]) {
  const char *names[18] = {
    "Mean", "Variance", "Std", "RMS",
    "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
    "Delta", "Theta", "Alpha", "Beta", "Gamma",
    "Engagement_Index", "STFT_Energy", "Wavelet_Energy",
    "Shannon_Entropy", "Fractal_Dimension", "Hurst_Exponent"
  };

  for (int i = 0; i < 18; ++i) {
    Serial.print(names[i]);
    Serial.print(" = ");
    Serial.println(f[i], 8);
  }
}

static uint32_t windowNumber = 0;

static void processWindow() {
  ++windowNumber;

  float f[18];
  extractFeatures(eeg, f);

  float confidence = 0.0f;
  int prediction = rf_predict(f, &confidence);

  Serial.println();
  Serial.println("======================================");
  Serial.print("WINDOW ");
  Serial.println(windowNumber);
  Serial.println("======================================");

  printFeatures(f);

  Serial.println("--------------------------------------");
  Serial.print("Prediction : ");
  Serial.println(prediction == 1 ? "FLOW" : "NON-FLOW");

  Serial.print("Confidence : ");
  Serial.print(confidence, 2);
  Serial.println("%");

  Serial.println("--------------------------------------");
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("======================================");
  Serial.println(" ESP32-S3 EEG FLOW DETECTOR");
  Serial.println("======================================");
  Serial.println("Sampling rate : 256 Hz");
  Serial.println("Window        : 512 samples");
  Serial.println("Window length : 2 seconds");
  Serial.println("RF            : 100 trees");
  Serial.println("Features      : 18");
  Serial.println();
  Serial.println("Waiting for raw EEG samples...");
}

void loop() {
  static uint16_t index = 0;

  float sample;

  if (!readSample(sample)) return;

  eeg[index++] = sample;

  if (index >= N) {
    processWindow();
    index = 0;
  }
}
