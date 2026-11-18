/*
  ESP32-S3 LIVE EEG FLOW DETECTOR — STAGE 2: WEBSOCKET STREAMING
  ==============================================================
  Stage 2 Objective:
    - Runs Hotspot "ESP32-EEG-FLOW" (IP: 192.168.4.1)
    - Streams live EEG batches & Flow classifications to Python over WebSockets
    - Python Server Address: ws://192.168.4.2:8765
    - 256 Hz ADC sampling + 5-50 Hz Bandpass + RF ML inference
    - 10-bit Model Scaling Alignment (512 Baseline)
*/

#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <math.h>
#include <string.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "flow_rf_optimized.h"

using namespace websockets;

/* -------------------- Hotspot & Server Settings -------------------- */
const char* AP_SSID = "ESP32-EEG-FLOW";
const char* AP_PASS = "12345678";

// In AP mode, the laptop connected to the ESP32 is assigned 192.168.4.2
const char* PYTHON_SERVER_URL = "ws://192.168.4.2:8765";

WebsocketsClient wsClient;
static bool wsConnected = false;
static uint32_t lastWsRetryMs = 0;

/* -------------------- Configuration -------------------- */
#define USE_SERIAL_INPUT    0
#define ENABLE_50HZ_NOTCH   1

// Set to 1 to simulate a synthetic 10Hz Alpha/Theta flow wave (diagnostic test)
#define SIMULATE_FLOW_TEST  0

static const float FS = 256.0f;
static const int N = 512;
static const int EEG_PIN = 4;
static const uint32_t SAMPLE_PERIOD_US = 1000000UL / 256UL;

/* -------------------- OLED Setup -------------------- */
static const int OLED_SDA = 8;
static const int OLED_SCL = 9;
static const uint8_t OLED_ADDRESS = 0x3C;
static const int OLED_WIDTH = 128;
static const int OLED_HEIGHT = 64;
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);
static bool oledReady = false;

static void showOLED(const char *status, float confidence, int analogVal, uint32_t windowNo) {
  if (!oledReady) return;
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("EEG FLOW ");
  if (wsConnected) {
    display.print("[WS:LIVE]");
  } else {
    display.print("[WS:WAIT]");
  }
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  display.setTextSize(2);
  display.setCursor(0, 14);
  display.println(status);

  display.setTextSize(1);
  display.setCursor(0, 35);
  display.print("Confidence: ");
  display.print(confidence, 1);
  display.println("%");

  display.setCursor(0, 45);
  display.print("ADC: ");
  display.print(analogVal);
  display.print(" | Win:#");
  display.println(windowNo);

  display.setCursor(0, 55);
  display.print("IP: ");
  display.println(WiFi.softAPIP());

  display.display();
}

/* -------------------- 5 Hz - 50 Hz Bandpass & Notch Filters -------------------- */
struct Biquad {
  float b0, b1, b2;
  float a1, a2;
  float z1, z2;

  inline float process(float x) {
    float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
};

static Biquad bp_stage0 = {
  0.16866481f,  0.33732961f, 0.16866481f,
 -0.50964648f,  0.26688248f,
  0.0f, 0.0f
};

static Biquad bp_stage1 = {
  1.00000000f, -2.00000000f, 1.00000000f,
 -1.82987100f,  0.84601622f,
  0.0f, 0.0f
};

#if ENABLE_50HZ_NOTCH
static Biquad notch50 = {
  0.97995413f, -0.66027320f, 0.97995413f,
 -0.66027320f,  0.95990825f,
  0.0f, 0.0f
};
#endif

static float dcBias = 2048.0f;

static float filterSample(float raw_adc) {
  dcBias = 0.999f * dcBias + 0.001f * raw_adc;
  float centered = raw_adc - dcBias;

  float out = bp_stage0.process(centered);
  out = bp_stage1.process(out);

#if ENABLE_50HZ_NOTCH
  out = notch50.process(out);
#endif

  return out;
}

/* -------------------- Buffers & Math Helpers -------------------- */
static float eeg[N];
static float re256[256], im256[256];
static float re128[128], im128[128];
static int latestRawADC = 2048;

static float sampleBatch[16];
static uint8_t batchIdx = 0;

static inline float sqf(float x) { return x * x; }

static void fft(float *re, float *im, int n) {
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

/* -------------------- Feature 0-3: Basic Time Domain -------------------- */
static void basic_features(const float *x, float &mean, float &variance,
                           float &stddev, float &rms) {
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

/* -------------------- Feature 4-6: Hjorth Parameters -------------------- */
static void hjorth(const float *x, float &activity,
                   float &mobility, float &complexity) {
  float d1[511];
  for (int i = 0; i < N - 1; ++i) {
    d1[i] = x[i + 1] - x[i];
  }

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
      (float)(v1 / (N - 1)) / (activity + 1e-10f)
  );

  complexity = (sqrtf(
      (float)(v2 / (N - 2)) / ((float)(v1 / (N - 1)) + 1e-10f)
  ) / (mobility + 1e-10f));
}

/* -------------------- Feature 7-11: Welch Band Power -------------------- */
static float welch_bandpower(const float *x, float low, float high) {
  float average_psd[129];
  for (int k = 0; k <= 128; ++k) average_psd[k] = 0.0f;

  const float win_sum_sq = 96.0f;

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
      float p = (re256[k] * re256[k] + im256[k] * im256[k]) / (FS * win_sum_sq);
      if (k != 0 && k != 128) p *= 2.0f;
      average_psd[k] += p / 3.0f;
    }
  }

  int k0 = (int)ceilf(low);
  int k1 = (int)floorf(high);
  if (k0 < 0) k0 = 0;
  if (k1 > 128) k1 = 128;
  if (k1 < k0) return 0.0f;

  float area = 0.0f;
  for (int k = k0; k <= k1; ++k) {
    float weight = (k == k0 || k == k1) ? 0.5f : 1.0f;
    area += average_psd[k] * weight;
  }
  return area;
}

/* -------------------- Feature 13: STFT Energy -------------------- */
static float stft_energy(const float *x) {
  float total = 0.0f;

  for (int frame = 0; frame < 9; ++frame) {
    int start = frame * 64 - 64;
    double mean = 0.0;
    float seg[128];

    for (int i = 0; i < 128; ++i) {
      int idx = start + i;
      seg[i] = (idx >= 0 && idx < N) ? x[idx] : 0.0f;
      mean += seg[i];
    }
    mean /= 128.0;

    for (int i = 0; i < 128; ++i) {
      float w = 0.5f - 0.5f * cosf(2.0f * PI * i / 128.0f);
      re128[i] = (seg[i] - (float)mean) * w;
      im128[i] = 0.0f;
    }

    fft(re128, im128, 128);

    for (int k = 0; k <= 64; ++k) {
      float zr = re128[k] / 64.0f;
      float zi = im128[k] / 64.0f;
      total += zr * zr + zi * zi;
    }
  }
  return total;
}

/* -------------------- Feature 14: db4 Wavelet Energy -------------------- */
static const float DB4_LO[8] = {
  -0.010597401785f,  0.032883011667f,  0.030841381836f, -0.187034811719f,
  -0.027983769417f,  0.630880767930f,  0.714846570553f,  0.230377813309f
};

static const float DB4_HI[8] = {
  -0.230377813309f,  0.714846570553f, -0.630880767930f, -0.027983769417f,
   0.187034811719f,  0.030841381836f, -0.032883011667f, -0.010597401785f
};

static int sym_index(int i, int n) {
  if (n <= 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i - 1;
    else i = 2 * n - 1 - i;
  }
  return i;
}

static int dwt_db4(const float *in, int in_n, float *approx, float *detail) {
  const int out_n = (in_n + 8 - 1) / 2;
  for (int k = 0; k < out_n; ++k) {
    float a = 0.0f;
    float d = 0.0f;
    for (int j = 0; j < 8; ++j) {
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

/* -------------------- Feature 15: Shannon Entropy -------------------- */
static float shannon_entropy(const float *x) {
  float xmin = x[0], xmax = x[0];
  for (int i = 1; i < N; ++i) {
    if (x[i] < xmin) xmin = x[i];
    if (x[i] > xmax) xmax = x[i];
  }

  if (xmax == xmin) return 0.0f;

  float hist[50] = {0.0f};
  float width = (xmax - xmin) / 50.0f;

  for (int i = 0; i < N; ++i) {
    int b = (int)((x[i] - xmin) / width);
    if (b < 0) b = 0;
    if (b >= 50) b = 49;
    hist[b] += 1.0f;
  }

  double H = 0.0;
  for (int i = 0; i < 50; ++i) {
    if (hist[i] > 0.0f) {
      double p = hist[i] / (double)N;
      H -= p * (log(p) / log(2.0));
    }
  }
  return (float)H;
}

/* -------------------- Feature 16: Fractal Dimension -------------------- */
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
  double denominator = log10((double)N) + log10(((double)d + eps) / (L + eps));
  if (fabs(denominator) < 1e-12) return 0.0f;

  return (float)(log10((double)N) / denominator);
}

/* -------------------- Feature 17: Hurst Exponent -------------------- */
static float hurst_exponent(const float *x) {
  double mean = 0.0;
  for (int i = 0; i < N; ++i) mean += x[i];
  mean /= N;

  double y = 0.0, ymin = 0.0, ymax = 0.0;
  for (int i = 0; i < N; ++i) {
    y += (double)x[i] - mean;
    if (i == 0) {
      ymin = y; ymax = y;
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

  return (float)(log((R / S) + 1e-10) / log((double)N));
}

/* -------------------- Complete 18-Feature Extraction -------------------- */
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

/* -------------------- State Machine & Inference -------------------- */
static uint32_t nextSampleUs = 0;
static uint16_t sampleIndex = 0;
static uint32_t windowNumber = 0;

static void sendWebSocketResult(const char* state, float confidence) {
  if (!wsConnected) return;

  char jsonPayload[160];
  snprintf(jsonPayload, sizeof(jsonPayload),
    "{\"type\":\"result\",\"window\":%lu,\"state\":\"%s\",\"confidence\":%.2f,\"adc\":%d}",
    (unsigned long)windowNumber, state, confidence, latestRawADC);

  wsClient.send(jsonPayload);
}

static void sendWebSocketBatch() {
  if (!wsConnected) return;

  String json = "{\"type\":\"eeg\",\"samples\":[";
  for (uint8_t i = 0; i < 16; ++i) {
    json += String(sampleBatch[i], 2);
    if (i < 15) json += ",";
  }
  json += "]}";

  wsClient.send(json);
}

static void processWindow() {
  // Check if EXG Pill is disconnected or electrodes are off (valid range: 500 to 3600, centered ~2048)
  if (latestRawADC < 500 || latestRawADC > 3600) {
    ++windowNumber;
    showOLED("LEADS OFF", 0.0f, latestRawADC, windowNumber);
    sendWebSocketResult("LEADS OFF", 0.0f);
    Serial.printf("RESULT,%lu,LEADS OFF,0.00,ADC:%d\n",
                  (unsigned long)windowNumber, latestRawADC);
    return;
  }

  float f[18];
  extractFeatures(eeg, f);

  float confidence = 0.0f;
  int prediction = rf_predict(f, &confidence);
  ++windowNumber;

  const char *stateStr = (prediction == 1) ? "FLOW" : "NOT FLOW";

  showOLED(stateStr, confidence, latestRawADC, windowNumber);
  sendWebSocketResult(stateStr, confidence);

  Serial.printf("RESULT,%lu,%s,%.2f,ADC:%d\n",
                (unsigned long)windowNumber,
                stateStr,
                confidence,
                latestRawADC);
}

/* -------------------- WebSocket Connection Manager -------------------- */
static void checkWebSocket() {
  if (WiFi.softAPgetStationNum() > 0) {
    if (!wsConnected && (millis() - lastWsRetryMs > 3000)) {
      lastWsRetryMs = millis();
      Serial.print("Connecting to Python Server at ");
      Serial.println(PYTHON_SERVER_URL);

      wsConnected = wsClient.connect(PYTHON_SERVER_URL);
      if (wsConnected) {
        Serial.println("[+] WebSocket Connected to Python Server!");
      }
    }
  } else {
    wsConnected = false;
  }

  if (wsConnected) {
    wsClient.poll();
  }
}

/* -------------------- Setup Hotspot -------------------- */
static void startHotspot() {
  Serial.print("Starting Hotspot: ");
  Serial.println(AP_SSID);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);

  IPAddress myIP = WiFi.softAPIP();
  Serial.print("ESP32 Hotspot IP: ");
  Serial.println(myIP);

  if (oledReady) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("HOTSPOT CREATED!");
    display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
    display.setCursor(0, 16);
    display.print("SSID: "); display.println(AP_SSID);
    display.setCursor(0, 28);
    display.print("Pass: "); display.println(AP_PASS);
    display.setCursor(0, 42);
    display.print("IP:   "); display.println(myIP);
    display.display();
    delay(2000);
  }
}

/* -------------------- Setup -------------------- */
void setup() {
  Serial.begin(115200);

  Wire.begin(OLED_SDA, OLED_SCL);
  oledReady = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS);

  startHotspot();

  analogReadResolution(12);
  pinMode(EEG_PIN, INPUT);

#if (!USE_SERIAL_INPUT && !SIMULATE_FLOW_TEST)
  for (int i = 0; i < 512; ++i) {
    float raw = (float)analogRead(EEG_PIN);
    latestRawADC = (int)raw;
    filterSample(raw);
    delayMicroseconds(SAMPLE_PERIOD_US);
  }
#endif

  nextSampleUs = micros();
  Serial.println("STAGE_2_READY_ALIGNED_10BIT");
  showOLED("READY", 0.0f, latestRawADC, 0);
}

/* -------------------- Main Loop -------------------- */
void loop() {
  checkWebSocket();

#if USE_SERIAL_INPUT
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      eeg[sampleIndex++] = line.toFloat();
      if (sampleIndex >= N) {
        processWindow();
        sampleIndex = 0;
      }
    }
  }

#elif SIMULATE_FLOW_TEST
  // Synthetic 10 Hz Alpha + 6 Hz Theta wave to test FLOW triggering
  uint32_t now = micros();
  if ((int32_t)(now - nextSampleUs) < 0) return;
  nextSampleUs += SAMPLE_PERIOD_US;

  static float t = 0.0f;
  t += 1.0f / 256.0f;
  // 10 Hz Alpha (amplitude 25) + 6 Hz Theta (amplitude 15) around 512.0f
  float simSignal = 512.0f + 25.0f * sinf(2.0f * PI * 10.0f * t) + 15.0f * sinf(2.0f * PI * 6.0f * t);

  latestRawADC = (int)(simSignal * 4.0f); // fake 12-bit readout
  eeg[sampleIndex++] = simSignal;

  sampleBatch[batchIdx++] = simSignal - 512.0f;
  if (batchIdx >= 16) {
    sendWebSocketBatch();
    batchIdx = 0;
  }

  if (sampleIndex >= N) {
    processWindow();
    sampleIndex = 0;
  }

#else
  // Live Hardware Acquisition
  uint32_t now = micros();
  if ((int32_t)(now - nextSampleUs) < 0) return;
  nextSampleUs += SAMPLE_PERIOD_US;

  float rawSample = (float)analogRead(EEG_PIN);
  latestRawADC = (int)rawSample;

  // Filter 5-50 Hz & 50Hz notch
  float filtered = filterSample(rawSample);

  // Squelch floating noise if EXG Pill is disconnected (ADC < 500 or > 3600)
  float displaySample = (latestRawADC < 500 || latestRawADC > 3600) ? 0.0f : (filtered / 4.0f);

  // SCALE FIX: Scale 12-bit down to 10-bit (/ 4.0) and restore 512 baseline
  // This matches the exact scale flow_rf_optimized.h was trained on!
  float eeg_10bit = displaySample + 512.0f;
  eeg[sampleIndex++] = eeg_10bit;

  // Buffer live samples (centered AC waveform) for WebSockets
  sampleBatch[batchIdx++] = displaySample;
  if (batchIdx >= 16) {
    sendWebSocketBatch();
    batchIdx = 0;
  }

  if (sampleIndex >= N) {
    processWindow();
    sampleIndex = 0;
  }
#endif
}