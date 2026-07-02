
# Multi-Model Clinical ECG Telemetry on ESP32-C3

Real-time ECG arrhythmia classification using an AD8232, ESP32-C3, and PyTorch. Benchmarks novel sequence models (KAN, S4, ConvKAN) against standard CNNs under simulated hardware noise.

## 🔬 Overview
Cardiovascular disease remains a leading cause of mortality[cite: 5]. While Convolutional Neural Networks (CNNs) dominate automated ECG analysis, this project benchmarks novel parameter-efficient architectures on the MIT-BIH Arrhythmia subset (109,446 beats, 5 classes)[cite: 5]. 

Models are evaluated not just on clean data, but under simulated hardware noise profiles specific to the ESP32-C3's internal 12-bit SAR ADC and external 16-bit ADCs (ADS1115)[cite: 3, 4]. 

**Key Findings:**
* **Baseline:** ResNet1D achieves the highest macro F1 score of 0.931 at 1.87M parameters[cite: 5].
* **Efficiency:** Kolmogorov-Arnold Networks (KAN) achieve a competitive 0.881 macro F1 with only 142k parameters, representing a 13-fold reduction in parameter count compared to ResNet1D[cite: 5]. KAN is identified as the optimal choice for resource-constrained cardiac monitoring[cite: 5].
* **Negative Result:** S4 models underperform on the 187-timestep fixed-length signals, suggesting their theoretical advantages are specific to much longer sequences[cite: 5].

## 🛠️ Hardware Architecture
The live telemetry system bypasses standard development boards in favor of a compact custom build:
* **Microcontroller:** ESP32-C3 Supermini[cite: 1, 3]
* **Analog Front-End:** AD8232 ECG Sensor[cite: 1, 3]
* **Baud Rate:** 115200[cite: 1]

## 🧠 Software & Models
The repository includes implementations of seven architectures[cite: 6]:
1. **ResNet1D** (Baseline)[cite: 6]
2. **TCN** (Temporal Convolutional Network)[cite: 6]
3. **TransformerECG**[cite: 6]
4. **KAN** (Kolmogorov-Arnold Network)[cite: 6]
5. **ConvKAN** (Hybrid CNN + KAN)[cite: 6]
6. **S4ECG** (Simplified S4 State Space Model)[cite: 6]
7. **BiGRU**[cite: 6]

To handle the severe class imbalance (class F is outnumbered 113:1 by class N), training utilizes a `WeightedRandomSampler` and Focal Loss[cite: 3, 5]. 

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/ecg-ml-telemetry.git](https://github.com/YOUR_USERNAME/ecg-ml-telemetry.git)
   cd ecg-ml-telemetry



2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Place the MIT-BIH Kaggle dataset CSV files into the `data/` directory.



## 💻 Usage

### 1. Training the Models

To train all models on the clean MIT-BIH dataset:

```bash
python src/run_experiments.py --epochs 30 --batch 256

```

### 2. Hardware Noise Benchmarking

To benchmark models under simulated AD8232 + ESP32-C3 hardware noise:

```bash
python src/run_noise_experiments.py --profile both --epochs 30

```

This generates comparative F1 degradation heatmaps and robustness scatter plots in `outputs/figures_noise/`.

### 3. Live GPU-Accelerated Telemetry UI

To run the live telemetry dashboard with hot-swappable ML models:

1. Flash your ESP32-C3 with your serial output firmware.
2. Update the `COM_PORT` variable in `src/live_log_tele_opt.py`.


3. Launch the UI:

```bash
python src/live_log_tele_opt.py

```

*The UI calculates rolling Heart Rate Variability (HRV-RMSSD), Beats Per Minute (BPM), and Respiratory Rate based on R-peak amplitude modulation*.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✍️ Author

**Richard Konsam** - North Eastern Regional Institute of Science and Technology (NERIST).


**OMITTED MIT-BIH DATASET UPLOAD DUE TO LARGE SIZE**
