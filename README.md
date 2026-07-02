# Multi-Model Clinical ECG Telemetry on ESP32-C3

Real-time ECG arrhythmia classification leveraging an AD8232 Analog Front-End, a custom ESP32-C3 Supermini build, and PyTorch. This repository contains the training pipelines, hardware noise simulations, and a live GPU-accelerated PyQtGraph telemetry UI for benchmarking novel sequence models against established CNNs.

## 🔬 Overview
Cardiovascular disease remains a leading cause of mortality. While Convolutional Neural Networks (CNNs) dominate automated ECG analysis, this project benchmarks novel parameter-efficient architectures on the MIT-BIH Arrhythmia subset (109,446 beats, 5 classes). 

Models are evaluated not just on clean data, but under simulated hardware noise profiles specific to the ESP32-C3's internal 12-bit SAR ADC and external 16-bit ADCs (ADS1115). 

**Key Findings:**
* **Baseline:** ResNet1D achieves the highest macro F1 score of 0.931 at 1.87M parameters.
* **Efficiency:** Kolmogorov-Arnold Networks (KAN) achieve a competitive 0.881 macro F1 with only 142k parameters, representing a 13-fold reduction in parameter count compared to ResNet1D. KAN is identified as the optimal choice for resource-constrained cardiac monitoring.
* **Negative Result:** S4 models underperform on the 187-timestep fixed-length signals, suggesting their theoretical advantages are specific to much longer sequences.

## 🛠️ Hardware Architecture
The live telemetry system bypasses standard development boards in favor of a compact custom build:
* **Microcontroller:** ESP32-C3 Supermini
* **Analog Front-End:** AD8232 ECG Sensor
* **Baud Rate:** 115200 

## 🧠 Software & Models
The repository includes implementations of seven architectures:
1. **ResNet1D** (Baseline)
2. **TCN** (Temporal Convolutional Network)
3. **TransformerECG**
4. **KAN** (Kolmogorov-Arnold Network)
5. **ConvKAN** (Hybrid CNN + KAN)
6. **S4ECG** (Simplified S4 State Space Model)
7. **BiGRU**

To handle the severe class imbalance (class F is outnumbered 113:1 by class N), training utilizes a `WeightedRandomSampler` and Focal Loss. 

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/ecg-ml-telemetry.git](https://github.com/YOUR_USERNAME/ecg-ml-telemetry.git)
   cd ecg-ml-telemetry
