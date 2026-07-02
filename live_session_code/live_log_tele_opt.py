import sys
import time
import threading
import serial
import csv
from datetime import datetime
import numpy as np
import torch
from scipy.signal import find_peaks, resample
from collections import deque

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from models import get_model

# --- Configuration ---
COM_PORT = 'COM3'  # Update to your ESP32's port
BAUD_RATE = 115200
CHECKPOINT_DIR = 'outputs/checkpoints/mitbih/'
PLOT_WINDOW = 750  # 3 seconds of data at 250Hz
LOG_FILE = f"ecg_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

AVAILABLE_MODELS = [
    "kan", "resnet1d", "tcn", "transformer", "convkan", "s4", "bigru"
]

CLASSES = {
    0: ("Normal (N)", "#00FF00"),
    1: ("Supraventricular Ectopic (S)", "#FFD700"),
    2: ("Ventricular Ectopic (V)", "#FF4500"),
    3: ("Fusion (F)", "#FF00FF"),
    4: ("Unknown/Paced (Q)", "#808080")
}

# --- Shared Memory & Telemetry Buffers ---
raw_buffer = deque(np.zeros(PLOT_WINDOW), maxlen=PLOT_WINDOW)
ml_buffer = deque(maxlen=750)

# Rolling 20-beat windows for Telemetry
rr_interval_history = deque(maxlen=20) 
peak_amp_history = deque(maxlen=20)    

# Application State Dictionary
app_state = {
    "text": "Waiting for heartbeat...", 
    "color": "#FFFFFF",
    "bpm": "--",
    "hrv": "--",
    "resp": "--",
    "requested_model": "kan",
    "current_model": "kan",
    "load_status": ""
}

def load_pytorch_model(model_name):
    """Dynamically loads the requested model and its weights."""
    app_state["load_status"] = f"Loading {model_name}..."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes=5)
    
    checkpoint_path = f"{CHECKPOINT_DIR}{model_name}_best.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        app_state["load_status"] = ""
        app_state["current_model"] = model_name
        return model, device
    except FileNotFoundError:
        app_state["load_status"] = f"Error: Weights missing for {model_name}"
        return None, None

def process_and_predict(signal_data, model, device):
    """Calculates telemetry, formats 250Hz data, and runs inference."""
    peaks, _ = find_peaks(signal_data, distance=150, prominence=200)
    
    bpm, hrv_rmssd, resp_rate = 0, 0, 0
    
    # 1. Advanced Software Telemetry Calculations
    if len(peaks) >= 2:
        latest_rr_ms = (peaks[-1] - peaks[-2]) * 4.0 
        rr_interval_history.append(latest_rr_ms)
        peak_amp_history.append(signal_data[peaks[-1]])
        
        if len(rr_interval_history) > 0:
            bpm = int(60000.0 / np.mean(rr_interval_history))
            
        if len(rr_interval_history) > 2:
            successive_diffs = np.diff(np.array(rr_interval_history))
            hrv_rmssd = int(np.sqrt(np.mean(successive_diffs**2)))
            
        if len(peak_amp_history) == 20:
            breath_peaks, _ = find_peaks(np.array(peak_amp_history), prominence=10)
            if len(breath_peaks) >= 2:
                beats_per_breath = np.mean(np.diff(breath_peaks))
                if beats_per_breath > 0 and bpm > 0:
                    resp_rate = int(bpm / beats_per_breath)

    if len(peaks) == 0:
        return None
        
    # 2. MIT-BIH Formatting for Inference
    r_peak = peaks[-1]
    left, right = r_peak - 100, r_peak + 150
    
    if left < 0 or right >= len(signal_data):
        return None 
        
    segment = np.array(signal_data[left:right])
    beat_187 = resample(segment, 187)
    
    min_val, max_val = np.min(beat_187), np.max(beat_187)
    if max_val == min_val:
        return None
    beat_norm = (beat_187 - min_val) / (max_val - min_val)
    
    # 3. Model Inference
    tensor_input = torch.tensor(beat_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_input)
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        
    return pred_idx, probs, bpm, hrv_rmssd, resp_rate

# ==========================================
# Background Worker Thread (Data, ML & Logging)
# ==========================================
def background_worker():
    model, device = load_pytorch_model(app_state["requested_model"])
    last_prediction_time = time.time()
    
    current_class, current_conf = "None", 0.0
    current_bpm, current_hrv, current_resp = 0, 0, 0
    log_batch = []
    
    print(f"Creating session log: {LOG_FILE}")
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to {COM_PORT}. Hardware stream active.\n")
        
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # Log headers now include the Active ML Model
            writer.writerow(["Timestamp", "Raw_ADC", "BPM", "HRV_RMSSD", "Resp_Rate", "Active_Model", "ML_Class", "ML_Confidence"])
            
            while True:
                # Hot-Swap Model Check
                if app_state["requested_model"] != app_state["current_model"]:
                    model, device = load_pytorch_model(app_state["requested_model"])
                
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line == "FLATLINE":
                        app_state.update({"text": "LEADS OFF", "color": "#FF0000", "bpm": "--", "hrv": "--", "resp": "--"})
                        ml_buffer.clear()
                        raw_buffer.append(2048)
                        continue
                        
                    if line.isdigit():
                        val = int(line)
                        raw_buffer.append(val)
                        ml_buffer.append(val)
                        
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        log_batch.append([timestamp, val, current_bpm, current_hrv, current_resp, app_state["current_model"], current_class, f"{current_conf:.1f}"])
                        
                        if len(log_batch) >= 250:
                            writer.writerows(log_batch)
                            f.flush()
                            log_batch.clear()
                        
                        if len(ml_buffer) == 750 and (time.time() - last_prediction_time) > 0.5:
                            if model is not None:
                                result = process_and_predict(list(ml_buffer), model, device)
                                
                                if result:
                                    pred_idx, probs, bpm, hrv, resp = result
                                    current_conf = probs[pred_idx] * 100
                                    current_class, class_color = CLASSES[pred_idx]
                                    
                                    if bpm > 0: current_bpm = bpm
                                    if hrv > 0: current_hrv = hrv
                                    if resp > 0: current_resp = resp
                                    
                                    app_state["text"] = f"[{app_state['current_model'].upper()}] {current_class} | Conf: {current_conf:.1f}%"
                                    app_state["color"] = class_color
                                    app_state["bpm"] = str(current_bpm)
                                    app_state["hrv"] = str(current_hrv)
                                    app_state["resp"] = str(current_resp) if current_resp > 0 else "Calc..."
                                    
                                    last_prediction_time = time.time()
                                
    except serial.SerialException:
        app_state["text"] = "SERIAL PORT ERROR"
        app_state["color"] = "#FF0000"
    except Exception as e:
        print(f"\n[Fatal Worker Error] {e}")

# ==========================================
# Main Thread (GPU-Accelerated UI)
# ==========================================
def main():
    worker = threading.Thread(target=background_worker, daemon=True)
    worker.start()

    app = QtWidgets.QApplication(sys.argv)
    
    win = QtWidgets.QWidget()
    win.setWindowTitle("Multi-Model Clinical ECG Telemetry")
    win.resize(1100, 600)
    win.setStyleSheet("background-color: black;")
    layout = QtWidgets.QVBoxLayout()
    win.setLayout(layout)
    
    # 1. Control Bar
    control_layout = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel("Active Machine Learning Model:")
    label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
    
    model_dropdown = QtWidgets.QComboBox()
    model_dropdown.addItems(AVAILABLE_MODELS)
    model_dropdown.setStyleSheet("background-color: #333; color: white; padding: 5px; font-size: 14px;")
    
    def on_model_change():
        app_state["requested_model"] = model_dropdown.currentText()
        app_state["text"] = f"Switching to {app_state['requested_model']}..."
        app_state["color"] = "#FFFFFF"
        
    model_dropdown.currentTextChanged.connect(on_model_change)
    
    control_layout.addWidget(label)
    control_layout.addWidget(model_dropdown)
    control_layout.addStretch()
    layout.addLayout(control_layout)

    # 2. Main Plot
    plot = pg.PlotWidget(title="AD8232 Live Feed | Advanced Telemetry")
    plot.setYRange(0, 4095, padding=0)
    plot.setXRange(0, PLOT_WINDOW, padding=0)
    plot.hideAxis('bottom')
    layout.addWidget(plot)
    
    curve = plot.plot(pen=pg.mkPen(color='#00FF00', width=2), aa=True)
    
    # UI Text Elements
    status_text = pg.TextItem(text="Initializing...", color="#FFFFFF", anchor=(0, 0))
    status_text.setFont(pg.QtGui.QFont("Arial", 16, pg.QtGui.QFont.Bold))
    status_text.setPos(10, 4000)
    plot.addItem(status_text)
    
    load_text = pg.TextItem(text="", color="#FFFF00", anchor=(0, 0))
    load_text.setFont(pg.QtGui.QFont("Arial", 12))
    load_text.setPos(10, 3600)
    plot.addItem(load_text)
    
    bpm_text = pg.TextItem(text="BPM: --", color="#00FFFF", anchor=(1, 0))
    bpm_text.setFont(pg.QtGui.QFont("Arial", 16, pg.QtGui.QFont.Bold))
    bpm_text.setPos(PLOT_WINDOW - 10, 4000)
    plot.addItem(bpm_text)

    hrv_text = pg.TextItem(text="HRV (RMSSD): -- ms", color="#FFA500", anchor=(1, 0))
    hrv_text.setFont(pg.QtGui.QFont("Arial", 14))
    hrv_text.setPos(PLOT_WINDOW - 10, 3600)
    plot.addItem(hrv_text)

    resp_text = pg.TextItem(text="Resp Rate: -- br/min", color="#FF69B4", anchor=(1, 0))
    resp_text.setFont(pg.QtGui.QFont("Arial", 14))
    resp_text.setPos(PLOT_WINDOW - 10, 3200)
    plot.addItem(resp_text)

    # 3. Render Loop
    def update_ui():
        curve.setData(list(raw_buffer))
        status_text.setText(app_state["text"])
        status_text.setColor(app_state["color"])
        load_text.setText(app_state["load_status"])
        bpm_text.setText(f"BPM: {app_state['bpm']}")
        hrv_text.setText(f"HRV (RMSSD): {app_state['hrv']} ms")
        resp_text.setText(f"Resp Rate: {app_state['resp']} br/min")

    timer = QtCore.QTimer()
    timer.timeout.connect(update_ui)
    timer.start(33) 

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()