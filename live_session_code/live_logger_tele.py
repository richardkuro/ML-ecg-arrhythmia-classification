import sys
import time
import threading
import serial
import csv
import os
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
CHECKPOINT_PATH = 'outputs/checkpoints/mitbih/kan_best.pt'
PLOT_WINDOW = 750  # 3 seconds of data at 250Hz
LOG_FILE = f"ecg_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

CLASSES = {
    0: ("Normal (N)", "#00FF00"),
    1: ("Supraventricular Ectopic (S)", "#FFD700"),
    2: ("Ventricular Ectopic (V)", "#FF4500"),
    3: ("Fusion (F)", "#FF00FF"),
    4: ("Unknown/Paced (Q)", "#808080")
}

# --- Shared Memory ---
raw_buffer = deque(np.zeros(PLOT_WINDOW), maxlen=PLOT_WINDOW)
ml_buffer = deque(maxlen=750)
app_state = {
    "text": "Waiting for heartbeat...", 
    "color": "#FFFFFF",
    "bpm": "--"
}

def load_kan_model():
    print("Loading 142k-parameter KAN Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("kan", num_classes=5)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, device

def process_and_predict(signal_data, model, device):
    """Calculates BPM, formats 250Hz data, and runs inference."""
    peaks, _ = find_peaks(signal_data, distance=150, prominence=200)
    
    # 1. Software Telemetry: Calculate BPM if we have at least 2 peaks
    bpm = 0
    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) # Number of samples between peaks
        avg_rr_samples = np.mean(rr_intervals)
        # 60 seconds / (average samples between beats / 250 samples per second)
        bpm = int(60.0 / (avg_rr_samples / 250.0))
        
    if len(peaks) == 0:
        return None
        
    # 2. MIT-BIH Formatting
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
    
    # 3. KAN Inference
    tensor_input = torch.tensor(beat_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_input)
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        
    return pred_idx, probs, bpm

# ==========================================
# Background Worker Thread (Data, ML & Logging)
# ==========================================
def background_worker():
    model, device = load_kan_model()
    last_prediction_time = time.time()
    
    # State tracking for the CSV logger
    current_class = "None"
    current_conf = 0.0
    current_bpm = 0
    log_batch = []
    
    print(f"Creating session log: {LOG_FILE}")
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to {COM_PORT}. Hardware stream active.\n")
        
        # Open CSV in append mode
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Raw_ADC", "BPM", "ML_Class", "ML_Confidence"])
            
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line == "FLATLINE":
                        app_state["text"] = "LEADS OFF - Check Electrodes"
                        app_state["color"] = "#FF0000"
                        app_state["bpm"] = "--"
                        ml_buffer.clear()
                        raw_buffer.append(2048)
                        continue
                        
                    if line.isdigit():
                        val = int(line)
                        raw_buffer.append(val)
                        ml_buffer.append(val)
                        
                        # Add to our hidden log batch
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        log_batch.append([timestamp, val, current_bpm, current_class, f"{current_conf:.1f}"])
                        
                        # Flush batch to disk every 250 samples (1 second) to prevent lag
                        if len(log_batch) >= 250:
                            writer.writerows(log_batch)
                            f.flush()
                            log_batch.clear()
                        
                        # Run inference max twice a second
                        if len(ml_buffer) == 750 and (time.time() - last_prediction_time) > 0.5:
                            result = process_and_predict(list(ml_buffer), model, device)
                            
                            if result:
                                pred_idx, probs, calc_bpm = result
                                current_conf = probs[pred_idx] * 100
                                class_name, class_color = CLASSES[pred_idx]
                                
                                # Only update BPM if the algorithm found a valid rhythm
                                if calc_bpm > 0:
                                    current_bpm = calc_bpm
                                    app_state["bpm"] = str(current_bpm)
                                    
                                current_class = class_name
                                
                                app_state["text"] = f"{class_name} | Conf: {current_conf:.1f}%"
                                app_state["color"] = class_color
                                
                                last_prediction_time = time.time()
                                
    except serial.SerialException:
        print(f"\n[Error] Could not open {COM_PORT}. Check connection.")
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
    
    win = pg.GraphicsLayoutWidget(show=True, title="Live ECG & KAN Telemetry")
    win.resize(1000, 500)
    win.setBackground('k') 

    plot = win.addPlot(title="AD8232 Feed | Telemetry")
    plot.setYRange(0, 4095, padding=0)
    plot.setXRange(0, PLOT_WINDOW, padding=0)
    plot.hideAxis('bottom') 
    
    curve = plot.plot(pen=pg.mkPen(color='#00FF00', width=2), aa=True)
    
    # ML Classification Text
    status_text = pg.TextItem(text="Initializing...", color="#FFFFFF", anchor=(0, 0))
    status_text.setFont(pg.QtGui.QFont("Arial", 16, pg.QtGui.QFont.Bold))
    status_text.setPos(10, 4000)
    plot.addItem(status_text)
    
    # BPM Telemetry Text
    bpm_text = pg.TextItem(text="BPM: --", color="#00FFFF", anchor=(1, 0))
    bpm_text.setFont(pg.QtGui.QFont("Arial", 16, pg.QtGui.QFont.Bold))
    bpm_text.setPos(PLOT_WINDOW - 10, 4000)
    plot.addItem(bpm_text)

    def update_ui():
        curve.setData(list(raw_buffer))
        status_text.setText(app_state["text"])
        status_text.setColor(app_state["color"])
        bpm_text.setText(f"BPM: {app_state['bpm']}")

    timer = QtCore.QTimer()
    timer.timeout.connect(update_ui)
    timer.start(33) 

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()