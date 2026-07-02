import sys
import time
import threading
import serial
import numpy as np
import torch
from scipy.signal import find_peaks, resample
from collections import deque

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# Import your model architecture factory
from models import get_model

# --- Configuration ---
COM_PORT = 'COM3'  # Update to your ESP32's port
BAUD_RATE = 115200
CHECKPOINT_DIR = 'outputs/checkpoints/mitbih/'
PLOT_WINDOW = 750  # 3 seconds of data at 250Hz

# The exact names must match the keys in your models.py get_model registry
AVAILABLE_MODELS = [
    "kan", "resnet1d", "tcn", "transformer", "convkan", "s4", "bigru"
]

CLASSES = {
    0: ("Normal (N)", "#00FF00"),
    1: ("Supraventricular (S)", "#FFD700"),
    2: ("Ventricular (V)", "#FF4500"),
    3: ("Fusion (F)", "#FF00FF"),
    4: ("Unknown/Paced (Q)", "#808080")
}

# --- Shared Memory ---
raw_buffer = deque(np.zeros(PLOT_WINDOW), maxlen=PLOT_WINDOW)
ml_buffer = deque(maxlen=750)

# We use this dictionary to pass messages between the UI and ML threads
app_state = {
    "text": "Waiting for heartbeat...", 
    "color": "#FFFFFF",
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
    """Formats 250Hz data and runs inference."""
    peaks, _ = find_peaks(signal_data, distance=150, prominence=200)
    if len(peaks) == 0:
        return None
        
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
    
    tensor_input = torch.tensor(beat_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_input)
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        
    return pred_idx, probs

# ==========================================
# Background Worker Thread (Data & ML)
# ==========================================
def background_worker():
    model, device = load_pytorch_model(app_state["requested_model"])
    last_prediction_time = time.time()
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to {COM_PORT}. Hardware stream active.\n")
        
        while True:
            # 1. Hot-Swap Model Check
            if app_state["requested_model"] != app_state["current_model"]:
                model, device = load_pytorch_model(app_state["requested_model"])
            
            # 2. Process Serial Data
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line == "FLATLINE":
                    app_state["text"] = "LEADS OFF - Check Electrodes"
                    app_state["color"] = "#FF0000"
                    ml_buffer.clear()
                    raw_buffer.append(2048)
                    continue
                    
                if line.isdigit():
                    val = int(line)
                    raw_buffer.append(val)
                    ml_buffer.append(val)
                    
                    # 3. Run Inference
                    if len(ml_buffer) == 750 and (time.time() - last_prediction_time) > 0.5:
                        if model is not None:
                            result = process_and_predict(list(ml_buffer), model, device)
                            
                            if result:
                                pred_idx, probs = result
                                confidence = probs[pred_idx] * 100
                                class_name, class_color = CLASSES[pred_idx]
                                
                                app_state["text"] = f"[{app_state['current_model'].upper()}] {class_name} | Conf: {confidence:.1f}%"
                                app_state["color"] = class_color
                                
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
    
    # Standard PyQt5 Window Layout
    win = QtWidgets.QWidget()
    win.setWindowTitle("Multi-Model ECG Inference Dashboard")
    win.resize(1000, 600)
    win.setStyleSheet("background-color: black;")
    layout = QtWidgets.QVBoxLayout()
    win.setLayout(layout)
    
    # 1. Top Control Bar (Dropdown)
    control_layout = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel("Active ML Model:")
    label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
    
    model_dropdown = QtWidgets.QComboBox()
    model_dropdown.addItems(AVAILABLE_MODELS)
    model_dropdown.setStyleSheet("background-color: #333; color: white; padding: 5px; font-size: 14px;")
    
    # Event listener for dropdown changes
    def on_model_change():
        app_state["requested_model"] = model_dropdown.currentText()
        app_state["text"] = f"Switching to {app_state['requested_model']}..."
        app_state["color"] = "#FFFFFF"
        
    model_dropdown.currentTextChanged.connect(on_model_change)
    
    control_layout.addWidget(label)
    control_layout.addWidget(model_dropdown)
    control_layout.addStretch()
    layout.addLayout(control_layout)

    # 2. Main ECG Plot
    plot = pg.PlotWidget(title="AD8232 Live Feed")
    plot.setYRange(0, 4095, padding=0)
    plot.setXRange(0, PLOT_WINDOW, padding=0)
    plot.hideAxis('bottom')
    layout.addWidget(plot)
    
    curve = plot.plot(pen=pg.mkPen(color='#00FF00', width=2), aa=True)
    
    status_text = pg.TextItem(text="", color="#FFFFFF", anchor=(0, 0))
    status_text.setFont(pg.QtGui.QFont("Arial", 16, pg.QtGui.QFont.Bold))
    status_text.setPos(10, 4000)
    plot.addItem(status_text)
    
    load_text = pg.TextItem(text="", color="#FFFF00", anchor=(1, 0))
    load_text.setFont(pg.QtGui.QFont("Arial", 12))
    load_text.setPos(PLOT_WINDOW - 10, 4000)
    plot.addItem(load_text)

    # 3. The Render Loop
    def update_ui():
        curve.setData(list(raw_buffer))
        status_text.setText(app_state["text"])
        status_text.setColor(app_state["color"])
        load_text.setText(app_state["load_status"])

    timer = QtCore.QTimer()
    timer.timeout.connect(update_ui)
    timer.start(33) 

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()