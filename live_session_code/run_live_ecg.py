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

# Import your KAN architecture
from models import get_model

# --- Configuration ---
COM_PORT = 'COM3'  # Update to your ESP32's port
BAUD_RATE = 115200
CHECKPOINT_PATH = 'outputs/checkpoints/mitbih/kan_best.pt'
PLOT_WINDOW = 750  # 3 seconds of data at 250Hz

CLASSES = {
    0: ("Normal (N)", "#00FF00"),                # Green
    1: ("Supraventricular Ectopic (S)", "#FFD700"), # Gold
    2: ("Ventricular Ectopic (V)", "#FF4500"),      # Orange-Red
    3: ("Fusion (F)", "#FF00FF"),                   # Magenta
    4: ("Unknown/Paced (Q)", "#808080")             # Gray
}

# --- Shared Memory ---
# deque is thread-safe for basic appends and reads
raw_buffer = deque(np.zeros(PLOT_WINDOW), maxlen=PLOT_WINDOW)
ml_buffer = deque(maxlen=750)
app_state = {
    "text": "Waiting for heartbeat...", 
    "color": "#FFFFFF"
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
    """Formats 250Hz data into the 187-timestep MIT-BIH format and runs inference."""
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
    model, device = load_kan_model()
    last_prediction_time = time.time()
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to {COM_PORT}. Hardware stream active.\n")
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line == "FLATLINE":
                    app_state["text"] = "LEADS OFF - Check Electrodes"
                    app_state["color"] = "#FF0000"
                    ml_buffer.clear()
                    raw_buffer.append(2048) # Draw flatline
                    continue
                    
                if line.isdigit():
                    val = int(line)
                    raw_buffer.append(val)
                    ml_buffer.append(val)
                    
                    # Run inference max twice a second to prevent overlapping calculations
                    if len(ml_buffer) == 750 and (time.time() - last_prediction_time) > 0.5:
                        result = process_and_predict(list(ml_buffer), model, device)
                        
                        if result:
                            pred_idx, probs = result
                            confidence = probs[pred_idx] * 100
                            class_name, class_color = CLASSES[pred_idx]
                            
                            # Safely update the UI state dictionary
                            app_state["text"] = f"{class_name} | Conf: {confidence:.1f}%"
                            app_state["color"] = class_color
                            
                            print(f"[{time.strftime('%H:%M:%S')}] {app_state['text']}")
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
    # 1. Start the Background ML Thread
    worker = threading.Thread(target=background_worker, daemon=True)
    worker.start()

    # 2. Setup the PyQt Application
    app = QtWidgets.QApplication(sys.argv)
    
    # 3. Setup the Window and Plot
    win = pg.GraphicsLayoutWidget(show=True, title="Live ECG Inference Dashboard")
    win.resize(1000, 500)
    win.setBackground('k') # Black background

    plot = win.addPlot(title="AD8232 Live Feed & KAN Pipeline")
    plot.setYRange(0, 4095, padding=0)
    plot.setXRange(0, PLOT_WINDOW, padding=0)
    plot.hideAxis('bottom') # Hide X axis numbers for a cleaner look
    
    # 4. Add the ECG Line and Status Text
    # Use anti-aliasing (aa=True) for a smoother medical curve
    curve = plot.plot(pen=pg.mkPen(color='#00FF00', width=2), aa=True)
    
    status_text = pg.TextItem(text="Initializing...", color="#FFFFFF", anchor=(0, 0))
    status_text.setFont(pg.QtGui.QFont("Arial", 16, pg.QtGui.QFont.Bold))
    status_text.setPos(10, 4000) # Position near top left
    plot.addItem(status_text)

    # 5. The Render Loop (Runs at exactly 30 FPS)
    def update_ui():
        # Update line geometry
        curve.setData(list(raw_buffer))
        
        # Update text and color dynamically
        status_text.setText(app_state["text"])
        status_text.setColor(app_state["color"])

    timer = QtCore.QTimer()
    timer.timeout.connect(update_ui)
    timer.start(33) # 33ms interval = ~30 FPS

    # 6. Execute the Application (Blocks until window is closed)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()