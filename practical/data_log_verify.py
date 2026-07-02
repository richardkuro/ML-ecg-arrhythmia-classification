import pandas as pd

# Replace with your actual log file name
LOG_FILE = "ecg_session_20260608_110500.csv"

def analyze_session(file_path):
    print(f"--- ECG Session Analysis: {file_path} ---")
    df = pd.read_csv(file_path)
    
    # Filter out the 'None' rows where the model was buffering
    ml_data = df[df['ML_Class'] != 'None'].copy()
    ml_data['ML_Confidence'] = pd.to_numeric(ml_data['ML_Confidence'])
    
    # 1. Heart Rate Statistics
    valid_bpm = ml_data[ml_data['BPM'] > 0]['BPM']
    print(f"\nHeart Rate (BPM):")
    print(f"  Average: {valid_bpm.mean():.1f}")
    print(f"  Min: {valid_bpm.min()} | Max: {valid_bpm.max()}")
    
    # 2. Arrhythmia Breakdown
    print(f"\nModel Classifications:")
    class_counts = ml_data['ML_Class'].value_counts()
    for cls, count in class_counts.items():
        avg_conf = ml_data[ml_data['ML_Class'] == cls]['ML_Confidence'].mean()
        print(f"  {cls}: {count} beats (Avg Confidence: {avg_conf:.1f}%)")

    # 3. High-Confidence Anomaly Filter
    abnormal = ml_data[(ml_data['ML_Class'] != 'Normal (N)') & (ml_data['ML_Confidence'] > 85.0)]
    print(f"\nHigh-Confidence Arrhythmias Detected (>85%): {len(abnormal)}")
    if not abnormal.empty:
        for _, row in abnormal.head(5).iterrows():
            print(f"  - [{row['Timestamp']}] {row['ML_Class']} ({row['ML_Confidence']}%) at {row['BPM']} BPM")

if __name__ == "__main__":
    analyze_session(LOG_FILE)