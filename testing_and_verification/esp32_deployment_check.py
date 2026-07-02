"""
esp32_deployment_check.py
--------------------------
Run this locally in your project folder where all checkpoints exist.

Usage:
    python esp32_deployment_check.py

Requires your checkpoints at:
    outputs/checkpoints/mitbih/*_best.pt

No ESP32 board needed. Simulates everything on your laptop.
"""

import os, sys, time, json, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR = "outputs/checkpoints/mitbih"
OUT_DIR  = "outputs/esp32"
os.makedirs(OUT_DIR, exist_ok=True)

# ESP32-C3 SuperMini hardware limits
ESP32_FLASH_KB      = 4096   # 4MB flash
ESP32_SRAM_KB       = 400    # total SRAM
ESP32_USABLE_RAM_KB = 320    # after FreeRTOS + WiFi stack + code
ESP32_CPU_MHZ       = 160    # single-core RISC-V

# Slowdown factors vs your laptop CPU (empirical for RISC-V no-SIMD)
FLOAT32_SLOWDOWN = 100       # soft-float, no vector units
INT8_SLOWDOWN    = 50        # integer ops faster on RISC-V

MODEL_NAMES = ["resnet1d", "tcn", "transformer", "kan", "convkan", "s4", "bigru"]

# Known F1 scores from your training (fallback if checkpoint missing metrics)
KNOWN_F1 = {
    "resnet1d": 0.931, "tcn": 0.748, "transformer": 0.770,
    "kan": 0.881, "convkan": 0.854, "s4": 0.537, "bigru": 0.899,
}

# ── BiGRU (add here in case not yet in models.py) ─────────────────────────────
class BiGRU(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.gru  = nn.GRU(1, 128, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, num_classes))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.head(out.mean(dim=1))


def build_model(name, num_classes=5):
    from models import get_model
    if name == "bigru":
        return BiGRU(num_classes)
    return get_model(name, num_classes=num_classes)


# ── Latency benchmark ─────────────────────────────────────────────────────────
def benchmark(model, n_runs=200):
    model.eval()
    dummy = torch.randn(1, 1, 187)
    with torch.no_grad():
        for _ in range(10): model(dummy)          # warmup
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return np.median(times), np.percentile(times, 95)


# ── Size & RAM estimation ─────────────────────────────────────────────────────
def estimate_sizes(params):
    f32_kb   = params * 4 / 1024
    int8_kb  = params * 1 / 1024
    act_kb   = max(4.0, params * 4 / 1024 * 0.08)   # ~8% of f32 size
    ram_kb   = int8_kb + act_kb + 8.0               # +8KB TFLite runtime
    return f32_kb, int8_kb, ram_kb


# ── Verdict ───────────────────────────────────────────────────────────────────
def get_verdict(int8_kb, ram_kb, esp_int8_ms):
    flags = []
    if int8_kb > ESP32_FLASH_KB * 0.8:
        flags.append(f"Flash too large ({int8_kb:.0f}KB)")
    if ram_kb > ESP32_USABLE_RAM_KB:
        flags.append(f"RAM overflow ({ram_kb:.0f}KB > {ESP32_USABLE_RAM_KB}KB)")
    if esp_int8_ms > 100:
        flags.append(f"Too slow ({esp_int8_ms:.0f}ms > 100ms real-time limit)")

    if not flags:
        return "✅  GO", "Deployable + real-time capable"
    elif len(flags) == 1 and "Flash" not in flags[0] and "RAM" not in flags[0]:
        return "⚠️  CAUTION", flags[0]
    else:
        return "❌  NO-GO", " | ".join(flags)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  ESP32-C3 SuperMini — Deployment Feasibility Checker")
    print(f"  SRAM: {ESP32_SRAM_KB}KB usable | Flash: {ESP32_FLASH_KB}KB | "
          f"CPU: {ESP32_CPU_MHZ}MHz RISC-V (no SIMD)")
    print(f"  Slowdown model: {FLOAT32_SLOWDOWN}x (f32) / {INT8_SLOWDOWN}x (int8) vs laptop")
    print("="*72)

    results = []

    for name in MODEL_NAMES:
        ckpt_path = os.path.join(CKPT_DIR, f"{name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"\n[{name.upper()}] — checkpoint not found at {ckpt_path}, skipping")
            continue

        print(f"\n── {name.upper()} ──────────────────────────────────────────")

        # Load model
        model = build_model(name)
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

        # Counts
        params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f32_kb, int8_kb, ram_kb = estimate_sizes(params)

        # Latency
        pc_med, pc_p95  = benchmark(model)
        esp_f32_ms      = pc_med * FLOAT32_SLOWDOWN
        esp_int8_ms     = pc_med * INT8_SLOWDOWN / 4   # int8 ~4x faster ops

        # F1
        f1 = ckpt.get("metrics", {}).get("f1_macro", KNOWN_F1.get(name, "?"))

        # Verdict
        status, reason = get_verdict(int8_kb, ram_kb, esp_int8_ms)

        print(f"  Parameters       : {params:,}")
        print(f"  Flash needed     : {int8_kb:.1f} KB (int8)  |  {f32_kb:.1f} KB (f32)")
        print(f"  RAM needed       : {ram_kb:.1f} KB  (limit: {ESP32_USABLE_RAM_KB} KB)")
        print(f"  Laptop latency   : {pc_med:.3f} ms  (p95: {pc_p95:.3f} ms)")
        print(f"  ESP32 f32 est.   : ~{esp_f32_ms:.0f} ms")
        print(f"  ESP32 int8 est.  : ~{esp_int8_ms:.0f} ms")
        print(f"  Macro F1 (test)  : {f1}")
        print(f"  Verdict          : {status}  {reason}")

        results.append({
            "model":          name,
            "params":         params,
            "f32_kb":         round(f32_kb, 1),
            "int8_kb":        round(int8_kb, 1),
            "ram_kb":         round(ram_kb, 1),
            "pc_ms":          round(pc_med, 3),
            "esp32_f32_ms":   round(esp_f32_ms, 1),
            "esp32_int8_ms":  round(esp_int8_ms, 1),
            "f1_macro":       f1,
            "verdict":        status,
            "reason":         reason,
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n\n" + "="*90)
    print("  SUMMARY — ALL MODELS")
    print("="*90)
    print(f"{'Model':<14} {'Params':>9} {'int8(KB)':>9} {'RAM(KB)':>8} "
          f"{'ESP32 int8':>11} {'F1':>6}  {'Verdict'}")
    print("-"*90)
    for r in results:
        print(f"{r['model']:<14} {r['params']:>9,} {r['int8_kb']:>9.1f} "
              f"{r['ram_kb']:>8.1f} {r['esp32_int8_ms']:>9.1f}ms "
              f"{str(r['f1_macro']):>6}  {r['verdict']}")

    print(f"\n  ESP32-C3 limits:")
    print(f"    Flash (model only) : ~{ESP32_FLASH_KB * 0.8:.0f} KB safe limit")
    print(f"    RAM                : ~{ESP32_USABLE_RAM_KB} KB")
    print(f"    Real-time limit    : <100 ms/beat (60-100 BPM = beat every 600-1000ms)")

    # ── Deployment recommendation ──────────────────────────────────────────────
    print("\n" + "="*90)
    print("  DEPLOYMENT RECOMMENDATION")
    print("="*90)
    go_models = [r for r in results if "GO" in r["verdict"] and "NO" not in r["verdict"]]
    if go_models:
        best = max(go_models, key=lambda r: float(r["f1_macro"]) if isinstance(r["f1_macro"], float) else 0)
        print(f"\n  Best deployable model: {best['model'].upper()}")
        print(f"    F1 Macro   : {best['f1_macro']}")
        print(f"    Flash      : {best['int8_kb']:.1f} KB")
        print(f"    RAM        : {best['ram_kb']:.1f} KB")
        print(f"    Latency    : ~{best['esp32_int8_ms']:.0f} ms/beat")
        print(f"    Throughput : ~{1000/best['esp32_int8_ms']:.0f} beats/sec")
        print(f"\n  This means the {best['model'].upper()} model can classify")
        print(f"  {1000/best['esp32_int8_ms'] * 60:.0f} beats per minute in real-time")
        print(f"  on a $3 chip with no internet connection.")

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "esp32_feasibility.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_path}")

    # ── Next steps ─────────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("  NEXT STEPS (when AD8232 arrives)")
    print("="*90)
    print("""
  1. Install Arduino IDE + ESP32 board package
     https://dl.espressif.com/dl/package_esp32_index.json

  2. Install libraries:
     - EloquentTinyML (by Simone Salerno)
     - ESP32 AnalogRead

  3. Convert your best model checkpoint:
     pip install onnx onnxruntime tensorflow
     python convert_to_tflite.py --model kan

  4. Generate C header:
     xxd -i kan_int8.tflite > kan_model.h

  5. Flash the ecg_inference.ino sketch (provided separately)

  6. Wire AD8232:
     3.3V → 3.3V  |  GND → GND  |  OUTPUT → GPIO2
     LO+  → GPIO3 |  LO- → GPIO4
  """)


if __name__ == "__main__":
    main()
