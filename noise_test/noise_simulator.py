"""
noise_simulator.py
------------------
Hardware-accurate ECG noise simulation for two platforms:

  'ad8232_esp32' — AD8232 AFE + ESP32-C3 internal 12-bit SAR ADC (~10-bit effective)
                   Gaussian noise, baseline wander, PLI, motion artifacts,
                   Wi-Fi TX spikes, 10-bit quantisation.

  'ads1115'      — AD8232 AFE + ADS1115 external 16-bit delta-sigma ADC
                   Much lower thermal/quant noise, same electrode baseline wander,
                   better PLI rejection, smaller motion artifacts, no Wi-Fi spikes.

Source           | AD8232+ESP32  | ADS1115       | Notes
-----------------|---------------|---------------|------------------------
Gaussian noise   | std 0.01–0.02 | std 0.003–008 | 10-bit vs 16-bit eff.
Baseline wander  | A 0.03–0.10   | A 0.02–0.06   | same electrode iface
PLI @ 50 Hz      | A 0.005–0.015 | A 0.003–0.010 | AD8232 BPF attenuates
Motion artifacts | P=0.15        | P=0.12        | amplitude differs
Wi-Fi TX spikes  | P=0.07        | none          | only internal ADC
Quantisation     | 10-bit eff.   | 16-bit (skip) |

Signals are assumed normalised to [0, 1] (Kaggle MIT-BIH / PTB CSVs).
FS = 360 Hz  |  SEQ_LEN = 187 samples (~0.52 s per beat window).
"""

import numpy as np

PROFILES = ("clean", "ad8232_esp32", "ads1115")
FS       = 360
SEQ_LEN  = 187


def _t(L: int) -> np.ndarray:
    """Return (1, L) time axis in seconds for vectorised broadcasting."""
    return (np.arange(L, dtype=np.float64) / FS).reshape(1, L)


def apply_noise(
    X:       np.ndarray,
    profile: str = "ad8232_esp32",
    seed:    int | None = None,
) -> np.ndarray:
    """
    Apply hardware-accurate noise to a batch of ECG beats.

    Parameters
    ----------
    X       : (N, 187) float – normalised ECG beats in [0, 1]
    profile : 'clean' | 'ad8232_esp32' | 'ads1115'
    seed    : RNG seed.  Fixed seed → reproducible test set.
              None → different noise each call (training augmentation).

    Returns
    -------
    (N, 187) float32, clipped to [0, 1]
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choose from {PROFILES}")

    if profile == "clean":
        return X.astype(np.float32)

    rng  = np.random.default_rng(seed)
    X    = X.copy().astype(np.float64)
    N, L = X.shape
    t    = _t(L)                           # (1, L)

    # ── 1. Baseline wander: low-freq electrode–skin drift ────────────────────
    # Both platforms share the AD8232 AFE → same electrode interface
    bw_amp_hi = 0.10 if profile == "ad8232_esp32" else 0.06
    bw_amp_lo = 0.03 if profile == "ad8232_esp32" else 0.02
    bw_amp    = rng.uniform(bw_amp_lo, bw_amp_hi, (N, 1))
    bw_freq   = rng.uniform(0.50, 2.00, (N, 1))          # 0.5–2 Hz
    bw_phase  = rng.uniform(0.0,  2 * np.pi, (N, 1))
    X += bw_amp * np.sin(2 * np.pi * bw_freq * t + bw_phase)

    # ── 2. Motion artifacts: burst noise from lead movement ──────────────────
    p_motion = 0.15 if profile == "ad8232_esp32" else 0.12
    amp_lo   = 0.06 if profile == "ad8232_esp32" else 0.03
    amp_hi   = 0.20 if profile == "ad8232_esp32" else 0.10
    len_hi   = 45   if profile == "ad8232_esp32" else 35

    for i in np.where(rng.random(N) < p_motion)[0]:
        start = rng.integers(0, max(1, L // 2))
        blen  = rng.integers(10, len_hi)
        amp   = rng.uniform(amp_lo, amp_hi)
        end   = min(start + blen, L)
        X[i, start:end] += rng.standard_normal(end - start) * amp

    # ── Platform-specific noise ───────────────────────────────────────────────
    if profile == "ad8232_esp32":

        # 3a. Gaussian thermal + quantisation noise (~10-bit effective ADC)
        X += rng.standard_normal(X.shape) * rng.uniform(0.010, 0.020)

        # 3b. Power-line interference @ 50 Hz
        #     Partially suppressed by AD8232's built-in 40 Hz low-pass filter
        pli_amp   = rng.uniform(0.005, 0.015, (N, 1))
        pli_phase = rng.uniform(0.0, 2 * np.pi, (N, 1))
        X += pli_amp * np.sin(2 * np.pi * 50 * t + pli_phase)

        # 3c. Wi-Fi TX spikes: capacitive coupling when ESP32 Tx fires
        spike_idx = np.where(rng.random(N) < 0.07)[0]
        if len(spike_idx):
            sp = rng.integers(0, L, len(spike_idx))
            sa = (rng.integers(0, 2, len(spike_idx)) * 2 - 1) * \
                 rng.uniform(0.10, 0.30, len(spike_idx))
            X[spike_idx, sp] += sa

        # 3d. 10-bit effective quantisation (1/1024 step)
        q = 1.0 / 1024
        X = np.round(X / q) * q

    else:  # ads1115

        # 3a. Gaussian thermal noise only — 16-bit Δ-Σ quantisation is negligible
        X += rng.standard_normal(X.shape) * rng.uniform(0.003, 0.008)

        # 3b. PLI @ 50 Hz — ADS1115 has better CMRR but NO built-in HW BPF
        pli_amp   = rng.uniform(0.003, 0.010, (N, 1))
        pli_phase = rng.uniform(0.0, 2 * np.pi, (N, 1))
        X += pli_amp * np.sin(2 * np.pi * 50 * t + pli_phase)

        # No Wi-Fi spikes — external I2C ADC is isolated from ESP32 Tx
        # 16-bit quantisation < thermal floor, skip simulation

    # ── Final clip to valid ADC range ─────────────────────────────────────────
    np.clip(X, 0.0, 1.0, out=X)
    return X.astype(np.float32)


def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Mean SNR in dB between a clean and a noisy batch."""
    noise   = noisy.astype(np.float64) - clean.astype(np.float64)
    sig_pow = np.mean(clean.astype(np.float64) ** 2, axis=1)
    noi_pow = np.mean(noise ** 2, axis=1)
    noi_pow = np.where(noi_pow < 1e-12, 1e-12, noi_pow)
    return float(np.mean(10.0 * np.log10(sig_pow / noi_pow)))


# ── Quick visual self-test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Synthesise a plausible ECG beat (real data not needed for this test)
    t_beat = np.linspace(0, 1, SEQ_LEN)
    beat   = (0.6 * np.exp(-((t_beat - 0.5) ** 2) / 0.001)   # QRS spike
            + 0.15 * np.sin(2 * np.pi * 1.2 * t_beat)        # P/T wave
            + 0.05)
    beat   = (beat - beat.min()) / ((beat.max() - beat.min()) + 1e-9)
    beat   = beat.reshape(1, -1).astype(np.float32)

    colors  = {"clean": "#2563EB", "ad8232_esp32": "#EF4444", "ads1115": "#16A34A"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)

    for ax, prof in zip(axes, PROFILES):
        noisy = apply_noise(beat, profile=prof, seed=42)
        snr   = snr_db(beat, noisy)
        ax.plot(noisy[0], color=colors[prof], linewidth=0.9)
        ax.set_title(f"{prof}  (SNR ≈ {snr:.1f} dB)", fontsize=10)
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Sample index")
    plt.suptitle("ECG Hardware Noise Profiles", fontweight="bold", fontsize=13)
    plt.tight_layout()
    out = "noise_preview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
