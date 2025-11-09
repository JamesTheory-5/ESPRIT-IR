# ESPRIT-IR

```python
#!/usr/bin/env python3
"""
============================================================
ESPRIT-Based Impulse Response Modeling (Auto-Order, Stable)
============================================================

Estimate a stable digital filter H(z) = B(z)/A(z)
from an impulse response (WAV).

Features:
  - ESPRIT pole estimation
  - Automatic or manual model order
  - Stability-aware order selection
  - Pole stabilization (|λ| < 0.98)
  - Regularized numerator estimation (no SVD crashes)
  - Stable reconstruction
  - SOS export + JSON/CSV/WAV

Usage:
    # Auto order
    python esprit_ir_to_diff.py impulse.wav --plot

    # Fixed order
    python esprit_ir_to_diff.py impulse.wav --order 20 --plot
"""

import argparse
import json
import csv
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv, toeplitz
from scipy.signal import lfilter, freqz, tf2sos

# ============================================================
# 1. Load impulse response
# ============================================================
def load_impulse_response(path, normalize=True, window_length=None):
    x, fs = sf.read(path)

    if hasattr(x, "ndim") and x.ndim > 1:
        x = x.mean(axis=1)

    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("Empty or invalid audio file.")

    peak = np.max(np.abs(x))
    if normalize and peak > 0:
        x = x / peak

    n0 = int(np.argmax(np.abs(x)))
    if window_length is None:
        window_length = int(0.1 * fs)

    end = min(n0 + window_length, len(x))
    if end - n0 < 8:
        raise ValueError(
            "Impulse window too short after peak detection. "
            "Check input file or increase window_length."
        )

    h = x[n0:end]
    return h, fs

# ============================================================
# 2. Trajectory matrix
# ============================================================
def build_trajectory(h, L):
    N = len(h)
    if L < 2 or L > N:
        raise ValueError(f"Invalid L={L} for N={N}")
    K = N - L + 1
    if K < 1:
        raise ValueError("Not enough samples for trajectory matrix.")
    X = np.empty((L, K))
    for k in range(K):
        X[:, k] = h[k:k+L]
    return X

# ============================================================
# 3. ESPRIT poles
# ============================================================
def esprit_poles(h, M, L=None):
    N = len(h)
    if M < 1 or M >= N:
        raise ValueError(f"Order M={M} invalid for N={N}")
    if L is None:
        L = max(M + 1, N // 3)
    L = max(M + 1, min(L, N - 1))

    X = build_trajectory(h, L)
    U, s, Vh = svd(X, full_matrices=False)
    Us = U[:, :M]
    U1, U2 = Us[:-1, :], Us[1:, :]
    Phi = pinv(U1) @ U2
    eigvals, _ = np.linalg.eig(Phi)
    if not np.all(np.isfinite(eigvals)):
        raise ValueError("ESPRIT returned NaN/Inf eigenvalues.")
    return eigvals

# ============================================================
# 4. Stabilize poles
# ============================================================
def stabilize_poles(lambdas, radius=0.98):
    lambdas = np.asarray(lambdas, dtype=complex)
    if lambdas.size == 0:
        raise ValueError("No poles to stabilize.")

    # Clean non-finite
    if not np.all(np.isfinite(lambdas)):
        print("⚠️  Non-finite poles detected; replacing with 0.")
        lambdas = np.where(np.isfinite(lambdas), lambdas, 0.0)

    mags = np.abs(lambdas)
    mask = mags >= 1.0
    if np.any(mask):
        print(f"⚠️  {np.count_nonzero(mask)} unstable poles reflected inside unit circle.")
        lambdas[mask] = (radius / mags[mask]) * lambdas[mask]

    # Nudge anything exactly at radius slightly inward
    mags = np.abs(lambdas)
    mask = mags >= radius
    if np.any(mask):
        lambdas[mask] *= (1.0 - 1e-6)

    return np.real_if_close(lambdas)

# ============================================================
# 5. Poles → denominator
# ============================================================
def diff_eq_from_poles(lambdas):
    poly = np.poly(lambdas)  # [1, a1, ..., aM]
    a = np.real_if_close(poly[1:], tol=1e-6)
    if not np.all(np.isfinite(a)):
        raise ValueError("Denominator coefficients contain NaN/Inf.")
    return a

# ============================================================
# 6. Regularized numerator estimation
# ============================================================
def estimate_numerator(h, a, M_num=None, reg=1e-6):
    """
    Regularized LS:
        (G^T G + λI) b = G^T h
    Uses denominator impulse response built from stabilized poles.
    Only fails on true numerical breakdown (NaN/Inf).
    """
    h = np.asarray(h, dtype=float)
    N = len(h)
    if N < 4:
        raise ValueError("Impulse too short for numerator estimation.")
    if not np.all(np.isfinite(h)):
        raise ValueError("Impulse contains NaN/Inf.")

    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError("Denominator coefficients contain NaN/Inf.")

    M_den = len(a)
    if M_num is None:
        M_num = min(M_den + 1, max(2, N // 8))
    M_num = max(1, min(M_num, N - 1))

    # Denominator impulse response
    impulse = np.zeros(N)
    impulse[0] = 1.0
    a_full = np.r_[1.0, a]

    # Try once; if it explodes numerically, shrink slightly and retry
    for shrink in (1.0, 0.99):
        g = lfilter([1.0], np.r_[1.0, a * shrink], impulse)
        if np.all(np.isfinite(g)):
            a = a * shrink
            break
    else:
        raise ValueError("Denominator impulse response contains NaN/Inf even after damping.")

    # Toeplitz convolution matrix
    first_col = np.r_[g, np.zeros(M_num - 1)]
    first_row = np.r_[g[0], np.zeros(M_num - 1)]
    G = toeplitz(first_col, first_row)[:N, :M_num]

    GTG = G.T @ G
    GTy = G.T @ h
    A_reg = GTG + reg * np.eye(M_num)

    b = np.linalg.solve(A_reg, GTy)
    b = np.real_if_close(b)
    if abs(b[0]) > 1e-10:
        b = b / b[0]

    return b

# ============================================================
# 7. Reconstruction & metrics
# ============================================================
def reconstruct_tf(h, b, a):
    h = np.asarray(h, dtype=float)
    N = len(h)
    impulse = np.zeros(N)
    impulse[0] = 1.0
    a_full = np.r_[1.0, a]
    h_rec = lfilter(b, a_full, impulse)

    if np.any(np.isfinite(h_rec)):
        max_rec = np.max(np.abs(h_rec))
        max_ref = np.max(np.abs(h))
        if max_rec > 0 and max_ref > 0:
            h_rec = h_rec * (max_ref / max_rec)
    return np.real_if_close(h_rec)

def error_metrics(h, h_rec):
    h = np.asarray(h, dtype=float)
    h_rec = np.asarray(h_rec, dtype=float)
    if h.shape != h_rec.shape:
        raise ValueError("Length mismatch in metrics.")
    err = h - h_rec
    num = float(np.sum(h**2))
    den = float(np.sum(err**2))
    mse = den / len(h) if len(h) > 0 else float("inf")
    if den == 0.0:
        snr = float("inf")
    elif num == 0.0:
        snr = float("-inf")
    else:
        snr = 10.0 * np.log10(num / den)
    return {"MSE": float(mse), "SNR_dB": float(snr)}

# ============================================================
# 8. Auto order estimation
# ============================================================
def estimate_esprit_order(h, M_max=None, energy_thresh=0.999, gap_thresh=10.0):
    N = len(h)
    if N < 16:
        return max(2, N // 4)

    L = max(8, N // 3)
    L = min(L, N - 1)
    X = build_trajectory(h, L)
    _, s, _ = svd(X, full_matrices=False)

    energy = np.cumsum(s**2) / np.sum(s**2)
    M_e = int(np.searchsorted(energy, energy_thresh)) + 1

    ratios = s[:-1] / (s[1:] + 1e-18)
    k_gap = int(np.argmax(ratios)) + 1
    M_g = k_gap if ratios[k_gap - 1] >= gap_thresh else M_e

    M0 = min(M_e, M_g)

    if M_max is None:
        M_max = max(4, N // 4)

    M0 = max(2, min(M0, M_max, L - 1, N - 2))
    return M0

def choose_stable_order(h, M_init, M_min=2, M_max=None):
    N = len(h)
    if M_max is None:
        M_max = max(M_init, M_min)

    for M in range(min(M_init, M_max), M_min - 1, -1):
        try:
            lambdas = esprit_poles(h, M)
            lambdas = stabilize_poles(lambdas)
            a = diff_eq_from_poles(lambdas)

            # Quick sanity: short denominator IR must be finite and not ridiculous
            Ltest = min(N, 4 * M)
            impulse = np.zeros(Ltest)
            impulse[0] = 1.0
            g = lfilter([1.0], np.r_[1.0, a], impulse)

            if np.all(np.isfinite(g)) and np.max(np.abs(g)) < 1e6:
                print(f"Auto-selected stable order M = {M}")
                return M
            else:
                print(f"Order {M} rejected: unstable denominator response.")
        except Exception as e:
            print(f"Order {M} rejected: {e}")
            continue

    print("No stable high order found; falling back to M = 2")
    return 2

# ============================================================
# 9. SOS conversion
# ============================================================
def to_sos(b, a_full):
    b = np.asarray(b, dtype=float)
    a_full = np.asarray(a_full, dtype=float)
    if a_full[0] != 0:
        b /= a_full[0]
        a_full /= a_full[0]
    return tf2sos(b, a_full)

# ============================================================
# 10. Plots
# ============================================================
def plot_pz(lambdas):
    plt.figure()
    plt.scatter(np.real(lambdas), np.imag(lambdas), color="r", label="Poles")
    uc = plt.Circle((0, 0), 1, color="k", fill=False, linestyle="--")
    ax = plt.gca()
    ax.add_artist(uc)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_aspect("equal", "box")
    plt.title("Pole Locations")
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_freq_response(b, a_full, fs):
    w, H = freqz(b, a_full, worN=2048)
    f = w * fs / (2*np.pi)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(f, 20*np.log10(np.maximum(np.abs(H), 1e-12)))
    plt.title("Magnitude Response (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2)
    plt.plot(f, np.unwrap(np.angle(H)))
    plt.title("Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (rad)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_impulse_fit(h, h_rec):
    plt.figure()
    plt.plot(h, label="Measured")
    plt.plot(h_rec, "--", label="Reconstructed")
    plt.title("Impulse Response Fit")
    plt.xlabel("Sample")
    plt.legend()
    plt.grid(True, alpha=0.3)

# ============================================================
# 11. Export
# ============================================================
def export_all(b, a, lambdas, fs, h_rec, metrics, prefix="esprit_tf"):
    a_full = np.r_[1.0, a]
    sos = to_sos(b, a_full)

    data = {
        "sampling_rate": float(fs),
        "numerator": [float(x) for x in b],
        "denominator": [float(x) for x in a_full],
        "sos": [[float(v) for v in row] for row in sos],
        "poles_real": [float(np.real(p)) for p in lambdas],
        "poles_imag": [float(np.imag(p)) for p in lambdas],
        "metrics": metrics,
    }

    json_path = f"{prefix}.json"
    csv_path = f"{prefix}.csv"
    wav_path = f"{prefix}_reconstructed.wav"

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["b_index", "b_value"])
        for i, bi in enumerate(b):
            w.writerow([i, bi])
        w.writerow([])
        w.writerow(["a_index", "a_value"])
        for i, ai in enumerate(a, 1):
            w.writerow([i, ai])

    h_norm = np.asarray(h_rec, dtype=float)
    peak = np.max(np.abs(h_norm))
    if peak > 0:
        h_norm = h_norm / peak
    sf.write(wav_path, h_norm, int(fs))

    print(f"\n📁 Exported {json_path}, {csv_path}, {wav_path}")

# ============================================================
# 12. Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="Impulse response WAV")
    ap.add_argument("--order", type=int,
                    help="Model order M. If omitted, chosen automatically.")
    ap.add_argument("--maxorder", type=int, default=None,
                    help="Max order for auto mode (optional).")
    ap.add_argument("--plot", action="store_true", help="Show diagnostic plots")
    ap.add_argument("--export_prefix", default="esprit_tf",
                    help="Output file prefix")
    args = ap.parse_args()

    h, fs = load_impulse_response(args.wav)
    print(f"Loaded {args.wav}: {len(h)} samples @ {fs} Hz")

    # ----- Choose order -----
    if args.order is not None:
        M = args.order
        print(f"Using user-specified order M = {M}")
    else:
        M0 = estimate_esprit_order(h, M_max=args.maxorder)
        print(f"Initial auto order estimate M0 = {M0}")
        M = choose_stable_order(h, M0, M_min=2, M_max=args.maxorder)

    # ----- ESPRIT with final order -----
    lambdas = esprit_poles(h, M)
    lambdas = stabilize_poles(lambdas)
    print(f"Max |pole| after stabilization = {np.max(np.abs(lambdas)):.4f}")

    a = diff_eq_from_poles(lambdas)
    print("\nDenominator coefficients (a₁…a_M):")
    for i, ai in enumerate(a, 1):
        print(f"  a{i:02d} = {ai:.6g}")

    # ----- Numerator with fallback -----
    try:
        b = estimate_numerator(h, a)
    except ValueError as e:
        print(f"\n⚠️  Numerator estimation failed for M={M}: {e}")
        # automatic fallback to smaller orders
        success = False
        for M_try in range(M-1, 1, -1):
            print(f"Trying fallback order M = {M_try}")
            try:
                lambdas = esprit_poles(h, M_try)
                lambdas = stabilize_poles(lambdas)
                a = diff_eq_from_poles(lambdas)
                b = estimate_numerator(h, a)
                M = M_try
                success = True
                print(f"✅ Fallback succeeded with M = {M_try}")
                break
            except Exception as e2:
                print(f"  Rejected M={M_try}: {e2}")
                continue
        if not success:
            raise RuntimeError("Failed to estimate numerator for any reasonable order.")

    print("\nNumerator coefficients (b₀…b_N):")
    for i, bi in enumerate(b):
        print(f"  b{i:02d} = {bi:.6g}")

    # ----- Reconstruction & metrics -----
    h_rec = reconstruct_tf(h, b, a)
    metrics = error_metrics(h, h_rec)

    print("\nReconstruction metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if np.isfinite(v) else f"  {k}: {v}")

    # ----- Export -----
    export_all(b, a, lambdas, fs, h_rec, metrics, prefix=args.export_prefix)

    if args.plot:
        plot_pz(lambdas)
        plot_freq_response(b, np.r_[1.0, a], fs)
        plot_impulse_fit(h, h_rec)
        plt.show()

if __name__ == "__main__":
    main()

```
