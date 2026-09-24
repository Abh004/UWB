"""
x7_physics.py
=============
Single source of truth for Novelda X7 IR-UWB radar physical constants and the
DSP primitives shared by the gesture and presence pipelines.

Every range/velocity figure quoted elsewhere in this repository derives from the
constants defined here. Nothing should hardcode a range in metres.

Hardware reference (Novelda X7 / X7F202, RadarDirect baseband):
  * RF centre frequency      : 7.29 GHz  (low-band X7 configuration)
  * Baseband bin spacing     : ~6.5 mm one-way range per bin in air
  * Virtual MIMO array       : 2 TX x 2 RX -> 4 virtual channels
"""

import numpy as np

# ── RF / PROPAGATION CONSTANTS ────────────────────────────────────────────────
C_AIR = 299_792_458.0          # speed of light in vacuum (m/s)
F_CENTER_HZ = 7.29e9           # X7 RF centre frequency
LAMBDA_M = C_AIR / F_CENTER_HZ # ~41.1 mm

# One-way range increment per baseband bin, and the range of bin 0.
#
# MEASURED from the SDK on 2026-09-10 via probe_range.py (human_preset.json):
#     bin_length    0.0714 m
#     range_offset -1.38 m
#     112 bins  ->  -1.38 m .. 6.55 m
#
# Earlier revisions assumed 0.0065 m from a source-code comment and never read
# the values the radar supplies in every callback. That was wrong by a factor of
# 11 and made every range figure in this project far too short.
BIN_LENGTH_M = 0.0714
RANGE_OFFSET_M = -1.38

# Bin 0 sits at -1.38 m, i.e. BEHIND the antenna. Bins below this index map to
# negative range and cannot contain a target return -- whatever energy they hold
# is TX->RX direct coupling or leakage, not a reflection. Derived rather than
# hardcoded so it stays correct if the constants above are ever re-measured.
#
# Verified on 56 real captures: bins 0-19 carry 4.3% of the MTI energy at 1/31
# the amplitude of the populated region, and the gesture peak lands at +0.48 m.
import math as _math
FIRST_PHYSICAL_BIN = max(0, _math.ceil(-RANGE_OFFSET_M / BIN_LENGTH_M))

# Antenna baselines of the X7 evaluation array (metres).
# The horizontal phasor pairs TX0/TX1 (a transmit-aperture baseline) and the
# vertical phasor pairs RX0/RX1 (a receive-aperture baseline). These are
# physically DIFFERENT apertures and must not be conflated.
D_TX_H_M = LAMBDA_M / 2.0      # transmit baseline  (horizontal)
D_RX_V_M = LAMBDA_M / 2.0      # receive baseline   (vertical)

EPS = 1e-9


# ── RANGE ─────────────────────────────────────────────────────────────────────
def bin_to_range(bin_idx, bin_length_m=BIN_LENGTH_M, range_offset_m=RANGE_OFFSET_M):
    """One-way range (metres) of a baseband bin index."""
    return range_offset_m + np.asarray(bin_idx) * bin_length_m


def max_range(n_bins, bin_length_m=BIN_LENGTH_M, range_offset_m=RANGE_OFFSET_M):
    """Maximum observable one-way range for an n_bins baseband window."""
    return range_offset_m + (n_bins - 1) * bin_length_m


# ── DOPPLER ───────────────────────────────────────────────────────────────────
def max_unambiguous_velocity(fps, lambda_m=LAMBDA_M):
    """
    Radial velocity (m/s) at which the frame-to-frame phase difference reaches
    +/-pi and therefore aliases.

        f_D = arg(D) / (2*pi*dt)  and  f_D = 2*v_r / lambda
        => v_r = lambda * arg(D) / (4*pi*dt);  |arg| <= pi  =>  v_max = lambda*fps/4
    """
    return lambda_m * fps / 4.0


def phase_to_velocity(phase_rad, fps, lambda_m=LAMBDA_M):
    """Convert an inter-frame phase difference (radians) to radial velocity."""
    return lambda_m * np.asarray(phase_rad) * fps / (4.0 * np.pi)


# ── ANGLE OF ARRIVAL ──────────────────────────────────────────────────────────
def phase_to_aoa(phase_rad, baseline_m, lambda_m=LAMBDA_M):
    """
    Convert a one-way inter-element phase difference to angle of arrival.

        delta_phi = 2*pi * d * sin(theta) / lambda   (ONE-WAY aperture)

    Returns NaN outside the unambiguous field of view.
    """
    s = np.asarray(phase_rad) * lambda_m / (2.0 * np.pi * baseline_m)
    return np.arcsin(np.clip(s, -1.0, 1.0))


def max_unambiguous_aoa(baseline_m, lambda_m=LAMBDA_M):
    """Half-width of the grating-lobe-free field of view (radians)."""
    return np.arcsin(min(1.0, lambda_m / (2.0 * baseline_m)))


# ── SHARED DSP PRIMITIVES ─────────────────────────────────────────────────────
def mti_filter(w, axis=0):
    """
    Moving Target Indicator: remove the temporally-static clutter component by
    subtracting the coherent mean along the frame axis.
    """
    return w - np.mean(w, axis=axis, keepdims=True)


def unit_phasor(z1, z2):
    """
    Amplitude-invariant cross-correlation phasor  z1 * conj(z2) / |z1 * conj(z2)|.
    Carries only the phase difference between the two channels.
    """
    interf = z1 * np.conj(z2)
    return interf / (np.abs(interf) + EPS)


def mti_noise_scale(w, frac):
    """
    Return an additive-noise standard deviation expressed as a fraction of the
    MTI RESIDUAL (the moving-target signal the models actually consume), not of
    the raw signal.

    Scaling noise to the raw std is a physics error: the raw std is dominated by
    the static clutter that the MTI filter subsequently removes, so noise sized
    that way arrives many times larger than the surviving signal.
    """
    residual = w - np.mean(w, axis=0, keepdims=True)
    return float(np.std(residual)) * frac


def complex_gaussian_noise(shape, sigma, rng=None):
    """
    Circularly-symmetric complex Gaussian noise whose TOTAL complex standard
    deviation is `sigma`.

    np.random.randn() + 1j*np.random.randn() has per-component std 1, hence a
    complex std of sqrt(2). The 1/sqrt(2) below makes `sigma` mean what it says.
    """
    rng = rng or np.random
    return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
            * (sigma / np.sqrt(2.0))).astype(np.complex64)


def describe(n_bins, fps, bin_length_m=BIN_LENGTH_M):
    """One-line human-readable summary of a radar configuration's physical envelope."""
    return (f"{n_bins} bins x {bin_length_m*1000:.2f} mm = "
            f"{max_range(n_bins, bin_length_m):.2f} m max range | "
            f"{fps} FPS -> +/-{max_unambiguous_velocity(fps):.2f} m/s unambiguous")
