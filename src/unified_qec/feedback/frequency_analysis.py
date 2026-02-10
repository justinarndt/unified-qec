"""
Frequency Domain Analysis for Feedback Controller

Provides Bode plot generation and stability margin analysis
for the syndrome feedback controller.

A controller must have:
- Phase margin > 45° (safety against resonance)
- Gain margin > 6 dB (robustness to gain variations)

Migrated from: stim-cirq-qec/src/adaptive_qec/feedback/frequency_analysis.py
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class StabilityMargins:
    """Controller stability margins.

    Attributes
    ----------
    gain_margin_db : float
        Gain margin in decibels.
    phase_margin_deg : float
        Phase margin in degrees.
    crossover_freq_hz : float
        Gain crossover frequency (|L| = 1).
    gain_margin_freq_hz : float
        Phase crossover frequency (∠L = -180°).
    stable : bool
        True if both margins are positive.
    """
    gain_margin_db: float
    phase_margin_deg: float
    crossover_freq_hz: float
    gain_margin_freq_hz: float
    stable: bool


class FrequencyAnalyzer:
    """
    Frequency domain analysis for feedback controllers.

    Analyzes the open-loop transfer function:
        L(s) = C(s) · G(s)

    where C(s) is the controller and G(s) is the plant.

    For our integral controller:
        C(s) = Ki / s

    With feedback latency τ:
        C(s) = Ki · exp(-τs) / s

    Parameters
    ----------
    Ki : float
        Integral gain.
    latency_s : float
        Feedback latency in seconds.
    plant_gain : float
        Plant DC gain (syndrome sensitivity).
    plant_pole_hz : float
        Dominant plant pole frequency.
    """

    def __init__(
        self,
        Ki: float = 0.05,
        latency_s: float = 500e-9,
        plant_gain: float = 1.0,
        plant_pole_hz: float = 1000.0
    ):
        self.Ki = Ki
        self.latency_s = latency_s
        self.plant_gain = plant_gain
        self.plant_pole_hz = plant_pole_hz
        self.plant_pole_rad = 2 * np.pi * plant_pole_hz

    def open_loop_tf(self, freq_hz: np.ndarray) -> np.ndarray:
        """
        Compute open-loop transfer function L(jω).

        L(s) = Ki · exp(-τs) / s · G(s)
        where G(s) = K / (1 + s/p) is first-order plant.

        Parameters
        ----------
        freq_hz : np.ndarray
            Frequency vector in Hz.

        Returns
        -------
        np.ndarray
            Complex transfer function values.
        """
        omega = 2 * np.pi * freq_hz
        s = 1j * omega

        controller = self.Ki * np.exp(-self.latency_s * s) / s
        plant = self.plant_gain / (1 + s / self.plant_pole_rad)

        return controller * plant

    def closed_loop_tf(self, freq_hz: np.ndarray) -> np.ndarray:
        """
        Compute closed-loop transfer function T(jω) = L/(1+L).
        """
        L = self.open_loop_tf(freq_hz)
        return L / (1 + L)

    def compute_margins(
        self,
        freq_range: Tuple[float, float] = (0.1, 1e6)
    ) -> StabilityMargins:
        """
        Compute gain and phase margins.

        Parameters
        ----------
        freq_range : tuple
            Frequency range (min, max) in Hz.

        Returns
        -------
        StabilityMargins
            Stability margin results.
        """
        freq = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 10000)
        L = self.open_loop_tf(freq)

        magnitude = np.abs(L)
        phase = np.angle(L, deg=True)
        phase = np.unwrap(phase * np.pi / 180) * 180 / np.pi

        # Find gain crossover (|L| = 1, i.e., 0 dB)
        mag_db = 20 * np.log10(magnitude + 1e-10)
        crossover_idx = np.argmin(np.abs(mag_db))
        if 0 < crossover_idx < len(freq) - 1:
            crossover_freq = freq[crossover_idx]
            phase_at_crossover = phase[crossover_idx]
            phase_margin = 180 + phase_at_crossover
        else:
            crossover_freq = 0
            phase_margin = 180

        # Find phase crossover (phase = -180°)
        phase_cross_idx = np.argmin(np.abs(phase + 180))
        if 0 < phase_cross_idx < len(freq) - 1:
            gain_margin_freq = freq[phase_cross_idx]
            gain_at_phase_cross = magnitude[phase_cross_idx]
            gain_margin = 1 / gain_at_phase_cross
            gain_margin_db = 20 * np.log10(gain_margin + 1e-10)
        else:
            gain_margin_freq = 0
            gain_margin_db = float('inf')

        stable = (phase_margin > 0) and (gain_margin_db > 0)

        return StabilityMargins(
            gain_margin_db=float(gain_margin_db),
            phase_margin_deg=float(phase_margin),
            crossover_freq_hz=float(crossover_freq),
            gain_margin_freq_hz=float(gain_margin_freq),
            stable=stable
        )

    def generate_bode_plot(
        self,
        freq_range: Tuple[float, float] = (0.1, 1e6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate Bode plot with stability margins annotated.

        Parameters
        ----------
        freq_range : tuple
            Frequency range (min, max) in Hz.
        save_path : str, optional
            Path to save figure.

        Returns
        -------
        plt.Figure
            Matplotlib figure.
        """
        freq = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 1000)
        L = self.open_loop_tf(freq)

        magnitude_db = 20 * np.log10(np.abs(L) + 1e-10)
        phase_deg = np.angle(L, deg=True)
        phase_deg = np.unwrap(phase_deg * np.pi / 180) * 180 / np.pi

        margins = self.compute_margins(freq_range)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Magnitude plot
        ax1.semilogx(freq, magnitude_db, 'b-', linewidth=2, label='Open-loop |L(jω)|')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        if margins.gain_margin_freq_hz > 0:
            idx = np.argmin(np.abs(freq - margins.gain_margin_freq_hz))
            ax1.plot(margins.gain_margin_freq_hz, magnitude_db[idx], 'ro', markersize=10)
            ax1.annotate(f'Gain Margin\n{margins.gain_margin_db:.1f} dB',
                        xy=(margins.gain_margin_freq_hz, magnitude_db[idx]),
                        xytext=(margins.gain_margin_freq_hz * 0.1, magnitude_db[idx] + 10),
                        fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

        ax1.set_ylabel('Magnitude (dB)', fontsize=12)
        ax1.set_title(f'Bode Plot: Syndrome Feedback Controller\n'
                     f'Ki = {self.Ki}, Latency = {self.latency_s*1e9:.0f} ns',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_ylim([-60, 40])

        # Phase plot
        ax2.semilogx(freq, phase_deg, 'b-', linewidth=2, label='Phase ∠L(jω)')
        ax2.axhline(y=-180, color='r', linestyle='--', alpha=0.7, label='-180°')
        ax2.axhline(y=-135, color='orange', linestyle=':', alpha=0.7, label='-135° (45° margin)')

        if margins.crossover_freq_hz > 0:
            idx = np.argmin(np.abs(freq - margins.crossover_freq_hz))
            ax2.plot(margins.crossover_freq_hz, phase_deg[idx], 'go', markersize=10)
            ax2.annotate(f'Phase Margin\n{margins.phase_margin_deg:.1f}°',
                        xy=(margins.crossover_freq_hz, phase_deg[idx]),
                        xytext=(margins.crossover_freq_hz * 10, phase_deg[idx] + 20),
                        fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))

        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Phase (degrees)', fontsize=12)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_ylim([-270, -90])

        # Stability summary
        status = "✓ STABLE" if margins.stable else "✗ UNSTABLE"
        color = 'green' if margins.stable else 'red'
        pm_status = "✓" if margins.phase_margin_deg > 45 else "✗"
        gm_status = "✓" if margins.gain_margin_db > 6 else "✗"

        summary = (f'{status}\n'
                  f'{pm_status} Phase Margin: {margins.phase_margin_deg:.1f}° (req: >45°)\n'
                  f'{gm_status} Gain Margin: {margins.gain_margin_db:.1f} dB (req: >6 dB)')

        ax1.text(0.02, 0.02, summary, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                         edgecolor=color, linewidth=2))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def analyze_controller_stability(
    Ki_values: list = [0.01, 0.02, 0.05, 0.10],
    latency_ns: float = 500
) -> Dict:
    """
    Analyze stability across different Ki values.

    Parameters
    ----------
    Ki_values : list
        Integral gain values to test.
    latency_ns : float
        Feedback latency in nanoseconds.

    Returns
    -------
    dict
        Stability analysis results per Ki value.
    """
    results = {}

    for Ki in Ki_values:
        analyzer = FrequencyAnalyzer(Ki=Ki, latency_s=latency_ns * 1e-9)
        margins = analyzer.compute_margins()
        results[Ki] = {
            'phase_margin': margins.phase_margin_deg,
            'gain_margin': margins.gain_margin_db,
            'stable': margins.stable,
            'safe': margins.phase_margin_deg > 45 and margins.gain_margin_db > 6
        }

    return results
