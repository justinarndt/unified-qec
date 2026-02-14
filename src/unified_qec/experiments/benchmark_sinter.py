"""
Sinter Benchmark Harness

Compares all decoder backends (BP+OSD, Union-Find, PyMatching, neural)
on identical noise models via sinter.collect(). Generates threshold
plots and CSV data for Section 7 compliance validation.

Usage:
    python -m unified_qec.experiments.benchmark_sinter
    python -m unified_qec.experiments.benchmark_sinter --distances 3 5 --max-shots 10000
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import stim

try:
    import sinter
    SINTER_AVAILABLE = True
except ImportError:
    sinter = None
    SINTER_AVAILABLE = False

from unified_qec.decoding.sinter_api import UnifiedQECDecoder


def build_tasks(
    distances: List[int],
    error_rates: Optional[np.ndarray] = None,
    rounds_factor: int = 3,
) -> list:
    """Build a list of sinter.Task objects for threshold estimation.

    Parameters
    ----------
    distances : list of int
        Code distances to sweep.
    error_rates : ndarray, optional
        Physical error rates. Default: logspace(1e-3, 0.1, 10).
    rounds_factor : int
        Rounds = rounds_factor * distance.

    Returns
    -------
    list of sinter.Task
        Tasks pairing circuits with decoder metadata.
    """
    if error_rates is None:
        error_rates = np.logspace(-3, -1, 10)

    tasks = []
    for d in distances:
        for p in error_rates:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=d,
                rounds=rounds_factor * d,
                after_clifford_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p,
            )
            tasks.append(sinter.Task(
                circuit=circuit,
                json_metadata={
                    "d": d,
                    "p": float(p),
                    "rounds": rounds_factor * d,
                },
            ))
    return tasks


def run_benchmark(
    distances: List[int] = None,
    error_rates: Optional[np.ndarray] = None,
    backends: List[str] = None,
    max_shots: int = 100_000,
    max_errors: int = 500,
    num_workers: int = 4,
    output_dir: str = "benchmark_results",
) -> Dict[str, list]:
    """Run Sinter benchmark across all specified backends.

    Parameters
    ----------
    distances : list of int
        Code distances. Default: [3, 5, 7].
    error_rates : ndarray, optional
        Physical error rates.
    backends : list of str
        Decoder backends to benchmark. Default: ["bposd", "pymatching"].
    max_shots : int
        Maximum shots per task.
    max_errors : int
        Target number of errors per task.
    num_workers : int
        Number of parallel worker processes.
    output_dir : str
        Directory for CSV output.

    Returns
    -------
    dict
        Mapping from backend name to list of sinter.TaskStats.
    """
    if not SINTER_AVAILABLE:
        raise ImportError(
            "sinter is required for benchmarks. "
            "Install with: pip install unified-qec[bposd]"
        )

    if distances is None:
        distances = [3, 5, 7]
    if backends is None:
        backends = ["bposd", "pymatching"]

    tasks = build_tasks(distances, error_rates)
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for backend in backends:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {backend}")
        print(f"{'='*60}")

        decoder = UnifiedQECDecoder(backend=backend)
        t0 = time.time()

        try:
            stats = sinter.collect(
                num_workers=num_workers,
                max_shots=max_shots,
                max_errors=max_errors,
                tasks=tasks,
                custom_decoders={"unified": decoder},
                decoders=["unified"],
            )
        except Exception as e:
            print(f"  ERROR: {backend} failed: {e}")
            results[backend] = []
            continue

        elapsed = time.time() - t0
        results[backend] = stats

        # Write CSV
        csv_path = output_path / f"benchmark_{backend}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "distance", "physical_error_rate", "logical_error_rate",
                "logical_error_rate_stderr", "shots", "errors",
                "seconds_per_shot",
            ])
            for s in stats:
                meta = s.json_metadata
                logical_rate = s.errors / s.shots if s.shots > 0 else 0.0
                stderr = (
                    np.sqrt(logical_rate * (1 - logical_rate) / s.shots)
                    if s.shots > 0 else 0.0
                )
                writer.writerow([
                    meta["d"],
                    meta["p"],
                    f"{logical_rate:.8f}",
                    f"{stderr:.8f}",
                    s.shots,
                    s.errors,
                    f"{elapsed / max(len(stats), 1):.4f}",
                ])

        print(f"  Completed {len(stats)} tasks in {elapsed:.1f}s")
        print(f"  Results: {csv_path}")

        # Summary
        for s in stats:
            meta = s.json_metadata
            rate = s.errors / s.shots if s.shots > 0 else 0.0
            print(
                f"  d={meta['d']:2d} p={meta['p']:.4f} "
                f"→ logical={rate:.6f} ({s.errors}/{s.shots})"
            )

    return results


def plot_threshold(
    results: Dict[str, list],
    output_path: str = "benchmark_results/threshold_plot.png",
):
    """Generate threshold plot from benchmark results.

    Parameters
    ----------
    results : dict
        Output of run_benchmark().
    output_path : str
        Path for the PNG figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (backend, stats) in zip(axes, results.items()):
        if not stats:
            continue

        distances = sorted(set(s.json_metadata["d"] for s in stats))
        for d in distances:
            d_stats = [s for s in stats if s.json_metadata["d"] == d]
            ps = [s.json_metadata["p"] for s in d_stats]
            rates = [
                s.errors / s.shots if s.shots > 0 else 0.0
                for s in d_stats
            ]
            ax.plot(ps, rates, "o-", label=f"d={d}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate")
        ax.set_ylabel("Logical error rate")
        ax.set_title(f"{backend.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Threshold plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sinter benchmark: compare QEC decoder backends"
    )
    parser.add_argument(
        "--distances", nargs="+", type=int, default=[3, 5, 7],
        help="Code distances to sweep (default: 3 5 7)"
    )
    parser.add_argument(
        "--backends", nargs="+", default=["bposd", "pymatching"],
        help="Decoder backends (default: bposd pymatching)"
    )
    parser.add_argument(
        "--max-shots", type=int, default=100_000,
        help="Max shots per task (default: 100000)"
    )
    parser.add_argument(
        "--max-errors", type=int, default=500,
        help="Target errors per task (default: 500)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--output", default="benchmark_results",
        help="Output directory (default: benchmark_results)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate threshold plot"
    )

    args = parser.parse_args()

    results = run_benchmark(
        distances=args.distances,
        backends=args.backends,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        num_workers=args.workers,
        output_dir=args.output,
    )

    if args.plot:
        plot_threshold(results, f"{args.output}/threshold_plot.png")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
