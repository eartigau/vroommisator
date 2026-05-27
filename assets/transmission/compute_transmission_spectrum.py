"""
Compute composite transmission spectrum for N surfaces from extracted_transmission_data.csv.
Transmission column is per-surface loss; composite = (1 - transmission) ** N.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

CSV_PATH = Path("extracted_transmission_data.csv")


def read_transmission_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read wavelength and transmission columns from CSV (case-insensitive headers)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        # Normalize headers for case-insensitive matching
        header_map = {h.lower(): h for h in reader.fieldnames}
        try:
            wl_key = header_map["wavelength"]
            tr_key = header_map["transmission"]
        except KeyError as exc:
            raise KeyError(
                "CSV must contain 'wavelength' and 'transmission' columns"
            ) from exc

        wavelengths: List[float] = []
        transmissions: List[float] = []
        for row in reader:
            try:
                wl = float(row[wl_key])
                tr = float(row[tr_key])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid numeric value in row: {row}") from exc
            wavelengths.append(wl)
            transmissions.append(tr)

    wl_arr = np.array(wavelengths, dtype=float)
    tr_arr = np.array(transmissions, dtype=float)

    # Sort by wavelength just in case
    order = np.argsort(wl_arr)
    wl_sorted = wl_arr[order]
    tr_sorted = tr_arr[order]

    # Consolidate duplicated wavelengths by averaging transmission loss
    unique_wl = []
    unique_tr = []
    if len(wl_sorted) == 0:
        return wl_sorted, tr_sorted

    run_start = 0
    for i in range(1, len(wl_sorted) + 1):
        end_of_run = i == len(wl_sorted) or wl_sorted[i] != wl_sorted[run_start]
        if end_of_run:
            wl_val = wl_sorted[run_start]
            tr_mean = float(np.mean(tr_sorted[run_start:i]))
            unique_wl.append(wl_val)
            unique_tr.append(tr_mean)
            run_start = i

    wl_unique = np.array(unique_wl, dtype=float)
    tr_unique = np.array(unique_tr, dtype=float)
    return wl_unique, tr_unique


def compute_composite(transmission_loss: np.ndarray, n_surfaces: int) -> np.ndarray:
    """Compute composite transmission from per-surface loss."""
    return np.power(1.0 - transmission_loss, n_surfaces)


def plot_spectrum(
    wavelengths_raw: np.ndarray,
    transmission_raw: np.ndarray,
    wavelengths_sampled: np.ndarray,
    composite_sampled: np.ndarray,
    n_surfaces: int,
    goal_composite: float,
    save_path: Path | None,
):
    """Plot raw points and sampled composite spectrum."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 7.5),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top panel: composite vs goal
    ax_top.plot(
        wavelengths_sampled,
        composite_sampled,
        "b-",
        lw=2,
        label=f"measured -- {n_surfaces} surfaces",
    )
    ax_top.axhline(
        goal_composite,
        color="orange",
        lw=2,
        ls="--",
        label=f"goal -- {n_surfaces} surfaces",
    )
    ax_top.set_ylabel("Composite Transmission")
    ax_top.set_title(f"Transmission Spectrum for N={n_surfaces} surfaces")
    ymin = 0.95 * float(np.min([np.min(composite_sampled), goal_composite]))
    ax_top.set_ylim(ymin, 1.0)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    # Bottom panel: delta vs goal
    delta_pct = (composite_sampled - goal_composite) / goal_composite * 100.0
    ax_bottom.plot(
        wavelengths_sampled,
        delta_pct,
        "k-",
        lw=1.5,
        label="measured - goal (%)",
    )
    ax_bottom.axhline(0.0, color="gray", lw=1, ls="--")
    ax_bottom.set_ylabel("Δ vs goal (%)")
    ax_bottom.set_xlabel("Wavelength")
    ax_bottom.set_ylim(-10.0, 10.0)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute composite transmission spectrum from extracted_transmission_data.csv"
    )
    parser.add_argument(
        "-n",
        "--surfaces",
        type=int,
        default=14,
        help="Number of surfaces (N), default 14",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Path to extracted_transmission_data.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path for the plot",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=10.0,
        help="Sampling step in nm (default 10)",
    )

    args = parser.parse_args()

    if args.surfaces <= 0:
        raise ValueError("Number of surfaces must be positive.")

    goal_loss = 0.005  # 0.5% per surface target

    wavelengths_raw, transmission_loss_raw = read_transmission_csv(args.csv)

    if args.step <= 0:
        raise ValueError("Sampling step must be positive.")

    if len(wavelengths_raw) < 2:
        raise ValueError("Need at least two distinct wavelength points for spline.")

    if np.any(np.diff(wavelengths_raw) <= 0):
        raise ValueError("Wavelengths must be strictly increasing after consolidation.")

    # Build spline on loss; no extrapolation beyond data range
    spline = CubicSpline(wavelengths_raw, transmission_loss_raw, extrapolate=False)
    wl_min, wl_max = wavelengths_raw.min(), wavelengths_raw.max()
    wavelengths_sampled = np.arange(wl_min, wl_max + args.step * 0.5, args.step)
    loss_sampled = spline(wavelengths_sampled)

    # Drop any NaNs that may appear at boundaries
    valid_mask = ~np.isnan(loss_sampled)
    wavelengths_sampled = wavelengths_sampled[valid_mask]
    loss_sampled = loss_sampled[valid_mask]

    composite_sampled = compute_composite(loss_sampled, args.surfaces)
    goal_composite = (1.0 - goal_loss) ** args.surfaces

    # Save composite sampled data back to CSV alongside inputs
    out_csv = args.csv.with_name(
        f"composite_transmission_N{args.surfaces}_step{int(args.step)}.csv"
    )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength", "transmission_loss", "composite_transmission"])
        for wl, loss, comp in zip(wavelengths_sampled, loss_sampled, composite_sampled):
            writer.writerow([wl, loss, comp])
    print(f"Saved composite data to {out_csv}")

    plot_spectrum(
        wavelengths_raw,
        transmission_loss_raw,
        wavelengths_sampled,
        composite_sampled,
        args.surfaces,
        goal_composite,
        args.output,
    )


if __name__ == "__main__":
    main()
