#!/usr/bin/env python3
"""
Build website assets for VROOMM documentation page.

What it does:
- Copies selected simulator outputs and plots into docs/assets/.
- Computes quick image statistics from NPY/FITS when available.
- Writes docs/assets/manifest.json consumed by docs/app.js.

Run:
  python make_webpage_assets.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

try:
    import yaml
except Exception:
    yaml = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from simulate_detector import (
    _bin_psf,
    _load_psf_file,
    load_xy_table,
    PIXEL_SIZE_MM,
    PSF_DIR_RECT,
    XY_TABLE,
)

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
MANIFEST = ASSETS / "manifest.json"
MANIFEST_JS = ASSETS / "manifest.js"
GALLERY_DIR = ASSETS / "gallery"

COPY_CANDIDATES = [
    "gui_screenshot.png",
    "detector_sim.png",
    "detector_sim.fits",
    "detector_sim.npy",
    "tmp_preview_test.png",
    "assets/transmission/compare_transmission_spectra.png",
    "assets/transmission/combined_transmission_spectrum.csv",
]

GALLERY_PATTERNS = [
    "detector_sim.png",
    "tmp_preview_test.png",
    "night_output/**/*.png",
]


def safe_rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def web_rel(path: Path) -> str:
    """Path relative to docs/ for direct use in docs/index.html."""
    return path.resolve().relative_to(DOCS.resolve()).as_posix()


def copy_if_exists(src_rel: str) -> dict[str, Any] | None:
    src = ROOT / src_rel
    if not src.exists() or not src.is_file():
        return None
    dst = ASSETS / src.name
    shutil.copy2(src, dst)
    return {
        "name": src.name,
        "source": safe_rel(src),
        "dest": safe_rel(dst),
        "web_path": web_rel(dst),
        "bytes": dst.stat().st_size,
    }


def percentile_stats(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "shape": list(arr.shape),
            "min": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    return {
        "shape": list(arr.shape),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
    }


def read_detector_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {}

    npy_path = ROOT / "detector_sim.npy"
    if npy_path.exists():
        try:
            arr = np.load(npy_path)
            stats["npy"] = percentile_stats(arr)
            stats["npy"]["path"] = safe_rel(npy_path)
        except Exception as exc:
            stats["npy_error"] = str(exc)

    fits_path = ROOT / "detector_sim.fits"
    if fits_path.exists():
        try:
            from astropy.io import fits

            with fits.open(fits_path) as hdul:
                arr = np.asarray(hdul[0].data)
            stats["fits"] = percentile_stats(arr)
            stats["fits"]["path"] = safe_rel(fits_path)
        except Exception as exc:
            stats["fits_error"] = str(exc)

    return stats


def read_transmission_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    csv_path = ROOT / "assets/transmission/combined_transmission_spectrum.csv"
    if not csv_path.exists():
        return out
    try:
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        if data.ndim != 2 or data.shape[1] < 2:
            return out
        wave = data[:, 0]
        tran = data[:, 1]
        i_peak = int(np.nanargmax(tran))
        out = {
            "path": safe_rel(csv_path),
            "wavelength_min_nm": float(np.nanmin(wave)),
            "wavelength_max_nm": float(np.nanmax(wave)),
            "transmission_min": float(np.nanmin(tran)),
            "transmission_max": float(np.nanmax(tran)),
            "peak_nm": float(wave[i_peak]),
            "samples": {
                "380": float(tran[int(np.nanargmin(np.abs(wave - 380.0)))]),
                "400": float(tran[int(np.nanargmin(np.abs(wave - 400.0)))]),
                "500": float(tran[int(np.nanargmin(np.abs(wave - 500.0)))]),
                "700": float(tran[int(np.nanargmin(np.abs(wave - 700.0)))]),
                "900": float(tran[int(np.nanargmin(np.abs(wave - 900.0)))]),
            },
        }
    except Exception as exc:
        out = {"error": str(exc)}
    return out


def read_methods_summary() -> dict[str, Any]:
    cfg = ROOT / "simulate_params.yaml"
    if not cfg.exists():
        return {}
    if yaml is None:
        return {"error": "PyYAML not available; cannot parse simulate_params.yaml"}
    try:
        p = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {"error": str(exc)}

    obs = p.get("observatory", {}) or {}
    tel = p.get("telescope", {}) or {}
    obv = p.get("observation", {}) or {}
    st = p.get("star", {}) or {}
    mod = p.get("model", {}) or {}
    sky = p.get("sky", {}) or {}
    telu = p.get("telluric", {}) or {}

    return {
        "config_path": safe_rel(cfg),
        "observatory": {
            "name": obs.get("name"),
            "lat_deg": obs.get("lat_deg"),
            "lon_deg": obs.get("lon_deg"),
            "elevation_m": obs.get("elevation_m"),
        },
        "telescope": {
            "name": tel.get("name"),
            "diameter_m": tel.get("diameter_m"),
            "peak_throughput": tel.get("peak_throughput"),
        },
        "observation": {
            "exposure_s": obv.get("exposure_s"),
            "jd_utc": obv.get("jd_utc"),
        },
        "source": {
            "spectrum_mode": p.get("spectrum_mode"),
            "model_teff": mod.get("teff"),
            "model_logg": mod.get("logg"),
            "star_mag": st.get("R_mag"),
            "star_mag_band": st.get("mag_band", "R"),
            "star_vsini_kms": st.get("vsini_kms"),
        },
        "sampling": {
            "wave_step_pix_frac": p.get("wave_step_pix_frac"),
            "blaze": p.get("blaze"),
        },
        "environment": {
            "sky_enabled": sky.get("enabled"),
            "sky_mag": sky.get("R_mag"),
            "telluric_enabled": telu.get("enabled"),
            "telluric_airmass": telu.get("airmass"),
            "telluric_water_od": telu.get("water_od"),
        },
    }


def build_gallery(max_items: int = 24) -> list[dict[str, Any]]:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    found: list[Path] = []
    seen = set()
    for pattern in GALLERY_PATTERNS:
        for p in ROOT.glob(pattern):
            if not p.is_file():
                continue
            if p.resolve() == (ROOT / "docs" / "assets" / "gui_screenshot.png").resolve():
                continue
            if p.suffix.lower() != ".png":
                continue
            r = str(p.resolve())
            if r in seen:
                continue
            seen.add(r)
            found.append(p)

    found.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    selected = found[:max_items]

    for old in GALLERY_DIR.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass

    out = []
    for i, src in enumerate(selected, start=1):
        base = src.name
        dst_name = f"{i:02d}_{base}"
        dst = GALLERY_DIR / dst_name
        shutil.copy2(src, dst)
        out.append(
            {
                "title": base,
                "source": safe_rel(src),
                "dest": safe_rel(dst),
                "web_path": web_rel(dst),
                "bytes": dst.stat().st_size,
            }
        )
    return out


def generate_lecture_plots() -> list[dict[str, Any]]:
    """Build pedagogical plots that explain how simulation products are assembled."""
    if plt is None:
        return []

    out: list[dict[str, Any]] = []

    def _record(path: Path, title: str) -> None:
        out.append(
            {
                "title": title,
                "dest": safe_rel(path),
                "web_path": web_rel(path),
                "bytes": path.stat().st_size,
            }
        )

    # 1) PSF construction: native 80x80 and true 4x4->20x20 binning.
    try:
        psf_files = sorted(Path(PSF_DIR_RECT).glob("R*.txt"))
        if psf_files:
            psf_raw = _load_psf_file(str(psf_files[0]))
            psf_bin, _ = _bin_psf(psf_raw)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
            ax[0].imshow(psf_raw, origin="lower", cmap="magma")
            ax[0].set_title("Native Zemax PSF (80x80 @ 3 um)")
            ax[1].imshow(psf_bin, origin="lower", cmap="magma")
            ax[1].set_title("True binned PSF (20x20 @ 12 um)")
            for a in ax:
                a.set_xlabel("x pixel")
                a.set_ylabel("y pixel")
            fig.tight_layout()
            p = ASSETS / "lecture_psf_construction.png"
            fig.savefig(p)
            plt.close(fig)
            _record(p, "PSF construction")
    except Exception:
        pass

    # 2) Trace geometry from XY table.
    try:
        table = load_xy_table(str(XY_TABLE))
        fig, ax = plt.subplots(figsize=(7, 6), dpi=170)
        orders = sorted(table.keys())
        sel = orders[:: max(1, len(orders) // 12)]
        for o in sel:
            rows = np.asarray(table[o], dtype=float)
            x = rows[:, 1] / PIXEL_SIZE_MM
            y = rows[:, 2] / PIXEL_SIZE_MM
            ax.plot(x, y, lw=1.1, alpha=0.8)
            ax.text(x[len(x) // 2], y[len(y) // 2], str(o), fontsize=7, alpha=0.8)
        ax.set_title("Order traces from Zemax XY map")
        ax.set_xlabel("x detector pixel")
        ax.set_ylabel("y detector pixel")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = ASSETS / "lecture_trace_map.png"
        fig.savefig(p)
        plt.close(fig)
        _record(p, "Trace geometry")
    except Exception:
        table = None

    # 3) Spline interpolation of one order (x(lambda), y(lambda)).
    try:
        if table is None:
            table = load_xy_table(str(XY_TABLE))
        order = sorted(table.keys())[len(table) // 2]
        rows = np.asarray(table[order], dtype=float)
        w_nm = rows[:, 0] * 1000.0
        x_px = rows[:, 1] / PIXEL_SIZE_MM
        y_px = rows[:, 2] / PIXEL_SIZE_MM
        i = np.argsort(w_nm)
        w_nm, x_px, y_px = w_nm[i], x_px[i], y_px[i]
        csx = CubicSpline(w_nm, x_px, extrapolate=True)
        csy = CubicSpline(w_nm, y_px, extrapolate=True)
        w_dense = np.linspace(w_nm.min(), w_nm.max(), 400)

        fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=170)
        ax[0].plot(w_nm, x_px, "o", ms=4, label="samples")
        ax[0].plot(w_dense, csx(w_dense), "-", lw=1.6, label="cubic spline")
        ax[0].set_title(f"Order {order}: x(lambda)")
        ax[0].set_xlabel("wavelength [nm]")
        ax[0].set_ylabel("x [px]")
        ax[0].legend(fontsize=8)

        ax[1].plot(w_nm, y_px, "o", ms=4, label="samples")
        ax[1].plot(w_dense, csy(w_dense), "-", lw=1.6, label="cubic spline")
        ax[1].set_title(f"Order {order}: y(lambda)")
        ax[1].set_xlabel("wavelength [nm]")
        ax[1].set_ylabel("y [px]")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        p = ASSETS / "lecture_spline_sampling.png"
        fig.savefig(p)
        plt.close(fig)
        _record(p, "Spline interpolation")
    except Exception:
        pass

    # 4) Toy 2D image build along one trace: sampled flux -> accumulated detector.
    try:
        if table is None:
            table = load_xy_table(str(XY_TABLE))
        order = sorted(table.keys())[len(table) // 3]
        rows = np.asarray(table[order], dtype=float)
        w_nm = rows[:, 0] * 1000.0
        x_px = rows[:, 1] / PIXEL_SIZE_MM
        y_px = rows[:, 2] / PIXEL_SIZE_MM
        i = np.argsort(w_nm)
        w_nm, x_px, y_px = w_nm[i], x_px[i], y_px[i]
        csx = CubicSpline(w_nm, x_px, extrapolate=True)
        csy = CubicSpline(w_nm, y_px, extrapolate=True)

        w_dense = np.linspace(w_nm.min(), w_nm.max(), 180)
        flux = (
            0.25
            + np.exp(-0.5 * ((w_dense - np.median(w_dense)) / 5.0) ** 2)
            + 0.6 * np.exp(-0.5 * ((w_dense - (w_dense.min() + 0.25 * (np.ptp(w_dense)))) / 2.0) ** 2)
        )

        x_d = csx(w_dense)
        y_d = csy(w_dense)
        x0, x1 = float(np.min(x_px)), float(np.max(x_px))
        y0, y1 = float(np.min(y_px)), float(np.max(y_px))
        x_n = (x_d - x0) / max(x1 - x0, 1e-6)
        y_n = (y_d - y0) / max(y1 - y0, 1e-6)

        det = np.zeros((320, 320), dtype=float)
        ker_x = np.arange(-3, 4)
        kx, ky = np.meshgrid(ker_x, ker_x)
        kern = np.exp(-0.5 * (kx**2 + ky**2) / 1.2**2)
        kern /= kern.sum()

        for xx, yy, ff in zip(x_n, y_n, flux):
            xi = int(np.clip(xx * (det.shape[1] - 1), 4, det.shape[1] - 5))
            yi = int(np.clip(yy * (det.shape[0] - 1), 4, det.shape[0] - 5))
            det[yi - 3 : yi + 4, xi - 3 : xi + 4] += ff * kern

        fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=170)
        ax[0].plot(x_n * (det.shape[1] - 1), y_n * (det.shape[0] - 1), "-", lw=1.2)
        ax[0].scatter(x_n[::12] * (det.shape[1] - 1), y_n[::12] * (det.shape[0] - 1),
                      c=flux[::12], s=15, cmap="viridis")
        ax[0].set_title("Spline trace + sampled wavelengths")
        ax[0].set_xlim(0, det.shape[1])
        ax[0].set_ylim(0, det.shape[0])

        ax[1].plot(w_dense, flux, color="#4cc9f0")
        ax[1].set_title("1D spectrum sampled on trace")
        ax[1].set_xlabel("wavelength [nm]")
        ax[1].set_ylabel("relative photons")

        ax[2].imshow(det, origin="lower", cmap="inferno")
        ax[2].set_title("Accumulated 2D detector image")
        ax[2].set_xlabel("x [px]")
        ax[2].set_ylabel("y [px]")
        fig.tight_layout()
        p = ASSETS / "lecture_image_assembly.png"
        fig.savefig(p)
        plt.close(fig)
        _record(p, "2D image assembly")
    except Exception:
        pass

    return out


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    copied = []
    for rel in COPY_CANDIDATES:
        info = copy_if_exists(rel)
        if info is not None:
            copied.append(info)

    gallery = build_gallery()
    lecture_plots = generate_lecture_plots()

    manifest = {
        "project": "VROOMM Simulator",
        "copied_assets": copied,
        "gallery": gallery,
        "lecture_plots": lecture_plots,
        "detector_stats": read_detector_stats(),
        "transmission_summary": read_transmission_snapshot(),
        "methods": read_methods_summary(),
        "notes": {
            "screenshot_hint": "Place GUI screenshot at docs/assets/gui_screenshot.png",
        },
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    MANIFEST_JS.write_text(
        "window.VROOMM_MANIFEST = " + json.dumps(manifest, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {safe_rel(MANIFEST)}")
    print(f"Wrote {safe_rel(MANIFEST_JS)}")
    print(f"Copied {len(copied)} asset(s) into {safe_rel(ASSETS)}")
    print(f"Prepared gallery with {len(gallery)} image(s)")
    print(f"Prepared lecture plots with {len(lecture_plots)} figure(s)")


if __name__ == "__main__":
    main()
