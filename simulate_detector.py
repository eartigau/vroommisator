"""
simulate_detector.py
====================
Simulate the 2D detector image for the VROOMM cross-dispersed spectrograph.

Data layout
-----------
zemax_data/VROOMM_V04_XY.txt
    Columns: ORDER  WAVELENGTH(µm)  XPOS(mm)  YPOS(mm)
    91 orders (67–157), 11 wavelength samples each.

zemax_data/images_fibre_rectangulaire/R{order}{N}.txt
    Zemax PSF image, 80×80 pixels at 3 µm/px.
    N is the 1-indexed row number of that order in the XY table.

Usage
-----
    python simulate_detector.py            # run simulation, save .npy + .png
    python simulate_detector.py --help
"""

import os
import re
import bz2
import argparse
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.interpolate import CubicSpline
from tqdm import tqdm
try:
    import yaml
except ImportError:
    yaml = None  # handled at runtime in load_params

# ── physical constants ─────────────────────────────────────────────────────────
PIXEL_SIZE_MM   = 0.012   # detector pixel size (12 µm)

# ── photometric constants ─────────────────────────────────────────────────────
# Planck constant × speed of light: hc = 1.9864×10⁻¹⁶ erg·nm
_H_ERG_S    = 6.626e-27       # [erg·s]
_C_NM_S     = 2.998e17        # [nm/s]
_HC_ERG_NM  = _H_ERG_S * _C_NM_S  # 1.9864e-16 erg·nm
# Cousins R-band (Bessell 1990): box approximation 570–730 nm, λ_eff = 641 nm
# Vega zero-point flux density: F_ν = 3080 Jy → F_λ ≈ 2.25×10⁻⁹ erg/cm²/s/Å
_VEGA_F_LAM_R = 2.25e-9   # erg/cm²/s/Å  (Vega, R_Cousins = 0)
_R_BAND_MIN   = 570.0     # nm
_R_BAND_MAX   = 730.0     # nm
_R_BAND_EFF   = 641.0     # nm  (effective wavelength for photon-energy conversion)
# Gaia DR3 RP (Grp): approximate Vega zero-point spectral flux density.
# This keeps absolute scaling in the right regime while the passband shape is
# taken from the fetched RP throughput profile.
_VEGA_F_LAM_GRP = 1.30e-9   # erg/cm²/s/Å  (approximate Vega RP zero-point)
PSF_PIXEL_SIZE  = 0.003   # PSF pixel size (3 µm)
BIN_FACTOR      = 4       # 3 µm → 12 µm (must divide 80 evenly → 20×20 output)
PSF_NATIVE_SIZE = 80      # pixels before binning
PSF_BIN_SIZE    = PSF_NATIVE_SIZE // BIN_FACTOR   # 20 pixels
PSF_HALF        = PSF_BIN_SIZE // 2               # 10  (stamp radius)
DETECTOR_SIZE   = 4096    # nominal detector size (pixels)
CANVAS_MARGIN   = 20      # extra pixels on each side during simulation;
                          # orders fading at edges are not clipped prematurely.
                          # The canvas is trimmed to the actual footprint afterward.

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(_HERE, "assets", "zemax_data")
XY_TABLE     = os.path.join(DATA_DIR, "VROOMM_V04_XY.txt")
PSF_DIR_RECT = os.path.join(DATA_DIR, "images_fibre_rectangulaire")
PSF_DIR_OCT  = os.path.join(DATA_DIR, "images_fibre_octogonale")
_TRANSMISSION_CSV = os.path.join(_HERE, "assets", "transmission",
                                 "combined_transmission_spectrum.csv")
_GAIA_GRP_VOT_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=GAIA/GAIA3.GRP"
_GAIA_GRP_CACHE = os.path.join(_HERE, "assets", "transmission", "gaia_grp.vot")

# ── wavelength-dependent system transmission ────────────────────────────────
# combined_transmission_spectrum.csv: back-end optics × front-end × EMCCD QE.
# Replaces the scalar peak_throughput with a fully wavelength-dependent curve.
# Wavelengths outside [360, 930] nm are set to zero (instrument doesn't respond).
_transmission_spline_cache = None

def _get_transmission_spline():
    """Return a callable T(wave_nm) → [0, 1]; zero outside the CSV's domain."""
    global _transmission_spline_cache
    if _transmission_spline_cache is not None:
        return _transmission_spline_cache
    from scipy.interpolate import UnivariateSpline
    data   = np.genfromtxt(_TRANSMISSION_CSV, delimiter=',', skip_header=1)
    w_csv, t_csv = data[:, 0], data[:, 1]
    spl    = UnivariateSpline(w_csv, t_csv, s=0, ext=1)   # ext=1 → 0 outside domain
    _transmission_spline_cache = spl
    return spl


def _load_gaia_grp_passband(cache_path: str = _GAIA_GRP_CACHE) -> tuple:
    """
    Return Gaia DR3 RP passband (wave_nm, transmission) from SVO, cached on disk.

    The SVO endpoint returns a VOTable. We keep a local cache so simulations can
    run offline after one successful download.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not os.path.exists(cache_path):
        urllib.request.urlretrieve(_GAIA_GRP_VOT_URL, cache_path)

    tree = ET.parse(cache_path)
    root = tree.getroot()
    rows = []
    for tr in root.findall('.//{*}TR'):
        vals = [td.text for td in tr.findall('{*}TD')]
        if len(vals) < 2:
            continue
        try:
            w = float(vals[0])
            t = float(vals[1])
        except (TypeError, ValueError):
            continue
        if np.isfinite(w) and np.isfinite(t):
            rows.append((w, t))

    if len(rows) < 10:
        raise ValueError("Could not parse Gaia RP passband from cached VOTable.")

    arr = np.asarray(rows, dtype=float)
    wave_nm = arr[:, 0]
    trans = arr[:, 1]
    # SVO wavelength axis is commonly in Angstrom; convert to nm if needed.
    if np.nanmedian(wave_nm) > 2000.0:
        wave_nm = wave_nm / 10.0
    if np.nanmax(trans) > 0:
        trans = trans / np.nanmax(trans)
    return wave_nm, trans


def _blackbody_nm(wave_nm: np.ndarray, T_K: float) -> np.ndarray:
    """Planck blackbody spectral radiance [erg/s/cm²/Å] vs wavelength [nm]."""
    _kB    = 1.3806e-16   # erg/K
    _h     = 6.626e-27    # erg·s
    _c_cms = 2.998e10     # cm/s
    lam_cm = wave_nm * 1.0e-7
    x = np.clip((_h * _c_cms) / (lam_cm * _kB * T_K), 0.0, 500.0)
    return (2.0 * _h * _c_cms**2 / lam_cm**5) / (np.expm1(x))


def _fp_airy_transmission(wave_nm: np.ndarray, cavity_cm: float, finesse: float) -> np.ndarray:
    """
    Fabry-Perot Airy transmission for a given cavity length and finesse.

    Parameters
    ----------
    wave_nm : ndarray
        Wavelength grid [nm].
    cavity_cm : float
        Physical cavity length [cm].
    finesse : float
        FP finesse (dimensionless).
    """
    lam_m = np.asarray(wave_nm, dtype=float) * 1.0e-9
    L_m = float(cavity_cm) * 1.0e-2
    f = max(float(finesse), 1.0)
    # Airy coefficient from finesse approximation: finesse ≈ pi*sqrt(F)/2
    airy_F = (2.0 * f / np.pi) ** 2
    delta = 4.0 * np.pi * L_m / lam_m
    return 1.0 / (1.0 + airy_F * np.sin(0.5 * delta) ** 2)


def _cs_linextrap(x: np.ndarray, y: np.ndarray) -> object:
    """
    Return a callable that is a CubicSpline inside [x[0], x[-1]] and
    switches to the tangent line (linear extrapolation) outside that range.

    This prevents the cubic polynomial from "rolling back" when evaluated
    far beyond the last reference point.
    """
    cs   = CubicSpline(x, y, extrapolate=False)
    d0   = cs(x[0],  1)    # first derivative at left  boundary
    d1   = cs(x[-1], 1)    # first derivative at right boundary
    y0, y1 = np.asarray(y[0]), np.asarray(y[-1])
    x0, x1 = float(x[0]),     float(x[-1])

    def _eval(x_new: np.ndarray) -> np.ndarray:
        x_new  = np.asarray(x_new, dtype=float)
        scalar = x_new.ndim == 0
        x_new  = np.atleast_1d(x_new)
        result = np.empty(x_new.shape + np.asarray(y0).shape, dtype=float)
        inside = (x_new >= x0) & (x_new <= x1)
        left   = x_new < x0
        right  = x_new > x1
        if inside.any():
            result[inside] = cs(x_new[inside])
        if left.any():
            result[left]  = y0 + d0 * (x_new[left]  - x0)
        if right.any():
            result[right] = y1 + d1 * (x_new[right] - x1)
        return result[0] if scalar else result

    return _eval

# ── PSF I/O ────────────────────────────────────────────────────────────────────

def _load_psf_file(path: str) -> np.ndarray:
    """Parse a Zemax PSF histogram file; return 80×80 float64 array."""
    rows = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == PSF_NATIVE_SIZE:
                try:
                    rows.append([float(v) for v in parts])
                except ValueError:
                    continue
    if len(rows) != PSF_NATIVE_SIZE:
        raise ValueError(
            f"Expected {PSF_NATIVE_SIZE} data rows, got {len(rows)} in {path}"
        )
    return np.array(rows, dtype=np.float64)


def _bin_psf(arr: np.ndarray, factor: int = BIN_FACTOR) -> tuple:
    """
    Block-sum bin a 2-D array by `factor` in each dimension.

    Returns
    -------
    binned : np.ndarray
        Normalised PSF (sums to 1).
    raw_total : float
        Sum of the binned array before normalisation, proportional to the
        actual grating throughput at this wavelength.
    """
    h, w = arr.shape
    nh, nw = h // factor, w // factor
    binned = arr[: nh * factor, : nw * factor]
    binned = binned.reshape(nh, factor, nw, factor).sum(axis=(1, 3))
    raw_total = float(binned.sum())
    if raw_total > 0:
        binned /= raw_total   # normalise: PSF shape sums to 1
    return binned, raw_total


# ── XY table ───────────────────────────────────────────────────────────────────

def load_xy_table(path: str = XY_TABLE) -> dict:
    """
    Load the order/wavelength/position table.

    Returns
    -------
    dict
        {order_int: [(wave_um, x_mm, y_mm), ...]}
        Rows are in file order (descending wavelength within each order).
    """
    table: dict = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                order = int(float(parts[0]))
                wave  = float(parts[1])
                x_mm  = float(parts[2])
                y_mm  = float(parts[3])
            except ValueError:
                continue
            table.setdefault(order, []).append((wave, x_mm, y_mm))
    return table


def _estimate_nm_per_pixel(xy_table: dict) -> float:
    """
    Estimate a typical local dispersion in nm/pixel from the XY table.

    Uses consecutive wavelength/position samples along each order:
      nm_per_pix = |delta_lambda_nm| / |delta_trace_pix|
    and returns the median over all finite positive samples.
    """
    samples = []
    for rows in xy_table.values():
        if len(rows) < 2:
            continue
        for i in range(len(rows) - 1):
            w0_um, x0_mm, y0_mm = rows[i]
            w1_um, x1_mm, y1_mm = rows[i + 1]
            dl_nm = abs((w1_um - w0_um) * 1000.0)
            dmm = np.hypot(x1_mm - x0_mm, y1_mm - y0_mm)
            dpix = dmm / PIXEL_SIZE_MM
            if dpix > 0.0 and dl_nm > 0.0:
                val = dl_nm / dpix
                if np.isfinite(val) and val > 0.0:
                    samples.append(val)

    if not samples:
        raise ValueError("Could not estimate nm/pixel from XY table (no valid samples).")

    return float(np.median(np.asarray(samples, dtype=np.float64)))


# ── PSF library ────────────────────────────────────────────────────────────────

class PSFLibrary:
    """
    Load and cache all PSF files for one fibre type.

    The filename encodes order and position index N:
        R{order}{N}.txt   e.g. R671.txt → order=67, N=1
    N is 1-indexed and corresponds to the Nth row of that order in the XY table.

    `get_psf(order, wave_um)` returns a 20×20 normalised PSF interpolated
    (or extrapolated from the two edge entries) to the requested wavelength.
    """

    def __init__(self, psf_dir: str, order_wave_map: dict):
        """
        Parameters
        ----------
        psf_dir : str
            Directory containing R*.txt PSF files.
        order_wave_map : dict
            {order_int: [wave1_um, wave2_um, ...]}  (in XY-table file order).
        """
        self.psfs: dict = {}         # order → [(wave_um, psf_20x20), ...] ascending
        self.throughput: dict = {}    # order → [(wave_um, raw_total), ...] ascending
        self.psf_splines: dict = {}   # order → CubicSpline(wave → 80×80 PSF)
        self._load(psf_dir, order_wave_map)

    def _load(self, psf_dir: str, order_wave_map: dict) -> None:
        """Discover and load all R*.txt files."""
        # Build a fast lookup: (order, N) → wavelength
        wave_lookup: dict = {}
        for order, waves in order_wave_map.items():
            for n, w in enumerate(waves, start=1):
                wave_lookup[(order, n)] = w

        # Sort known orders longest-string first to avoid prefix ambiguity
        # e.g. order 100 before order 10 (not present, but defensive)
        known_orders = sorted(order_wave_map.keys(),
                              key=lambda o: len(str(o)), reverse=True)

        pattern = re.compile(r"^R(\d+)\.txt$", re.IGNORECASE)
        loaded = skipped = 0

        for fname in sorted(os.listdir(psf_dir)):
            m = pattern.match(fname)
            if not m:
                continue
            num_str = m.group(1)

            # Identify order prefix
            order = None
            n_str = None
            for o in known_orders:
                o_str = str(o)
                if num_str.startswith(o_str):
                    tail = num_str[len(o_str):]
                    if tail.isdigit() and tail:
                        order = o
                        n_str = tail
                        break
            if order is None:
                skipped += 1
                continue

            n = int(n_str)
            wave_um = wave_lookup.get((order, n))
            if wave_um is None:
                skipped += 1
                continue

            path = os.path.join(psf_dir, fname)
            try:
                psf_raw = _load_psf_file(path)
                _psf_bin, raw_total = _bin_psf(psf_raw)
                # Normalise the unbinned PSF for storage; binning happens after
                # the sub-pixel shift so spline artefacts on the coarse grid
                # are avoided.
                raw_sum = psf_raw.sum()
                psf_norm = psf_raw / raw_sum if raw_sum > 0 else psf_raw
            except Exception as exc:
                print(f"  [warn] could not load {fname}: {exc}")
                skipped += 1
                continue

            self.psfs.setdefault(order, []).append((wave_um, psf_norm))
            self.throughput.setdefault(order, []).append((wave_um, raw_total))
            loaded += 1

        # Sort each order's entries by wavelength ascending
        for order in self.psfs:
            self.psfs[order].sort(key=lambda t: t[0])
            self.throughput[order].sort(key=lambda t: t[0])

        # Precompute a CubicSpline for smooth PSF interpolation per order.
        # CubicSpline on the (n_ref, 80, 80) unbinned stack gives continuous
        # first and second derivatives across the reference wavelengths.
        # Binning is deferred until after the sub-pixel shift.
        for order, entries in self.psfs.items():
            waves     = np.array([e[0] for e in entries])      # (n_ref,)
            psf_stack = np.array([e[1] for e in entries])      # (n_ref, 80, 80)
            self.psf_splines[order] = CubicSpline(waves, psf_stack,
                                                  axis=0, extrapolate=True)

        print(f"  PSF library: {loaded} files loaded, {skipped} skipped")

    def get_psf(self, order: int, wave_um: float) -> np.ndarray:
        """
        Return the normalised 80×80 unbinned PSF at `wave_um` for `order`.

        The caller is responsible for shifting (at the 80×80 scale) and then
        binning to 20×20 so that sub-pixel shifts are applied before
        down-sampling, avoiding spline artefacts on the coarse grid.
        """
        cs = self.psf_splines.get(order)
        if cs is None:
            raise KeyError(f"Order {order} not found in PSF library")

        # Clamp to avoid cubic-extrapolation runaway beyond reference range
        wave_clamped = float(np.clip(wave_um, cs.x[0], cs.x[-1]))
        psf = cs(wave_clamped)              # shape (80, 80)
        psf = np.clip(psf, 0.0, None)
        total = psf.sum()
        if total > 0:
            psf /= total
        return psf


# ── spectrum ───────────────────────────────────────────────────────────────────

def make_spectrum(wave_um: np.ndarray) -> np.ndarray:
    """
    Flat continuum (= 1) plus Gaussian emission lines every 10 nm,
    FWHM = 0.1 nm.

    Parameters
    ----------
    wave_um : array_like
        Wavelength(s) in µm.

    Returns
    -------
    np.ndarray
        Flux values, same shape as `wave_um`.
    """
    wave_um = np.asarray(wave_um, dtype=np.float64)
    flux = np.ones_like(wave_um)

    fwhm_um  = 0.1e-3                          # 0.1 nm in µm
    sigma_um = fwhm_um / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    wave_min, wave_max = float(wave_um.min()), float(wave_um.max())
    # first centre at or just below wave_min, then step every 10 nm
    first_centre = np.floor(wave_min / 0.01) * 0.01
    centres = np.arange(first_centre, wave_max + 0.01, 0.01)

    for c in centres:
        flux += np.exp(-0.5 * ((wave_um - c) / sigma_um) ** 2)

    return flux


# ── detector coordinate helper ─────────────────────────────────────────────────

def mm_to_pix_exact(x_mm: float, y_mm: float, margin: int = 0) -> tuple:
    """
    Convert Zemax focal-plane coordinates (mm, origin = optical axis)
    to exact floating-point detector pixel coordinates.

    Parameters
    ----------
    margin : int
        Extra pixels added to each side of the canvas (e.g. CANVAS_MARGIN).
        The optical axis sits at DETECTOR_SIZE/2 + margin from the corner.

    Returns
    -------
    cx_f, cy_f : float
        Sub-pixel position on the canvas (0-indexed, origin at corner).
    """
    cx_f = x_mm / PIXEL_SIZE_MM + DETECTOR_SIZE / 2 + margin
    cy_f = y_mm / PIXEL_SIZE_MM + DETECTOR_SIZE / 2 + margin
    return cx_f, cy_f


# ── sky line broadening ───────────────────────────────────────────────────────

_C_KMS = 2.998e5   # speed of light [km/s]


def _broaden_lines_to_grid(
    line_wave_nm: np.ndarray,
    line_flux: np.ndarray,
    v_fwhm_kms: float = 1.0,
    grid_step_nm: float = 2e-4,
) -> tuple:
    """
    Spread a sparse emission line list onto a dense wavelength grid by
    convolving each line with a Gaussian of FWHM = v_fwhm_kms / c × λ.

    The Gaussian profile is normalised so that its integral equals the
    total flux of the line, conserving photon counts.

    Parameters
    ----------
    line_wave_nm  : wavelength of each line [nm]
    line_flux     : total flux per line (arbitrary or absolute units)
    v_fwhm_kms    : velocity FWHM [km/s]  (thermal + turbulent broadening)
    grid_step_nm  : output grid spacing [nm]  (default 0.0002 nm = 0.2 pm)

    Returns
    -------
    grid_nm    : np.ndarray  dense wavelength array [nm]
    flux_nm    : np.ndarray  flux density [line_flux units / nm]
    """
    sig_factor = v_fwhm_kms / (2.3548202 * _C_KMS)   # σ_nm = λ_nm × sig_factor
    n_clip     = 5                                     # truncate at ± n_clip σ

    wave_min = max(0.0, float(line_wave_nm.min()) - 1.0)
    wave_max = float(line_wave_nm.max()) + 1.0
    grid_nm  = np.arange(wave_min, wave_max + grid_step_nm, grid_step_nm)
    result   = np.zeros(len(grid_nm), dtype=np.float64)

    for lw, lf in zip(line_wave_nm, line_flux):
        sig  = lw * sig_factor
        half = n_clip * sig
        i_lo = max(0, int((lw - half - wave_min) / grid_step_nm))
        i_hi = min(len(grid_nm), int((lw + half - wave_min) / grid_step_nm) + 2)
        if i_lo >= i_hi:
            continue
        g     = grid_nm[i_lo:i_hi]
        gauss = np.exp(-0.5 * ((g - lw) / sig) ** 2)
        # Normalise so ∫G dλ = 1 [1/nm]; then lf × G has units [lf/nm]
        gauss *= 1.0 / (gauss.sum() * grid_step_nm)
        result[i_lo:i_hi] += lf * gauss

    return grid_nm, result


# ── rotational broadening ────────────────────────────────────────────────────

def _apply_vsini(wave_nm: np.ndarray, flux: np.ndarray,
                vsini_kms: float, epsilon: float = 0.6) -> tuple:
    """
    Apply rotational broadening (Gray 2005 §18) to a stellar spectrum.

    The spectrum is resampled onto a log-uniform wavelength grid so that the
    rotational profile has the same width (in pixels) at every wavelength;
    a single kernel is therefore sufficient for the entire domain.
    ``astropy.convolution.convolve`` is used for the convolution.

    Parameters
    ----------
    wave_nm   : wavelength array [nm], must be sorted ascending
    flux      : flux array (any consistent units)
    vsini_kms : projected rotation velocity [km/s]
    epsilon   : linear limb-darkening coefficient (Gray eq. 18.14; 0 ≤ ε ≤ 1)

    Returns
    -------
    (wave_nm, flux_broadened) — same wavelength grid as input
    """
    from astropy.convolution import convolve, CustomKernel

    _C_KMS = 2.998e5

    # Guard: discard non-positive wavelengths (BT-Settl models can start at 0 Å)
    valid   = wave_nm > 0.0
    wave_nm = wave_nm[valid]
    flux    = flux[valid]

    # Velocity step: 1/20 of vsini gives ≥20 pixels per half-width.
    # Capped at 0.5 km/s to avoid over-sampling for slow rotators.
    vel_step_kms = max(0.5, vsini_kms / 20.0)
    d_ln = vel_step_kms / _C_KMS          # uniform step in ln(λ)

    # 1. Build log-uniform grid
    ln_min  = np.log(wave_nm[0])
    ln_max  = np.log(wave_nm[-1])
    n_log   = int(np.ceil((ln_max - ln_min) / d_ln)) + 1
    wave_log = np.exp(np.linspace(ln_min, ln_max, n_log))

    # 2. Spline input spectrum onto the log grid
    spl      = CubicSpline(wave_nm, flux)
    flux_log = spl(wave_log)

    # 3. Rotational broadening kernel (Gray 2005 eq. 18.14)
    #    x = Δv / vsini_kms  ∈ [-1, 1]
    nhalf = int(np.ceil(vsini_kms / vel_step_kms)) + 2
    k_x   = np.arange(-nhalf, nhalf + 1) * (vel_step_kms / vsini_kms)
    mask  = np.abs(k_x) <= 1.0
    kernel = np.zeros_like(k_x)
    xm = k_x[mask]
    # numerator: two-component profile (disk-integrated)
    kernel[mask] = (
        2.0 * (1.0 - epsilon) * np.sqrt(np.maximum(1.0 - xm**2, 0.0))
        + 0.5 * np.pi * epsilon * (1.0 - xm**2)
    )
    kernel /= kernel.sum()   # normalise so convolution conserves total flux

    # 4. Convolve via astropy (boundary='extend' avoids edge artefacts)
    flux_broad_log = convolve(flux_log, CustomKernel(kernel), boundary='extend')

    # 5. Spline back to original wavelength grid
    flux_broad = CubicSpline(wave_log, flux_broad_log)(wave_nm)

    n_kernel_pix = int(mask.sum())
    print(
        f"  vsini broadening: vsini={vsini_kms:.1f} km/s, ε={epsilon:.2f}, "
        f"vel_step={vel_step_kms:.2f} km/s/pix, kernel={n_kernel_pix} pixels "
        f"(log grid: {n_log:,} pts)"
    )
    return wave_nm, flux_broad


# ── arc lamp line list fetcher ───────────────────────────────────────────────

_ARC_SPECIES = {
    "thar": ("Th I;Th II;Ar I;Ar II", "ThAr"),
    "une":  ("U I;U II;Ne I;Ne II",    "UNe"),
}


def _air_to_vac_nm(wave_air_nm: np.ndarray) -> np.ndarray:
    """Morton (2000) air → vacuum wavelength conversion."""
    sigma2 = (1e3 / wave_air_nm) ** 2
    n = 1.0 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigma2) + 2.5540e-4 / (41.0 - sigma2)
    return wave_air_nm * n


def fetch_arc_spectrum(
    lamp_type: str,
    cache_dir: str = "assets/spectral_models",
    wave_min_nm: float = 350.0,
    wave_max_nm: float = 2500.0,
) -> tuple:
    """
    Fetch ThAr or UNe arc-lamp line list from NIST ASD and return
    (wave_nm_vacuum, intensity) arrays of discrete emission lines.

    Lines are cached as .npy files; subsequent calls load from cache.
    NIST returns air wavelengths above 200 nm → converted to vacuum.
    """
    import urllib.parse
    import urllib.request

    lt = lamp_type.lower()
    if lt not in _ARC_SPECIES:
        raise ValueError(f"Unknown lamp type '{lamp_type}'. Choose 'thar' or 'une'.")
    species, label = _ARC_SPECIES[lt]

    os.makedirs(cache_dir, exist_ok=True)
    wave_cache = os.path.join(cache_dir, f"arc_{lt}_wave.npy")
    flux_cache = os.path.join(cache_dir, f"arc_{lt}_flux.npy")

    if os.path.exists(wave_cache) and os.path.exists(flux_cache):
        wave = np.load(wave_cache)
        flux = np.load(flux_cache)
        print(f"  {label} arc loaded from cache: {len(wave)} lines")
        return wave, flux

    print(f"  Fetching {label} lines from NIST ASD …")
    url  = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
    data = urllib.parse.urlencode({
        "spectra"     : species,
        "low_w"       : f"{wave_min_nm * 10.0:.0f}",   # nm → Å (unit=0 = Å)
        "upp_w"       : f"{wave_max_nm * 10.0:.0f}",
        "unit"        : "0",      # 0 = Å
        "format"      : "1",      # ASCII text (in <pre> block)
        "output"      : "0",
        "show_obs_wl" : "1",
        "order_out"   : "0",      # by wavelength
        "show_av"     : "2",      # air wavelengths
        "intens_out"  : "on",
        "allowed_out" : "1",
        "forbid_out"  : "1",
    }).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"User-Agent": "vroomm_simu/1.0"})
    html = urllib.request.urlopen(req, timeout=120).read().decode("utf-8",
                                                                   errors="replace")
    import re as _re
    pre_match = _re.search(r"<pre>(.*?)</pre>", html, _re.DOTALL)
    if not pre_match:
        raise RuntimeError(
            f"NIST ASD returned no data block for {label} — check network or URL.")

    waves, intens = [], []
    for row in pre_match.group(1).split("\n"):
        row = row.strip()
        if not row or row.startswith("-") or "|" not in row:
            continue
        parts = [p.strip() for p in row.split("|")]
        if len(parts) < 3:
            continue
        # Single-species response: wavelength | intensity | ...
        # Multi-species response: species | wavelength | intensity | ...
        # Detect by trying parts[0] as float first, fall back to parts[1].
        try:
            wl_air = float(parts[0])
            ii_raw = parts[1]
        except ValueError:
            try:
                wl_air = float(parts[1])
                ii_raw = parts[2]
            except ValueError:
                continue
        try:
            ii_str = ii_raw.split("/")[0].strip()
            ii_str = "".join(c for c in ii_str if c.isdigit() or c == ".")
            ii = float(ii_str) if ii_str else 0.0
        except ValueError:
            ii = 0.0
        if wl_air > 0 and ii > 0:
            waves.append(wl_air / 10.0)       # Å → nm (still air)
            intens.append(ii)

    if len(waves) < 10:
        raise RuntimeError(
            f"NIST ASD returned only {len(waves)} {label} lines — check network/URL.")

    wave_air = np.array(waves)
    flux     = np.array(intens)
    order    = np.argsort(wave_air)
    wave_air, flux = wave_air[order], flux[order]

    # NIST delivers air wavelengths for λ > 200 nm → convert to vacuum
    wave = _air_to_vac_nm(wave_air)

    np.save(wave_cache, wave)
    np.save(flux_cache, flux)
    print(f"  {label}: {len(wave)} lines fetched and cached (vacuum nm)")
    return wave, flux


# ── parameter file ────────────────────────────────────────────────────────────

DEFAULT_PARAMS = os.path.join(_HERE, "simulate_params.yaml")


def load_params(path: str = DEFAULT_PARAMS) -> dict:
    """
    Load simulate_params.yaml and return a dict ready to be unpacked into
    simulate_detector().  Resolves relative paths against the project root.

    Returns defaults for any key not present in the file.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to read the parameter file.  "
            "Install it with:  pip install pyyaml"
        )

    with open(path) as fh:
        p = yaml.safe_load(fh)

    root = os.path.dirname(os.path.abspath(path))

    def _abs(rel):
        """Make a path absolute relative to the YAML file's directory."""
        return os.path.join(root, rel) if not os.path.isabs(rel) else rel

    # ── resolve paths ──────────────────────────────────────────────────────
    psf_dir      = _abs(p.get("psf_dir",      "assets/zemax_data/images_fibre_rectangulaire"))
    xy_table     = _abs(p.get("xy_table",     "assets/zemax_data/VROOMM_V04_XY.txt"))
    output_fits  = p.get("output_fits",  "detector_sim.fits") or None
    output_png   = p.get("output_png",   "detector_sim.png")  or None
    if output_fits:
        output_fits = _abs(output_fits)
    if output_png:
        output_png  = _abs(output_png)

    # ── sampling ───────────────────────────────────────────────────────────
    # Preferred mode: set wave_step_pix_frac in YAML (fraction of 1 pixel).
    # Backward-compatible mode: if wave_step_pix_frac is absent, use wave_step_nm.
    wave_step_pix_frac = p.get("wave_step_pix_frac", None)
    nm_per_pix_typ = None
    if wave_step_pix_frac is not None:
        wave_step_pix_frac = float(wave_step_pix_frac)
        if wave_step_pix_frac <= 0.0:
            raise ValueError("wave_step_pix_frac must be > 0")
        _xy_for_step = load_xy_table(xy_table)
        nm_per_pix_typ = _estimate_nm_per_pixel(_xy_for_step)
        wave_step_nm = wave_step_pix_frac * nm_per_pix_typ
        print(
            f"  Sampling: wave_step_pix_frac={wave_step_pix_frac:.3f} px "
            f"× {nm_per_pix_typ:.5f} nm/px -> {wave_step_nm:.6f} nm"
        )
    else:
        wave_step_nm = float(p.get("wave_step_nm", 0.01))
        print(f"  Sampling: wave_step_nm={wave_step_nm:.6f} nm (legacy mode)")

    # ── spectrum ───────────────────────────────────────────────────────────
    spectrum_wave = None
    spectrum_flux = None

    mode = p.get("spectrum_mode", "synthetic")
    if mode == "file":
        sf   = p.get("spectrum_file", {})
        spath = _abs(sf["path"])
        wcol  = int(sf.get("wave_col", 0))
        fcol  = int(sf.get("flux_col", 1))
        data  = np.loadtxt(spath, usecols=(wcol, fcol))
        spectrum_wave = data[:, 0]          # kept in nm; converted inside simulate_detector
        spectrum_flux = data[:, 1]
        _needs_photon_calibration = True
        print(f"  Loaded spectrum from {spath}  ({len(spectrum_wave)} points, wavelength in nm)")
    elif mode == "model":
        mc        = p.get("model", {})
        teff      = mc.get("teff", 3000)
        logg      = float(mc.get("logg", 5.0))
        cache_dir = _abs(mc.get("cache_dir", "assets/spectral_models"))
        spectrum_wave, spectrum_flux = fetch_bt_settl_spectrum(
            teff=teff, logg=logg, cache_dir=cache_dir
        )
        print(f"  BT-Settl CIFIST model: Teff={teff} K, logg={logg:.2f}  "
              f"({len(spectrum_wave)} wavelength points)")
        _needs_photon_calibration = True
    elif mode == "synthetic":
        _needs_photon_calibration = False
        # make_spectrum() will be called inside simulate_detector; nothing to do
    else:
        raise ValueError(f"Unknown spectrum_mode '{mode}' in {path}")

    # ── spectrophotometric calibration ────────────────────────────────────────
    tel_conf  = p.get("telescope",   {})
    obs_conf  = p.get("observation", {})
    star_conf = p.get("star",        {})
    diam_m          = float(tel_conf.get("diameter_m",       1.6))
    peak_throughput = float(tel_conf.get("peak_throughput",  0.10))
    exposure_s      = float(obs_conf.get("exposure_s",       1800.0))
    star_mag        = float(star_conf.get("R_mag",           15.0))
    star_mag_band   = str(star_conf.get("mag_band",          "R")).strip().lower()
    A_tel_cm2       = np.pi * (diam_m * 50.0) ** 2   # πr² with r = diam/2 in cm
    # Global wavelength-dependent system transmission T(lambda).
    # Used for stellar, lamp, sky, and flat spectra so blue/red orders are
    # attenuated consistently across all exposure types.
    T_cs            = _get_transmission_spline()

    # Only print stellar/telescope info for science exposures
    _is_science = (
        not bool(p.get("lamp",      {}).get("enabled", False)) and
        not bool(p.get("flatfield", {}).get("enabled", False))
    )
    if _is_science:
        print(
            f"  Telescope: D={diam_m:.2f} m → A_tel={A_tel_cm2:.0f} cm²  "
            f"| peak η={peak_throughput:.1%}  | exp={exposure_s:.0f} s"
        )
        print(f"  Star mag={star_mag:.3f} ({star_mag_band.upper()})")
    else:
        print(
            f"  Telescope: D={diam_m:.2f} m  | exp={exposure_s:.0f} s"
        )

    if _needs_photon_calibration and spectrum_wave is not None:
        # Normalize model/file spectrum to the requested magnitude band.
        # Grp uses fetched Gaia RP passband throughput; R uses box approximation.
        calib_name = "R"
        if star_mag_band in ("grp", "gaia_rp", "rp"):
            try:
                grp_w, grp_t = _load_gaia_grp_passband()
                t_interp = np.interp(spectrum_wave, grp_w, grp_t, left=0.0, right=0.0)
                if np.count_nonzero(t_interp > 0.0) < 2:
                    raise ValueError("Spectrum does not overlap Gaia RP throughput support.")
                model_band_erg = np.trapezoid(spectrum_flux * t_interp, spectrum_wave * 10.0)
                vega_band_erg  = _VEGA_F_LAM_GRP * np.trapezoid(grp_t, grp_w * 10.0)
                erg_scale = vega_band_erg * 10.0 ** (-star_mag / 2.5) / model_band_erg
                calib_name = "GaiaRP"
            except Exception as exc:
                print(f"  Warning: Gaia RP passband load failed ({exc}); falling back to R-box.")
                mask_R      = (spectrum_wave >= _R_BAND_MIN) & (spectrum_wave <= _R_BAND_MAX)
                if mask_R.sum() < 2:
                    raise ValueError(
                        f"Input spectrum does not cover R band ({_R_BAND_MIN}–{_R_BAND_MAX} nm)."
                    )
                model_R_erg = np.trapezoid(spectrum_flux[mask_R], spectrum_wave[mask_R] * 10.0)
                vega_R_erg  = _VEGA_F_LAM_R * (_R_BAND_MAX - _R_BAND_MIN) * 10.0
                erg_scale   = vega_R_erg * 10.0 ** (-star_mag / 2.5) / model_R_erg
        else:
            mask_R      = (spectrum_wave >= _R_BAND_MIN) & (spectrum_wave <= _R_BAND_MAX)
            if mask_R.sum() < 2:
                raise ValueError(
                    f"Input spectrum does not cover R band ({_R_BAND_MIN}–{_R_BAND_MAX} nm)."
                )
            model_R_erg = np.trapezoid(spectrum_flux[mask_R], spectrum_wave[mask_R] * 10.0)
            vega_R_erg  = _VEGA_F_LAM_R * (_R_BAND_MAX - _R_BAND_MIN) * 10.0
            erg_scale   = vega_R_erg * 10.0 ** (-star_mag / 2.5) / model_R_erg
        # Convert erg/cm²/s/Å → photons/nm at detector, for the full exposure:
        #   photons/nm = flux_erg_A × 10 Å/nm × λ_nm/hc × A_tel × T_exp × T(λ)
        spectrum_flux = (
            spectrum_flux * erg_scale * 10.0
            * spectrum_wave / _HC_ERG_NM
            * A_tel_cm2 * exposure_s
            * T_cs(spectrum_wave)
        )
        _m = spectrum_flux[np.isfinite(spectrum_flux) & (spectrum_flux > 0.0)]
        med_all = float(np.nanmedian(_m)) if _m.size else 0.0
        print(
            f"  Spectral calibration ({calib_name}): median = {med_all:.2e} photons/nm "
            f"(wavelength-dependent transmission applied)"
        )

    # ── flat-field (blackbody lamp) ─────────────────────────────────────────────
    # When a fiber is in flat-field mode it receives a calibrated blackbody
    # spectrum.  No vsini broadening, no Doppler shift, no telluric absorption,
    # and no OH sky emission are applied to that fiber.
    ff_conf    = p.get("flatfield", {})
    ff_enabled = bool(ff_conf.get("enabled",   False))
    ff_rect    = ff_enabled and bool(ff_conf.get("rect_fiber", True))
    ff_oct     = ff_enabled and bool(ff_conf.get("oct_fiber",  True))
    ff_tbb_K   = float(ff_conf.get("tbb_K",   5000.0))
    ff_R_mag   = float(ff_conf.get("R_mag",   star_mag))
    _ff_rect   = False   # flipped to True when rect fiber carries a flat lamp

    def _make_flat_spectrum(tbb_K, r_mag, modulation=None):
        """Return (wave_nm, flux_photons_per_nm) for a BB source at the given R mag."""
        fw        = np.linspace(350.0, 2500.0, 100_000)   # nm, dense + smooth
        fr_raw    = _blackbody_nm(fw, tbb_K)
        if modulation is not None:
            fr_raw = fr_raw * modulation(fw)
        mask_R    = (fw >= _R_BAND_MIN) & (fw <= _R_BAND_MAX)
        R_erg     = np.trapezoid(fr_raw[mask_R], fw[mask_R] * 10.0)
        vR_erg    = _VEGA_F_LAM_R * (_R_BAND_MAX - _R_BAND_MIN) * 10.0
        scale     = vR_erg * 10.0 ** (-r_mag / 2.5) / R_erg
        T_cs      = _get_transmission_spline()
        fr        = (fr_raw * scale * 10.0
                     * fw / _HC_ERG_NM
                     * A_tel_cm2 * exposure_s * T_cs(fw))
        return fw, fr

    if ff_rect:
        spectrum_wave, spectrum_flux = _make_flat_spectrum(ff_tbb_K, ff_R_mag)
        _ff_rect = True
        print(f"  Flat field (rect fiber): T_BB={ff_tbb_K:.0f} K, R={ff_R_mag:.1f}  "
              f"(no vsini / Doppler / telluric / sky)")

    # ── vsini rotational broadening ────────────────────────────────────────────
    # Applied in the stellar rest frame, before any Doppler shift.
    # Not applicable when the fiber carries a flat-field lamp.
    vsini_kms = float(star_conf.get("vsini_kms", 0.0))
    if vsini_kms > 0.0 and spectrum_wave is not None and not _ff_rect:
        spectrum_wave, spectrum_flux = _apply_vsini(spectrum_wave, spectrum_flux, vsini_kms)

    # ── sky emission ───────────────────────────────────────────────────────────
    sky_wave  = None
    sky_flux  = None
    sky_conf   = p.get("sky", {})
    sky_scale  = float(sky_conf.get("scale",      1.0))   # fine-tune knob; 1.0 = absolute
    sky_R_mag  = float(sky_conf.get("R_mag",      19.0))  # sky R mag (total through fiber)
    v_fwhm_kms = float(sky_conf.get("v_fwhm_kms",  1.0))  # OH line velocity FWHM [km/s]
    if sky_conf.get("enabled", False):
        sky_cache = _abs(sky_conf.get("cache_dir", "assets/spectral_models"))
        sky_wave, sky_flux = fetch_sky_spectrum(cache_dir=sky_cache)
        # Apply wavelength-dependent throughput line-by-line before absolute
        # normalisation so the detector-frame sky spectrum has the correct
        # chromatic attenuation (especially in the blue orders).
        sky_flux = sky_flux * T_cs(sky_wave)
        # Absolute calibration: sum line intensities in R band → scale to sky_target.
        # Using the direct line sum (not a dense interpolation) so flux is conserved
        # exactly when the lines are subsequently broadened.
        mask_R     = (sky_wave >= _R_BAND_MIN) & (sky_wave <= _R_BAND_MAX)
        sky_R_sum  = sky_flux[mask_R].sum()   # sum of peak-norm intensities in R band
        vega_R_phot_nm = _VEGA_F_LAM_R * 10.0 * _R_BAND_EFF / _HC_ERG_NM   # ph/s/cm²/nm
        vega_R_total   = vega_R_phot_nm * (_R_BAND_MAX - _R_BAND_MIN)        # ph/s/cm²
        sky_target     = (vega_R_total * 10.0**(-sky_R_mag / 2.5)
                          * A_tel_cm2 * exposure_s)
        sky_flux_abs   = sky_flux * (sky_target / sky_R_sum)   # photons per line
        # Broaden each line to a Gaussian of FWHM = v_fwhm_kms / c × λ.
        # Returns a dense grid in photons/nm ready for np.interp in simulate_detector.
        print(f"  Broadening {len(sky_wave)} sky lines to FWHM={v_fwhm_kms:.1f} km/s …")
        sky_wave, sky_flux = _broaden_lines_to_grid(
            sky_wave, sky_flux_abs, v_fwhm_kms=v_fwhm_kms
        )
        print(
            f"  Sky R={sky_R_mag:.1f} (integrated over R): "
            f"≈ {sky_target:.2e} photons in R band  "
            f"(dense grid: {len(sky_wave):,} pts, step=0.0002 nm)"
        )

    # ── target, observatory, BERV, and relativistic Doppler shift ───────────────
    # Parse the new YAML sections for science target and observatory.
    # BERV (barycentric Earth radial velocity) is computed via barycorrpy.
    # Positive BERV = Earth moving toward star = star appears less redshifted.
    # The relativistic Doppler shift brings the stellar spectrum from the rest
    # (barycentric) frame to the topocentric frame BEFORE applying telluric.
    tgt_conf        = p.get("target",      {})
    observ_conf     = p.get("observatory", {})
    jd_utc          = float(obs_conf.get("jd_utc", 0.0))

    tgt_name        = str(tgt_conf.get("name",         "Unknown"))
    tgt_ra_deg      = float(tgt_conf.get("ra_deg",       0.0))
    tgt_dec_deg     = float(tgt_conf.get("dec_deg",      0.0))
    tgt_pmra_masyr  = float(tgt_conf.get("pmra_masyr",   0.0))
    tgt_pmdec_masyr = float(tgt_conf.get("pmdec_masyr",  0.0))
    tgt_px_mas      = float(tgt_conf.get("px_mas",       0.0))
    rv_sys_kms      = float(tgt_conf.get("rv_sys_kms",   0.0))

    obs_name  = str(observ_conf.get("name",        ""))
    obs_lat   = float(observ_conf.get("lat_deg",     0.0))
    obs_lon   = float(observ_conf.get("lon_deg",     0.0))
    obs_elev  = float(observ_conf.get("elevation_m", 0.0))

    # ── BERV via barycorrpy ─────────────────────────────────────────────────
    berv_kms = 0.0
    if _is_science and jd_utc > 0.0:
        try:
            from barycorrpy import get_BC_vel
            # Returns (BERV [m/s], warning, status).  BERV[0] is the scalar result.
            berv_ms, _, _ = get_BC_vel(
                JDUTC = jd_utc,
                ra    = tgt_ra_deg,
                dec   = tgt_dec_deg,
                lat   = obs_lat,
                longi = obs_lon,
                alt   = obs_elev,
                epoch = 2451545.0,       # J2000.0 FK5
                pmra  = tgt_pmra_masyr,
                pmdec = tgt_pmdec_masyr,
                px    = tgt_px_mas,
            )
            berv_kms = float(berv_ms[0]) / 1000.0   # m/s → km/s
            print(f"  BERV = {berv_kms:+.4f} km/s  (barycorrpy)")
        except ImportError:
            print("  [warn] barycorrpy not installed; BERV set to 0 km/s")
        except Exception as exc:
            print(f"  [warn] barycorrpy failed ({exc!r}); BERV set to 0 km/s")

    # ── relativistic Doppler shift (rest frame → topocentric) ───────────────
    # Sign convention:
    #   rv_sys_kms > 0  star receding from barycenter (redshift)
    #   berv_kms   > 0  Earth moving toward star (blueshift; reduces apparent recession)
    # Relativistic velocity addition:
    #   beta_topo = (beta_sys - beta_BERV) / (1 - beta_sys * beta_BERV)
    # Doppler factor:
    #   D = sqrt((1 + beta_topo) / (1 - beta_topo))
    #   lambda_obs = lambda_rest * D
    _C_KMS     = 2.998e5                          # speed of light [km/s]
    beta_sys   = rv_sys_kms / _C_KMS
    beta_berv  = berv_kms   / _C_KMS
    beta_topo  = (beta_sys - beta_berv) / (1.0 - beta_sys * beta_berv)
    doppler_D  = np.sqrt((1.0 + beta_topo) / (1.0 - beta_topo))
    rv_topo_kms = beta_topo * _C_KMS             # topocentric RV [km/s]

    if _is_science:
        print(
            f"  Radial velocity:  v_sys = {rv_sys_kms:+.3f} km/s,  "
            f"BERV = {berv_kms:+.4f} km/s,  "
            f"v_topo = {rv_topo_kms:+.4f} km/s  (D = {doppler_D:.10f})"
        )

    if spectrum_wave is not None and not _ff_rect:
        # Shift stellar spectrum from rest frame to topocentric wavelength frame.
        # Telluric transmission is defined in the topocentric frame, so the RV
        # shift must happen BEFORE the telluric multiplication below.
        spectrum_wave = spectrum_wave * doppler_D
        # Flux density correction: bandwidth dilation dλ_obs = dλ_rest × D,
        # so photons/nm_rest → photons/nm_obs requires dividing by D.
        spectrum_flux = spectrum_flux / doppler_D

    # ── telluric absorption ────────────────────────────────────────────────────
    tell_conf = p.get("telluric", {})
    if tell_conf.get("enabled", False) and spectrum_wave is not None and not _ff_rect:
        tell_fits   = _abs(tell_conf.get("fits_path", "assets/LaSilla_tapas.fits"))
        tell_am     = float(tell_conf.get("airmass",  1.2))
        tell_wod    = float(tell_conf.get("water_od", 1.5))
        tell_cache  = _abs(tell_conf.get("cache_dir", "assets/spectral_models"))
        tell_wave, tell_trans = load_tapas_telluric(
            fits_path   = tell_fits,
            airmass     = tell_am,
            water_od    = tell_wod,
            wave_min_nm = float(spectrum_wave.min()),
            wave_max_nm = float(spectrum_wave.max()),
            cache_dir   = tell_cache,
        )
        # Interpolate telluric onto the stellar spectrum wavelength grid and apply
        tell_interp   = np.interp(spectrum_wave, tell_wave, tell_trans,
                                  left=1.0, right=1.0)
        spectrum_flux = spectrum_flux * tell_interp
        print(
            f"  Telluric: airmass={tell_am:.2f}, water_od={tell_wod:.2f}  "
            f"(mean transmission {tell_interp.mean():.3f})"
        )

    # ── arc lamp ──────────────────────────────────────────────────────────────
    lamp_conf    = p.get("lamp", {})
    lamp_enabled = bool(lamp_conf.get("enabled",     False))
    lamp_rect    = lamp_enabled and bool(lamp_conf.get("rect_fiber", False))
    lamp_oct     = lamp_enabled and bool(lamp_conf.get("oct_fiber",  False))
    lamp_type    = str(lamp_conf.get("type",         "thar")).lower()
    lamp_R_mag   = float(lamp_conf.get("R_mag",      10.0))
    lamp_fwhm    = float(lamp_conf.get("v_fwhm_kms", 0.3))
    lamp_fp_cavity_cm = float(lamp_conf.get("fp_cavity_cm", 1.0))
    lamp_fp_finesse   = float(lamp_conf.get("fp_finesse", 10.0))
    lamp_fp_tbb_K     = float(lamp_conf.get("fp_tbb_K", ff_tbb_K))
    lamp_cache   = _abs(lamp_conf.get("cache_dir",   "assets/spectral_models"))
    _lamp_rect   = False
    _lamp_oct_wave = _lamp_oct_flux = None

    def _make_lamp_spectrum(ltype, r_mag, fwhm_kms, cache_d):
        if ltype == "fp":
            # Build FP comb on a cavity-aware high-resolution grid.
            # A 1 cm cavity has FSR of order 0.01-0.04 nm in the optical,
            # so the coarse 100k-point blackbody grid is too sparse and aliases
            # peaks. We explicitly oversample the Airy pattern here.
            wmin, wmax = 350.0, 930.0
            L_nm = max(lamp_fp_cavity_cm, 1e-6) * 1.0e7
            fsr_min_nm = (wmin ** 2) / (2.0 * L_nm)
            step_nm = min(0.0010, 0.20 * fsr_min_nm)
            step_nm = max(step_nm, 1.0e-4)
            fw = np.arange(wmin, wmax + step_nm, step_nm)

            fp_mod = _fp_airy_transmission(fw, lamp_fp_cavity_cm, lamp_fp_finesse)
            fr_raw = _blackbody_nm(fw, lamp_fp_tbb_K) * fp_mod

            mask_R = (fw >= _R_BAND_MIN) & (fw <= _R_BAND_MAX)
            R_erg  = np.trapezoid(fr_raw[mask_R], fw[mask_R] * 10.0)
            vR_erg = _VEGA_F_LAM_R * (_R_BAND_MAX - _R_BAND_MIN) * 10.0
            scale  = vR_erg * 10.0 ** (-r_mag / 2.5) / max(R_erg, 1e-30)
            ff = (fr_raw * scale * 10.0
                  * fw / _HC_ERG_NM
                  * A_tel_cm2 * exposure_s * T_cs(fw))

            fsr_ref_nm = (700.0 ** 2) / (2.0 * L_nm)
            if nm_per_pix_typ is not None and np.isfinite(nm_per_pix_typ) and nm_per_pix_typ > 0:
                pix_ref = fsr_ref_nm / nm_per_pix_typ
                print(
                    f"  FP comb spacing near 700 nm: {fsr_ref_nm:.5f} nm "
                    f"(~{pix_ref:.1f} px with {nm_per_pix_typ:.5f} nm/px)"
                )
            print(
                f"  FP comb model: cavity={lamp_fp_cavity_cm:.3f} cm, "
                f"finesse={lamp_fp_finesse:.2f}, T_BB={lamp_fp_tbb_K:.0f} K, "
                f"grid step={step_nm:.6f} nm"
            )
            return fw, ff

        lw, lf_raw = fetch_arc_spectrum(ltype, cache_d)
        # Apply system throughput to lamp line intensities so blue orders are
        # dimmed consistently with science/flat exposures.
        lf_raw = lf_raw * T_cs(lw)
        mask_R     = (lw >= _R_BAND_MIN) & (lw <= _R_BAND_MAX)
        l_R_sum    = lf_raw[mask_R].sum()
        vega_R_phot_nm = _VEGA_F_LAM_R * 10.0 * _R_BAND_EFF / _HC_ERG_NM
        vega_R_total   = vega_R_phot_nm * (_R_BAND_MAX - _R_BAND_MIN)
        lamp_target    = (vega_R_total * 10.0**(-r_mag / 2.5)
                          * A_tel_cm2 * exposure_s)
        lf_abs = lf_raw * (lamp_target / l_R_sum)
        print(f"  Broadening {len(lw)} {ltype.upper()} lines "
              f"to FWHM={fwhm_kms:.2f} km/s …")
        return _broaden_lines_to_grid(lw, lf_abs, v_fwhm_kms=fwhm_kms)

    if lamp_rect:
        spectrum_wave, spectrum_flux = _make_lamp_spectrum(
            lamp_type, lamp_R_mag, lamp_fwhm, lamp_cache)
        _lamp_rect = True
        print(f"  Arc lamp (rect fiber): {lamp_type.upper()}, R={lamp_R_mag:.1f}  "
              f"(no vsini / Doppler / telluric / sky)")
    if lamp_oct:
        _lamp_oct_wave, _lamp_oct_flux = _make_lamp_spectrum(
            lamp_type, lamp_R_mag, lamp_fwhm, lamp_cache)
        print(f"  Arc lamp (oct fiber): {lamp_type.upper()}, R={lamp_R_mag:.1f}  "
              f"(no sky)")

    # Combined calibration flag — suppresses sky on rect fiber
    _skip_rect_cal = _ff_rect or _lamp_rect
    if _skip_rect_cal:
        sky_wave = sky_flux = None

    # ── octagonal fiber ────────────────────────────────────────────────────────
    oct_raw  = p.get("octagonal_fiber", {})
    oct_conf = None
    if oct_raw.get("enabled", False):
        oct_conf = dict(
            psf_dir       = _abs(oct_raw.get("psf_dir", PSF_DIR_OCT)),
            y_offset_pix  = float(oct_raw.get("y_offset_pix", -10)),
            sky_scale     = float(oct_raw.get("sky_scale", sky_scale)),
            sky_wave      = sky_wave,    # oct always gets sky unless in cal mode
            sky_flux      = sky_flux,
            spectrum_wave = None,
            spectrum_flux = None,
        )
        if ff_oct:
            oct_ff_w, oct_ff_f = _make_flat_spectrum(ff_tbb_K, ff_R_mag)
            oct_conf["spectrum_wave"] = oct_ff_w
            oct_conf["spectrum_flux"] = oct_ff_f
            oct_conf["sky_wave"]      = None
            oct_conf["sky_flux"]      = None
            print(f"  Flat field (oct fiber): T_BB={ff_tbb_K:.0f} K, "
                  f"R={ff_R_mag:.1f}")
        if lamp_oct:
            oct_conf["spectrum_wave"] = _lamp_oct_wave
            oct_conf["spectrum_flux"] = _lamp_oct_flux
            oct_conf["sky_wave"]      = None
            oct_conf["sky_flux"]      = None
        print(
            f"  Octagonal fiber: offset={oct_conf['y_offset_pix']:+.0f} px, "
            f"sky_scale={oct_conf['sky_scale']:.3g}"
        )

    # ── science fiber routing ─────────────────────────────────────────────────
    # science_fiber controls which fiber carries the stellar spectrum.
    # The OTHER fiber receives sky only (spectrum_wave/flux = None).
    # Both fibers always receive sky emission.
    science_fiber = p.get("science_fiber", "rect")
    if oct_conf is not None and science_fiber == "oct" and spectrum_wave is not None:
        # Move science spectrum from rect to oct
        oct_conf["spectrum_wave"] = spectrum_wave
        oct_conf["spectrum_flux"] = spectrum_flux
        spectrum_wave = None
        spectrum_flux = None
        # sky stays on both (oct already has it; rect retains its sky_wave/sky_flux)
        print("  Science spectrum routed to octagonal fiber; "
              "rectangular fiber carries sky only")
    elif oct_conf is not None and science_fiber == "rect" and spectrum_wave is not None:
        print("  Science spectrum routed to rectangular fiber; "
              "octagonal fiber carries sky only")

    # ── FITS metadata ─────────────────────────────────────────────────────────
    # Build a {keyword: (value, comment)} dict of all simulation input parameters
    # for full provenance in the output FITS primary header.
    from datetime import datetime, timezone
    if jd_utc > 0.0:
        _unix_s  = (jd_utc - 2440587.5) * 86400.0   # 2440587.5 = JD of Unix epoch
        date_obs = datetime.fromtimestamp(_unix_s, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    else:
        date_obs = ""

    _mc_p = p.get("model", {})
    fits_meta = {
        # ── instrument ────────────────────────────────────────────────────────
        "INSTRUME": ("VROOMM",             "spectrograph name"),
        "TELESCOP": (obs_name or "Unknown",  "observatory / telescope name"),
        "TELDIA"  : (diam_m,               "primary mirror diameter [m]"),
        "PEAKETA" : (peak_throughput,       "peak throughput, fiber to detector"),
        "PIXSCALE": (PIXEL_SIZE_MM * 1e3,  "detector pixel size [um]"),
        "PSFPIX"  : (PSF_PIXEL_SIZE * 1e3, "PSF native pixel size [um]"),
        "BINFACT" : (BIN_FACTOR,           "PSF binning factor (PSF px / det px)"),
        "DETSIZE" : (DETECTOR_SIZE,        "detector side length [pixels]"),
        "BUNIT"   : ("photon",             "pixel value in detected photons"),
        # ── observation ───────────────────────────────────────────────────────
        "DATE-OBS": (date_obs,             "UTC date/time at observation midpoint"),
        "JD-MID"  : (jd_utc,              "Julian Date (UTC) at midpoint"),
        "EXPTIME" : (exposure_s,           "exposure time [s]"),
        # ── observatory ───────────────────────────────────────────────────────
        "OBSERVAT": (obs_name or "Unknown",  "observatory name"),
        "OBSLAT"  : (obs_lat,              "observatory geodetic latitude [deg N]"),
        "OBSLON"  : (obs_lon,              "observatory longitude [deg E; neg=W]"),
        "OBSALT"  : (obs_elev,             "observatory altitude above sea level [m]"),
        # ── science target ────────────────────────────────────────────────────
        "OBJECT"  : (tgt_name,             "science target name"),
        "RA"      : (tgt_ra_deg,           "target J2000 right ascension [deg]"),
        "DEC"     : (tgt_dec_deg,          "target J2000 declination [deg]"),
        "PMRA"    : (tgt_pmra_masyr,       "proper motion mu_alpha*cos(dec) [mas/yr]"),
        "PMDEC"   : (tgt_pmdec_masyr,      "proper motion mu_delta [mas/yr]"),
        "PARALLAX": (tgt_px_mas,           "trigonometric parallax [mas]"),
        "VSINI"   : (vsini_kms,             "projected rotation velocity [km/s]"),
        "RVSYS"   : (rv_sys_kms,           "systemic RV, barycentric frame [km/s]"),
        "BERV"    : (berv_kms,             "barycentric RV correction (barycorrpy) [km/s]"),
        "RVTOPO"  : (rv_topo_kms,          "topocentric RV = relativ.(RVSYS,BERV) [km/s]"),
        "DOPPFACT": (doppler_D,            "Doppler D; lambda_obs = D * lambda_rest"),
        # ── photometry / spectrum model ────────────────────────────────────────
        "RMAG"    : (star_mag,              "science target magnitude scalar"),
        "MAGBAND" : (star_mag_band.upper(), "magnitude band used for normalization"),
        "SPECMODE": (mode,                  "spectrum_mode: synthetic|file|model"),
        "TEFF"    : (float(_mc_p.get("teff",  0.0)), "BT-Settl model Teff [K]"),
        "LOGG"    : (float(_mc_p.get("logg",  0.0)), "BT-Settl model log g [cm/s2]"),
        # ── sky ───────────────────────────────────────────────────────────────
        "SKYENB"  : (bool(sky_conf.get("enabled", False)), "sky emission enabled"),
        "SKYRMAG" : (sky_R_mag,             "sky R-band magnitude through fiber [mag]"),
        "SKYOFWHM": (v_fwhm_kms,            "OH sky line velocity FWHM [km/s]"),
        # ── telluric ──────────────────────────────────────────────────────────
        "TELLENB" : (bool(tell_conf.get("enabled", False)), "telluric absorption enabled"),
        "TELLAIRT": (float(tell_conf.get("airmass",  0.0)), "telluric reference airmass"),
        "TELLWATD": (float(tell_conf.get("water_od", 0.0)), "H2O optical depth scaling factor"),
        "TELLFILE": (str(tell_conf.get("fits_path", "")),   "TAPAS transmission FITS file"),
        # ── fibers / PSF ──────────────────────────────────────────────────────
        "BLAZEON" : (bool(p.get("blaze", True)), "blaze function applied"),
        "WAVESTEP": (wave_step_nm,           "wavelength sampling step [nm]"),
        "WSTEPPX" : (float(wave_step_pix_frac) if wave_step_pix_frac is not None else -1.0,
                 "sampling step [pixel fraction]"),
        "NMPERPIX": (float(nm_per_pix_typ) if nm_per_pix_typ is not None else -1.0,
                 "typical local dispersion [nm/pix]"),
        "PSFSRC"  : (str(psf_dir),           "rectangular fiber PSF directory"),
        "XYTABLE" : (str(xy_table),          "order X/Y wavelength table"),
        "OCTENB"  : (bool(oct_raw.get("enabled", False)), "octagonal fiber simulated"),
        "OCTOFF"  : (float(oct_raw.get("y_offset_pix", 0.0)), "octagonal fiber Y offset [pix]"),
        "OCTDIR"  : (str(oct_raw.get("psf_dir", "")),     "octagonal fiber PSF directory"),
        # ── calibration modes ─────────────────────────────────────────────────
        "FFENB"   : (ff_enabled,                            "flat-field mode enabled"),
        "FFRECT"  : (ff_rect,                               "flat-field in rectangular fiber"),
        "FFOCT"   : (ff_oct,                                "flat-field in octagonal fiber"),
        "FFTBB"   : (ff_tbb_K,                              "flat-field blackbody temperature [K]"),
        "LAMPENB" : (lamp_enabled,                          "arc lamp mode enabled"),
        "LAMPRECT": (lamp_rect,                             "arc lamp in rectangular fiber"),
        "LAMPOCT" : (lamp_oct,                              "arc lamp in octagonal fiber"),
        "LAMPTYPE": (lamp_type if lamp_enabled else "",     "arc lamp type: thar or une"),
    }

    return dict(
        psf_dir=psf_dir,
        xy_path=xy_table,
        output_fits=output_fits,
        output_png=output_png,
        wave_step_nm=wave_step_nm,
        wave_step_pix_frac=wave_step_pix_frac,
        spectrum_wave=spectrum_wave,
        spectrum_flux=spectrum_flux,
        blaze=bool(p.get("blaze", True)),
        sky_wave=sky_wave,
        sky_flux=sky_flux,
        sky_scale=sky_scale,
        oct_conf=oct_conf,
        fits_meta=fits_meta,
    )


# ── BT-Settl CIFIST model fetcher ─────────────────────────────────────────────

_SVO_BASE    = "http://svo2.cab.inta-csic.es/theory/newov2/"
_SVO_SEARCH  = _SVO_BASE + "index.php?"
_SVO_FETCH   = _SVO_BASE + "ssap.php"
_SVO_HEADERS = {"User-Agent": "vroomm_simu/1.0 (contact via GitHub)"}


def _svo_grid_index() -> list:
    """
    Fetch the full BT-Settl CIFIST grid index from SVO.

    Returns
    -------
    list of (teff_int, logg_float, fid_int)
    """
    data = urllib.parse.urlencode({
        "models": ",bt-settl-cifist",
        "boton":  "Search",
    }).encode()
    req  = urllib.request.Request(_SVO_SEARCH, data=data, headers=_SVO_HEADERS)
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8",
                                                                  errors="replace")

    row_re  = re.compile(r"<tr>\s*(<td.*?)</tr>", re.S | re.I)
    fid_re  = re.compile(r"fid=(\d+)")
    cell_re = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
    strip   = lambda s: re.sub(r"<[^>]+>", "", s).strip()

    entries = []
    for m in row_re.finditer(html):
        row  = m.group(1)
        fids = fid_re.findall(row)
        if not fids:
            continue
        cells = [strip(c) for c in cell_re.findall(row)]
        # cells layout: [ModelName, Teff, Logg, FeH, Alpha, …]
        # If cells[0] is non-numeric it is the model name; otherwise skip it.
        try:
            if cells and not cells[0].replace(".", "", 1).lstrip("-").isdigit():
                teff_s, logg_s = cells[1], cells[2]
            else:
                teff_s, logg_s = cells[0], cells[1]
            entries.append((int(float(teff_s)), float(logg_s), int(fids[0])))
        except (ValueError, IndexError):
            continue
    return entries


def fetch_bt_settl_spectrum(
    teff: int | float,
    logg: float,
    cache_dir: str = "assets/spectral_models",
) -> tuple:
    """
    Return (wave_nm, flux) for the nearest BT-Settl CIFIST grid point.

    Spectra are cached in `cache_dir` as plain-text two-column files
    (wavelength_nm  flux_erg_cm2_s_A).  Subsequent calls load from cache
    without hitting the network.

    Parameters
    ----------
    teff : int or float  — target effective temperature [K]
    logg : float         — target log surface gravity [log(cm/s²)]
    cache_dir : str      — local cache directory

    Returns
    -------
    wave_nm : np.ndarray — wavelength in nm
    flux    : np.ndarray — flux in erg/cm²/s/Å
    """
    teff = int(round(teff))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"bt-settl-cifist_teff{teff:05d}_logg{logg:.2f}.dat"
    )

    if os.path.exists(cache_path):
        print(f"  Model spectrum loaded from cache: {cache_path}")
        data = np.loadtxt(cache_path)
        return data[:, 0], data[:, 1]

    print("  Fetching BT-Settl CIFIST grid index from SVO …")
    entries = _svo_grid_index()
    if not entries:
        raise RuntimeError(
            "Could not parse any grid entries from the SVO BT-Settl CIFIST page. "
            "Check network connectivity or the SVO URL."
        )

    # Nearest-neighbour: normalise by typical grid steps so a 100 K Teff
    # mismatch is equivalent to a 0.5 dex logg mismatch.
    best_fid = best_teff_found = best_logg_found = None
    best_d2  = float("inf")
    for (t, g, fid) in entries:
        d2 = ((t - teff) / 100.0) ** 2 + ((g - logg) / 0.5) ** 2
        if d2 < best_d2:
            best_d2, best_fid = d2, fid
            best_teff_found, best_logg_found = t, g

    print(
        f"  Nearest grid point: Teff={best_teff_found} K, "
        f"logg={best_logg_found:.2f}  (fid={best_fid})  "
        f"[requested Teff={teff} K, logg={logg:.2f}]"
    )

    url = f"{_SVO_FETCH}?model=bt-settl-cifist&fid={best_fid}&format=ascii"
    print(f"  Downloading {url} …")
    req = urllib.request.Request(url, headers=_SVO_HEADERS)
    raw = urllib.request.urlopen(req, timeout=120).read().decode(
        "utf-8", errors="replace"
    )

    waves, fluxes = [], []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            waves.append(float(parts[0]))
            fluxes.append(float(parts[1]))
        except ValueError:
            continue

    wave_nm = np.array(waves,  dtype=np.float64) / 10.0   # Å → nm
    flux    = np.array(fluxes, dtype=np.float64)

    np.savetxt(cache_path,
               np.column_stack([wave_nm, flux]),
               header="wavelength_nm  flux_erg_cm2_s_A",
               fmt="%.6f  %.6e")
    print(f"  Cached to {cache_path}  ({len(wave_nm)} points)")
    return wave_nm, flux


# ── OH airglow sky emission fetcher ───────────────────────────────────────────

_EXOMOL_OH_STATES = (
    "https://www.exomol.com/db/OH/16O-1H/MoLLIST-OH/"
    "16O-1H__MoLLIST-OH.states.bz2"
)
_EXOMOL_OH_TRANS = (
    "https://www.exomol.com/db/OH/16O-1H/MoLLIST-OH/"
    "16O-1H__MoLLIST-OH.trans.bz2"
)

# Prominent atomic sky emission lines (wavelength_nm, relative_intensity).
# Intensities are on the same scale as the OH lines after Boltzmann weighting
# (O I 5577 ≈ 100 is the bright green line; others scaled to typical ratios
# from Osterbrock & Martel 1992, Hanuschik 2003).
_ATOMIC_SKY_LINES = [
    (557.7338, 100.0),   # O I   (green airglow, very bright)
    (589.000,   15.0),   # Na I  D2
    (589.592,   10.0),   # Na I  D1
    (630.030,   30.0),   # O I
    (636.378,   10.0),   # O I
    (777.417,    5.0),   # O I   triplet
    (777.539,    5.0),   # O I   triplet
    (777.675,    5.0),   # O I   triplet
    (844.625,    3.0),   # O I   triplet
    (844.636,    3.0),   # O I   triplet
    (844.676,    3.0),   # O I   triplet
]

# OH mesospheric rotational temperature [K] and k_B in cm⁻¹/K
_T_OH    = 200.0
_KB_CM1  = 0.6950356


def fetch_sky_spectrum(
    cache_dir: str = "assets/spectral_models",
    wave_min_nm: float = 400.0,
    wave_max_nm: float = 910.0,
) -> tuple:
    """
    Download and cache the OH Meinel airglow line list from ExoMol (MoLLIST-OH)
    plus prominent atomic sky emission lines.

    Lines are weighted by the Boltzmann factor at the OH mesospheric rotational
    temperature (~200 K) so relative intensities are physically motivated.
    The returned flux array is normalised to its peak value.

    Sources
    -------
    - Brooke et al. 2016, JQSRT 138, 142  (OH line strengths, X²Π state)
    - Bernath 2020, JQSRT 240, 106687     (MoLLIST compilation)
    - ExoMol MoLLIST-OH  (https://www.exomol.com/data/molecules/OH/16O-1H/MoLLIST-OH/)
    - Atomic lines: Osterbrock & Martel 1992, Hanuschik 2003

    Parameters
    ----------
    cache_dir    : directory for cached data files
    wave_min_nm  : short-wavelength cutoff [nm]
    wave_max_nm  : long-wavelength cutoff [nm]

    Returns
    -------
    wave_nm : np.ndarray   wavelength grid [nm]
    flux    : np.ndarray   relative flux, peak-normalised to 1
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"sky_OH_airglow_{int(wave_min_nm)}-{int(wave_max_nm)}nm.dat",
    )

    if os.path.exists(cache_path):
        print(f"  Sky spectrum loaded from cache: {cache_path}")
        data = np.loadtxt(cache_path)
        return data[:, 0], data[:, 1]

    # ── States file ────────────────────────────────────────────────────────────
    print("  Downloading ExoMol MoLLIST-OH states …")
    req  = urllib.request.Request(_EXOMOL_OH_STATES, headers=_SVO_HEADERS)
    raw  = urllib.request.urlopen(req, timeout=120).read()
    txt  = bz2.decompress(raw).decode(errors="replace")

    # columns: id  E(cm⁻¹)  g  J  e/f  v  …
    states = {}   # id → (E_cm1, g)
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            states[int(parts[0])] = (float(parts[1]), int(parts[2]))
        except ValueError:
            continue
    print(f"    {len(states)} energy levels loaded")

    # ── Transitions file ───────────────────────────────────────────────────────
    print("  Downloading ExoMol MoLLIST-OH transitions …")
    req  = urllib.request.Request(_EXOMOL_OH_TRANS, headers=_SVO_HEADERS)
    raw  = urllib.request.urlopen(req, timeout=120).read()
    txt  = bz2.decompress(raw).decode(errors="replace")

    wn_min = 1e7 / wave_max_nm   # cm⁻¹  (900 nm → 11 111 cm⁻¹)
    wn_max = 1e7 / wave_min_nm   # cm⁻¹  (400 nm → 25 000 cm⁻¹)

    waves, fluxes = [], []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            uid = int(parts[0])
            A   = float(parts[2])
            wn  = float(parts[3])
        except ValueError:
            continue
        if not (wn_min <= wn <= wn_max):
            continue
        if uid not in states:
            continue
        E_u, g_u = states[uid]
        intensity = A * g_u * np.exp(-E_u / (_KB_CM1 * _T_OH))
        if intensity > 0.0:
            waves.append(1e7 / wn)   # cm⁻¹ → nm
            fluxes.append(intensity)

    print(f"    {len(waves)} OH transitions in {wave_min_nm:.0f}–{wave_max_nm:.0f} nm")

    # ── Atomic sky lines ───────────────────────────────────────────────────────
    oh_max = max(fluxes) if fluxes else 1.0
    for wl, rel in _ATOMIC_SKY_LINES:
        if wave_min_nm <= wl <= wave_max_nm:
            waves.append(wl)
            fluxes.append(rel * oh_max)

    # ── Sort, normalise, cache ─────────────────────────────────────────────────
    wave_arr = np.array(waves,  dtype=np.float64)
    flux_arr = np.array(fluxes, dtype=np.float64)
    idx      = np.argsort(wave_arr)
    wave_arr = wave_arr[idx]
    flux_arr = flux_arr[idx]
    flux_arr /= flux_arr.max()

    np.savetxt(
        cache_path,
        np.column_stack([wave_arr, flux_arr]),
        header="wavelength_nm  relative_flux_peak_norm",
        fmt="%.6f  %.6e",
    )
    print(f"  Sky spectrum cached: {cache_path}  ({len(wave_arr)} lines)")
    return wave_arr, flux_arr


def load_tapas_telluric(
    fits_path: str,
    airmass: float,
    water_od: float,
    wave_min_nm: float,
    wave_max_nm: float,
    cache_dir: str = "assets/spectral_models",
) -> tuple:
    """
    Load the TAPAS per-species transmission FITS file and return a combined
    telluric transmission spectrum over [wave_min_nm, wave_max_nm].

    Scaling rules (Beer-Lambert, reference at airmass=1):
      - Dry absorbers (CO2, CH4, O2, O3, N2O, NO2):
            T_dry(λ) = [T_CO2 × T_CH4 × T_O2 × T_O3 × T_N2O × T_NO2] ^ airmass
      - Water vapour:
            T_H2O(λ) = T_H2O_ref(λ) ^ water_od
        where water_od = airmass × (PWV / PWV_ref).
        Useful range: 0.5 (dry/low airmass) → 5 (wet/high airmass).

    The per-species arrays are cached as .npy files in cache_dir to avoid
    re-reading the large FITS on every run.

    Returns
    -------
    wave_nm : np.ndarray
        Wavelength array [nm], trimmed to [wave_min_nm, wave_max_nm].
    transmission : np.ndarray
        Combined telluric transmission (0–1).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_wave   = os.path.join(cache_dir, "tapas_wave_nm.npy")
    cache_species = os.path.join(cache_dir, "tapas_species.npy")

    if os.path.exists(cache_wave) and os.path.exists(cache_species):
        wave_nm  = np.load(cache_wave)
        species  = np.load(cache_species)   # shape (N, 6): CO2 CH4 O2 O3 N2O NO2
        h2o      = np.load(os.path.join(cache_dir, "tapas_h2o.npy"))
    else:
        print(f"  Loading TAPAS file: {fits_path} …")
        from astropy.io import fits as _fits
        with _fits.open(fits_path) as hdul:
            t = hdul[1].data
            w = t["wavelength"].astype(np.float64)
            mask = (w >= wave_min_nm) & (w <= wave_max_nm)
            wave_nm = w[mask]
            h2o     = t["H2O"][mask].astype(np.float64)
            dry     = np.column_stack([
                t["CO2"][mask].astype(np.float64),
                t["CH4"][mask].astype(np.float64),
                t["O2" ][mask].astype(np.float64),
                t["O3" ][mask].astype(np.float64),
                t["N2O"][mask].astype(np.float64),
                t["NO2"][mask].astype(np.float64),
            ])
        np.save(cache_wave,                    wave_nm)
        np.save(cache_species,                 dry)
        np.save(os.path.join(cache_dir, "tapas_h2o.npy"), h2o)
        print(f"  TAPAS cache written ({len(wave_nm):,} pts, "
              f"{wave_nm[0]:.1f}–{wave_nm[-1]:.1f} nm)")
        species = dry

    # Clip to [0, 1] to guard against minor numerical noise in the FITS
    h2o     = np.clip(h2o,     0.0, 1.0)
    species = np.clip(species, 0.0, 1.0)

    # T_dry = product of all dry absorbers raised to the airmass power
    t_dry = np.prod(species, axis=1) ** airmass
    # T_H2O = reference H2O transmission raised to the water optical depth
    t_h2o = h2o ** water_od

    transmission = np.clip(t_dry * t_h2o, 0.0, 1.0)
    return wave_nm, transmission


def simulate_detector(
    psf_dir: str = PSF_DIR_RECT,
    xy_path: str = XY_TABLE,
    output_fits: str | None = "detector_sim.fits",
    output_png: str | None = "detector_sim.png",
    wave_step_nm: float = 0.01,
    wave_step_pix_frac: float | None = None,
    spectrum_wave: np.ndarray | None = None,
    spectrum_flux: np.ndarray | None = None,
    blaze: bool = True,
    sky_wave: np.ndarray | None = None,
    sky_flux: np.ndarray | None = None,
    sky_scale: float = 0.0,
    y_offset_pix: float = 0.0,
) -> np.ndarray:
    """
    Build a 4096×4096 detector image by depositing interpolated PSF stamps
    weighted by the input spectrum.

    The simulation walks each order on a dense wavelength grid.  Sampling can
    be fixed in nm (`wave_step_nm`) or adaptive from a pixel-fraction
    (`wave_step_pix_frac`) converted to nm using that order's local dispersion.
    At every step the detector position and PSF are linearly
    interpolated from the Zemax reference points, and the flux-weighted PSF
    stamp is accumulated into the output image.

    Parameters
    ----------
    psf_dir : str
        Directory with PSF .txt files.
    xy_path : str
        Path to the order/wavelength/position table.
    output_fits : str or None
        If given, save the image array to this FITS file.
    output_png : str or None
        If given, save a log-scale preview PNG.
    wave_step_nm : float
        Wavelength sampling step in nm (default 0.01 nm).  Finer steps give
        smoother lines but increase runtime linearly.
    wave_step_pix_frac : float or None
        If provided, overrides the fixed nm step and uses an adaptive step per
        order:  step_nm(order) = wave_step_pix_frac × nm_per_pix(order), where
        nm_per_pix(order) is estimated from consecutive XY samples.
    spectrum_wave : array_like of float, optional
        Wavelength grid in **nm** for a custom input spectrum.  Must be sorted
        ascending.  If None, make_spectrum() is used.
    spectrum_flux : array_like of float, optional
        Flux values corresponding to spectrum_wave.  Required when
        spectrum_wave is provided.
    blaze : bool
        If True (default), multiply each order's flux by the grating blaze
        envelope sinc²(π·(wavelength − λ_central) / Δλ_FSR).
    sky_wave : array_like of float, optional
        Wavelength grid [nm] for the sky emission spectrum.
    sky_flux : array_like of float, optional
        Relative flux of the sky (must be peak-normalised to 1).
    sky_scale : float
        Sky peak intensity relative to the stellar median continuum.
        0 = no sky, 1 = sky peak equals stellar median, etc.
    y_offset_pix : float
        Shift the entire trace by this many pixels in the cross-dispersion
        (Y) direction.  Positive = up, negative = down.  Used to place the
        octagonal sky-reference fiber below the science fiber.

    Returns
    -------
    np.ndarray  shape (4096, 4096), float64
    """
    print("Loading XY table …")
    xy_table = load_xy_table(xy_path)

    # Build order → wavelength list (file order = XY table order)
    order_wave_map = {
        order: [row[0] for row in rows]
        for order, rows in xy_table.items()
    }

    print("Loading PSF library …")
    lib = PSFLibrary(psf_dir=psf_dir, order_wave_map=order_wave_map)

    # Prepare spectrum evaluator.
    # If a custom (wave, flux) grid is supplied use it; otherwise fall back to
    # make_spectrum, which is evaluated lazily inside the loop.
    # spectrum_wave is user-facing (nm); convert to µm for internal arithmetic.
    use_custom_spectrum = spectrum_wave is not None
    if use_custom_spectrum:
        spec_wave = np.asarray(spectrum_wave, dtype=np.float64) * 1e-3  # nm → µm
        spec_flux = np.asarray(spectrum_flux, dtype=np.float64)
        if spec_wave.shape != spec_flux.shape:
            raise ValueError("spectrum_wave and spectrum_flux must have the same length")

    # Sky emission setup.
    use_sky = sky_wave is not None and sky_flux is not None
    if use_sky:
        sky_wave_um  = np.asarray(sky_wave,  dtype=np.float64) * 1e-3  # nm → µm
        sky_flux_arr = np.asarray(sky_flux,  dtype=np.float64)   # photons/nm at detector
        print(f"  Sky emission: scale×{sky_scale:.3g}  (sky in absolute photons/nm)")

    adaptive_sampling = wave_step_pix_frac is not None and wave_step_pix_frac > 0.0
    wave_step_um = wave_step_nm * 1e-3   # used in fixed-step mode

    canvas_size = DETECTOR_SIZE + 2 * CANVAS_MARGIN
    detector    = np.zeros((canvas_size, canvas_size), dtype=np.float64)
    deposited   = 0
    clipped     = 0

    if adaptive_sampling:
        print(f"Simulating (adaptive wave step: {wave_step_pix_frac:.3f} pix frac) …")
    else:
        print(f"Simulating (wave step = {wave_step_nm} nm) …")
    _order_list = sorted(xy_table.items())
    _n_orders   = len(_order_list)
    for _oi, (order, rows) in enumerate(tqdm(_order_list, desc="Orders", unit="order")):
        if order not in lib.psfs:
            print(f"  [skip] order {order}: no PSF data available")
            continue

        # Reference arrays for this order (sorted ascending by wavelength so
        # that np.interp works correctly; the XY table is descending, hence sort)
        ref_waves = np.array([r[0] for r in rows])
        ref_x     = np.array([r[1] for r in rows])
        ref_y     = np.array([r[2] for r in rows])
        sort_idx  = np.argsort(ref_waves)
        ref_waves = ref_waves[sort_idx]
        ref_x     = ref_x[sort_idx]
        ref_y     = ref_y[sort_idx]

        # Blaze parameters: centre wavelength and FSR for this order.
        # These are needed before building the wave_grid because we extend
        # the grid to the sinc² first zeros (λ_c ± Δλ) so each order fades
        # smoothly to zero flux at both ends instead of cutting off abruptly.
        lambda_c     = ref_waves[len(ref_waves) // 2]   # middle reference λ
        delta_lambda = lambda_c / order                  # FSR = λ_c / m

        # Periodic progress print every 10 orders (visible in GUI console)
        if _oi % 10 == 0:
            pct = int(100 * _oi / _n_orders)
            print(f"  Orders {_oi+1}–{min(_oi+10, _n_orders)}/{_n_orders}"
                  f"  ({pct}%)  λ_c = {lambda_c*1000:.0f} nm  m={order}")

        # Dense wavelength grid extended to the sinc² first zeros.
        # Beyond the Zemax reference range the position and PSF CubicSplines
        # extrapolate; the detector-edge bounds check in the stamp loop clips
        # anything that wanders off the 4096×4096 array.
        wave_min  = lambda_c - delta_lambda
        wave_max  = lambda_c + delta_lambda
        if adaptive_sampling:
            # Estimate local dispersion for this order from consecutive XY samples:
            #   nm_per_pix = |d lambda_nm| / sqrt((dx_mm/pix_mm)^2 + (dy_mm/pix_mm)^2)
            dl_nm = np.abs(np.diff(ref_waves)) * 1000.0
            dmm = np.hypot(np.diff(ref_x), np.diff(ref_y))
            dpix = dmm / PIXEL_SIZE_MM
            valid = (dpix > 0.0) & (dl_nm > 0.0)
            if np.any(valid):
                nm_per_pix_ord = float(np.median(dl_nm[valid] / dpix[valid]))
                order_wave_step_nm = max(1e-6, float(wave_step_pix_frac) * nm_per_pix_ord)
            else:
                order_wave_step_nm = wave_step_nm
        else:
            order_wave_step_nm = wave_step_nm

        order_wave_step_um = order_wave_step_nm * 1e-3
        wave_grid = np.arange(wave_min, wave_max + order_wave_step_um, order_wave_step_um)

        # Detector position at each wavelength step.
        # _cs_linextrap uses CubicSpline inside the reference range and the
        # tangent line outside, so the trace never rolls back on itself when
        # extrapolated into the blaze wings beyond the Zemax reference points.
        x_fn   = _cs_linextrap(ref_waves, ref_x)
        y_fn   = _cs_linextrap(ref_waves, ref_y)
        x_grid = x_fn(wave_grid)
        y_grid = y_fn(wave_grid)

        # Spectrum flux at each wavelength step
        if use_custom_spectrum:
            flux_grid = np.interp(wave_grid, spec_wave, spec_flux, left=0.0, right=0.0)
        else:
            flux_grid = make_spectrum(wave_grid)

        # Add sky emission on top of stellar flux (both now in photons/nm)
        if use_sky:
            sky_grid  = np.interp(wave_grid, sky_wave_um, sky_flux_arr,
                                  left=0.0, right=0.0)
            flux_grid = flux_grid + sky_scale * sky_grid

        # Blaze envelope: sinc²(m · (λ − λ_c) / λ_c)
        # Peaks at λ_c, reaches its first zeros at λ_c ± Δλ (the wave_grid
        # boundaries), so the trace fades naturally to zero at both ends.
        if blaze:
            blaze_grid = np.sinc((wave_grid - lambda_c) / delta_lambda) ** 2
            flux_grid  = flux_grid * blaze_grid

        for wave_um, x_mm, y_mm, flux in zip(wave_grid, x_grid, y_grid, flux_grid):
            # flux is in photons/nm; multiply by wave_step_nm to get photons per step.
            # Skip steps that contribute < 1e-6 photons to the detector.
            stamp_photons = flux * order_wave_step_nm
            if stamp_photons < 1e-6:
                continue

            # Exact (sub-pixel) detector position on the padded canvas
            cx_f, cy_f = mm_to_pix_exact(x_mm, y_mm, margin=CANVAS_MARGIN)
            cy_f += y_offset_pix   # cross-dispersion shift for this fiber
            # Integer pixel of the stamp centre
            cx = int(round(cx_f))
            cy = int(round(cy_f))
            # Fractional offset from the integer centre (in pixels)
            dx = cx_f - cx
            dy = cy_f - cy

            # Bounds check against the padded canvas
            x0, x1 = cx - PSF_HALF, cx + PSF_HALF
            y0, y1 = cy - PSF_HALF, cy + PSF_HALF
            if x0 < 0 or x1 > canvas_size or y0 < 0 or y1 > canvas_size:
                clipped += 1
                continue

            psf = lib.get_psf(order, wave_um)   # shape (80, 80), normalised

            # Shift the unbinned 80×80 PSF by the sub-pixel offset (scaled by
            # BIN_FACTOR) so the centroid lands at the true trace position.
            # Shifting before binning avoids spline artefacts on the coarse
            # 20×20 grid.
            dy_fine = dy * BIN_FACTOR
            dx_fine = dx * BIN_FACTOR
            if dy_fine != 0.0 or dx_fine != 0.0:
                psf = nd_shift(psf, shift=[dy_fine, dx_fine], order=3,
                               mode='constant', cval=0.0)
                total_flux = psf.sum()
                if total_flux > 0:
                    psf /= total_flux   # renormalise after shift

            # Bin to 20×20 after shifting
            psf, _ = _bin_psf(psf)

            detector[y0:y1, x0:x1] += stamp_photons * psf
            deposited += 1

    total = deposited + clipped
    print(f"  Orders {_n_orders}/{_n_orders} (100%)  — simulation complete")
    print(
        f"  Deposited {deposited}/{total} stamps "
        f"({clipped} clipped at detector edge)"
    )

    # Crop the padded canvas back to the nominal DETECTOR_SIZE×DETECTOR_SIZE
    # region, centred on the optical axis.  The CANVAS_MARGIN border was only
    # needed to avoid clipping orders whose blaze wings extend slightly beyond
    # the physical detector boundary.
    detector = detector[
        CANVAS_MARGIN : CANVAS_MARGIN + DETECTOR_SIZE,
        CANVAS_MARGIN : CANVAS_MARGIN + DETECTOR_SIZE,
    ]
    detector = np.clip(detector, 0.0, None)  # no negative values in output

    if output_fits:
        _save_fits(detector, output_fits)

    if output_png:
        _save_preview(detector, output_png)

    return detector


def _save_fits(
    detector: np.ndarray,
    path: str,
    meta: dict | None = None,
) -> None:
    """Save the detector image as a single-extension FITS file (float32).

    Parameters
    ----------
    detector : np.ndarray
        2D detector image in units of detected photons.
    path : str
        Output FITS file path (created or overwritten).
    meta : dict, optional
        ``{keyword: (value, comment)}`` pairs to write into the primary FITS
        header.  Keyword strings must be valid FITS keys (8 characters max).
        Values may be booleans, ints, floats, or strings.
    """
    try:
        from astropy.io import fits
    except ImportError:
        print("  [warn] astropy not available; skipping FITS output")
        return

    hdu = fits.PrimaryHDU(detector.astype(np.float32))
    if meta:
        for key, (val, comment) in meta.items():
            try:
                hdu.header[key] = (val, comment)
            except Exception as exc:
                print(f"  [warn] FITS keyword {key!r} skipped: {exc}")
    hdu.writeto(path, overwrite=True)
    print(f"Saved FITS  → {path}")


def _save_preview(detector: np.ndarray, path: str) -> None:
    """Save a 2x2-binned PNG preview scaled from 0 to 2x P90."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available; skipping PNG preview")
        return

    img = np.clip(detector, 0.0, None)

    # True detector binning for preview: 2x2 native pixels -> one output pixel.
    # This reduces aliasing compared with pure resampling.
    ny, nx = img.shape
    ny2 = (ny // 2) * 2
    nx2 = (nx // 2) * 2
    if ny2 > 0 and nx2 > 0:
        binned = img[:ny2, :nx2].reshape(ny2 // 2, 2, nx2 // 2, 2).sum(axis=(1, 3))
    else:
        binned = img

    pos = binned[binned > 0.0]
    if pos.size:
        p90 = float(np.percentile(pos, 90.0))
    else:
        p90 = 0.0
    vmax = 2.0 * p90
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(
        binned,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title("VROOMM simulated detector preview (2x2 binned, 0..2xP90)", fontsize=12)
    ax.set_xlabel("X pixel (2x2 binned)")
    ax.set_ylabel("Y pixel (2x2 binned)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved preview → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate VROOMM 2D detector image"
    )
    parser.add_argument(
        "--params",
        default=DEFAULT_PARAMS,
        metavar="YAML",
        help="Parameter YAML file (default: simulate_params.yaml)",
    )
    # Individual overrides – any supplied value beats the YAML
    parser.add_argument("--psf-dir",     default=None, help="Override psf_dir")
    parser.add_argument("--xy-table",    default=None, help="Override xy_table")
    parser.add_argument("--output-fits", default=None, help="Override output_fits ('' to skip)")
    parser.add_argument("--output-png",  default=None, help="Override output_png  ('' to skip)")
    parser.add_argument("--wave-step",   default=None, type=float, metavar="NM",
                        help="Override wave_step_nm [nm]")
    args = parser.parse_args()

    # Load base parameters from YAML, then apply any CLI overrides
    kwargs = load_params(args.params)
    if args.psf_dir     is not None: kwargs["psf_dir"]      = args.psf_dir
    if args.xy_table    is not None: kwargs["xy_path"]       = args.xy_table
    if args.output_fits is not None: kwargs["output_fits"]   = args.output_fits or None
    if args.output_png  is not None: kwargs["output_png"]    = args.output_png  or None
    if args.wave_step   is not None: kwargs["wave_step_nm"]  = args.wave_step

    oct_conf    = kwargs.pop("oct_conf",  None)
    fits_meta   = kwargs.pop("fits_meta", None)
    output_fits = kwargs.get("output_fits")
    output_png  = kwargs.get("output_png")

    if oct_conf is not None:
        # Dual-fiber mode: simulate each fiber without saving, then combine.
        print("\n── Rectangular fiber (star + sky) ──────────────────────────────")
        img = simulate_detector(**{**kwargs, "output_fits": None, "output_png": None})

        oct_mode = ("lamp" if oct_conf.get("spectrum_wave") is not None
                    else "sky only")
        print(f"\n── Octagonal fiber ({oct_mode}) "
              "───────────────────────────────────")
        oct_kwargs = dict(
            psf_dir      = oct_conf["psf_dir"],
            xy_path      = kwargs["xy_path"],
            output_fits  = None,
            output_png   = None,
            wave_step_nm = kwargs["wave_step_nm"],
            spectrum_wave= oct_conf.get("spectrum_wave"),
            spectrum_flux= oct_conf.get("spectrum_flux"),
            blaze        = kwargs["blaze"],
            sky_wave     = oct_conf.get("sky_wave"),
            sky_flux     = oct_conf.get("sky_flux"),
            sky_scale    = oct_conf["sky_scale"],
            y_offset_pix = oct_conf["y_offset_pix"],
        )
        img_oct = simulate_detector(**oct_kwargs)

        print("\nCombining rectangular + octagonal fiber images …")
        img = img + img_oct
        if output_fits:
            _save_fits(img, output_fits, fits_meta)
        if output_png:
            _save_preview(img, output_png)
    else:
        # Single-fiber mode: suppress internal save so we can write with metadata.
        img = simulate_detector(**{**kwargs, "output_fits": None, "output_png": None})
        if output_fits:
            _save_fits(img, output_fits, fits_meta)
        if output_png:
            _save_preview(img, output_png)

    print(
        f"\nDetector stats:"
        f"\n  shape      : {img.shape}"
        f"\n  non-zero px: {(img > 0).sum()}"
        f"\n  min / max  : {img.min():.3e} / {img.max():.3e}"
        f"\n  total flux : {img.sum():.3e}"
    )
