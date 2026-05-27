#!/usr/bin/env python3
"""
night_planner.py — VROOMM Night Planner GUI

Build a sequence of calibrations and science observations for one night.
Each slot gets a UTC timestamp-based filename.  Science targets are resolved
via SIMBAD (fails loudly if unresolved).  BERV is computed with barycorrpy.

Usage:
    python night_planner.py
"""

import datetime
import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

import numpy as np
import yaml
from run_night import RunNightFrame
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

try:
    from astroquery.simbad import Simbad
    _HAS_ASTROQUERY = True
except ImportError:
    _HAS_ASTROQUERY = False

try:
    from astroquery.gaia import Gaia
    _HAS_GAIA = True
except ImportError:
    _HAS_GAIA = False

try:
    from barycorrpy import get_BC_vel
    _HAS_BARYCORRPY = True
except ImportError:
    _HAS_BARYCORRPY = False

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_LOGO_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
_TARGET_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "targets")
_SIM_PARAMS   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulate_params.yaml")
os.makedirs(_TARGET_CACHE, exist_ok=True)

# ── Default observatories ──────────────────────────────────────────────────────
OBS_DEFAULTS = {
    "CFHT" : dict(lat=19.8267,  lon=-155.4742, alt=4204.1),
    "OMM"  : dict(lat=45.4554,  lon=-71.1527,  alt=1111.0),
    "OHP"  : dict(lat=43.9308,  lon=5.7133,    alt=650.0),
    "La Silla": dict(lat=-29.2567, lon=-70.7346, alt=2400.0),
}

OBS_TIMEZONES = {
    "CFHT": "Pacific/Honolulu",
    "OMM": "America/Toronto",
    "OHP": "Europe/Paris",
    "La Silla": "America/Santiago",
}

# ── Observation types ──────────────────────────────────────────────────────────
CAL_TYPES = ["FLAT", "THAR", "UNE", "FP", "DARK", "BIAS"]
SCI_TYPE  = "SCIENCE"

DEFAULT_EXP = {
    "FLAT"   : 30.0,
    "THAR"   : 30.0,
    "UNE"    : 60.0,
    "FP"     : 30.0,
    "DARK"   : 300.0,
    "BIAS"   : 0.1,
    "SCIENCE": 900.0,
}


def _load_default_exposures_from_config() -> None:
    """Update DEFAULT_EXP from simulate_params.yaml calibration_defaults."""
    try:
        with open(_SIM_PARAMS) as fh:
            p = yaml.safe_load(fh) or {}
    except Exception:
        return

    cdef = p.get("calibration_defaults", {})
    if isinstance(cdef, dict):
        for key in ("FLAT", "THAR", "UNE", "FP", "DARK", "BIAS", "SCIENCE"):
            if key in cdef:
                try:
                    v = float(cdef[key])
                    if np.isfinite(v) and v >= 0.0:
                        DEFAULT_EXP[key] = v
                except (TypeError, ValueError):
                    pass

    # Optional science default from observation block
    obs = p.get("observation", {})
    if isinstance(obs, dict):
        val = obs.get("default_science_exp_s")
        if val is not None:
            try:
                v = float(val)
                if np.isfinite(v) and v >= 0.0:
                    DEFAULT_EXP["SCIENCE"] = v
            except (TypeError, ValueError):
                pass


_load_default_exposures_from_config()

READOUT_S = 30.0   # default detector readout overhead [s] (overridden by GUI)


# ── Vacuum wavelength conversion (Morton 2000) — not used here but kept ────────
def air_to_vac_nm(wave_air_nm: np.ndarray) -> np.ndarray:
    sigma2 = (1e3 / wave_air_nm) ** 2
    n = 1 + 6.4328e-5 + 2.94981e-2 / (146 - sigma2) + 2.5540e-4 / (41 - sigma2)
    return wave_air_nm * n


# ── SIMBAD resolver ──────────────────────────────────────────────────────────

def _target_cache_path(name: str) -> str:
    """Return the YAML cache path for a target name."""
    import re
    safe = re.sub(r'[^\w\-]', '_', name.strip()).strip('_').lower()
    return os.path.join(_TARGET_CACHE, f"{safe}.yaml")


def load_target_cache(name: str) -> dict | None:
    """Load cached SIMBAD data for *name*, or None if not cached."""
    path = _target_cache_path(name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return yaml.safe_load(fh)


def save_target_cache(name: str, data: dict):
    """Persist *data* (as returned by resolve_simbad) to disk for *name*."""
    path = _target_cache_path(name)
    with open(path, 'w') as fh:
        yaml.dump({**data, 'name': name}, fh, allow_unicode=True, sort_keys=False)


def _list_cached_names() -> list:
    """Return all target names stored in the cache, sorted alphabetically."""
    names = []
    if not os.path.isdir(_TARGET_CACHE):
        return names
    for fname in os.listdir(_TARGET_CACHE):
        if not fname.endswith(".yaml"):
            continue
        try:
            with open(os.path.join(_TARGET_CACHE, fname)) as fh:
                d = yaml.safe_load(fh)
            if d and "name" in d:
                names.append(d["name"])
        except Exception:
            pass
    return sorted(names)


def resolve_simbad(name: str) -> dict:
    """
    Query SIMBAD. Returns dict with astrometry, RV, and (when available)
    stellar parameters Teff/logg/vsini and Gaia RP magnitude (Grp).
    Raises ValueError if not found.  Raises RuntimeError if astroquery missing.
    """
    if not _HAS_ASTROQUERY:
        raise RuntimeError("astroquery is not installed.  Run:  pip install astroquery")

    def _safe_float(val, default=0.0):
        if val is None or np.ma.is_masked(val):
            return default
        try:
            v = float(val)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    s = Simbad()
    # rvz_radvel is the current field name (rv_value was retired)
    # mesFe_h includes teff/log_g, mesRot includes vsini.
    s.add_votable_fields("pmra", "pmdec", "rvz_radvel", "mesFe_h", "mesRot")
    result = s.query_object(name)
    if result is None or len(result) == 0:
        raise ValueError(f"SIMBAD could not resolve '{name}'")
    row = result[0]

    # Grp magnitude from Gaia DR3 (phot_rp_mean_mag), linked via SIMBAD IDs.
    grp_val = np.nan
    if _HAS_GAIA:
        try:
            ids_tbl = s.query_objectids(name)
            gaia_dr3_id = None
            if ids_tbl is not None and len(ids_tbl) > 0:
                import re
                id_col = None
                for cname in ids_tbl.colnames:
                    if cname.lower() == "id":
                        id_col = cname
                        break
                if id_col is None:
                    id_col = ids_tbl.colnames[0]
                for rid in ids_tbl[id_col]:
                    sid = str(rid)
                    m = re.match(r"^Gaia\s+DR3\s+(\d+)$", sid)
                    if m:
                        gaia_dr3_id = m.group(1)
                        break
            if gaia_dr3_id is not None:
                job = Gaia.launch_job(
                    "SELECT phot_rp_mean_mag "
                    "FROM gaiadr3.gaia_source "
                    f"WHERE source_id = {gaia_dr3_id}"
                )
                gaia_res = job.get_results()
                if gaia_res is not None and len(gaia_res) > 0:
                    grp_val = gaia_res[0]["phot_rp_mean_mag"]
            # Fallback: nearest Gaia DR3 source around SIMBAD coordinates.
            if not np.isfinite(_safe_float(grp_val, default=np.nan)):
                ra0 = float(row["ra"])
                dec0 = float(row["dec"])
                rad_deg = 120.0 / 3600.0
                q = (
                    "SELECT TOP 1 phot_rp_mean_mag, "
                    "DISTANCE(POINT('ICRS', ra, dec), "
                    f"POINT('ICRS', {ra0}, {dec0})) AS dist "
                    "FROM gaiadr3.gaia_source "
                    "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
                    f"CIRCLE('ICRS', {ra0}, {dec0}, {rad_deg})) "
                    "ORDER BY dist ASC"
                )
                job = Gaia.launch_job(q)
                gaia_res = job.get_results()
                if gaia_res is not None and len(gaia_res) > 0:
                    grp_val = gaia_res[0]["phot_rp_mean_mag"]
        except Exception:
            grp_val = np.nan

    # New SIMBAD TAP API: columns are lowercase, ra/dec already in degrees
    return dict(
        ra    = float(row["ra"]),
        dec   = float(row["dec"]),
        pmra  = _safe_float(row["pmra"]),
        pmdec = _safe_float(row["pmdec"]),
        rv    = _safe_float(row["rvz_radvel"]),
        teff  = _safe_float(row.get("mesfe_h.teff"), default=np.nan),
        logg  = _safe_float(row.get("mesfe_h.log_g"), default=np.nan),
        vsini = _safe_float(row.get("mesrot.vsini"), default=np.nan),
        grp_mag = _safe_float(grp_val, default=np.nan),
    )


# ── BERV via barycorrpy ────────────────────────────────────────────────────────
def compute_berv(ra_deg, dec_deg, pmra, pmdec, utc_dt: datetime.datetime, obs: dict) -> float:
    if not _HAS_BARYCORRPY:
        return 0.0
    jd = Time(utc_dt.strftime("%Y-%m-%dT%H:%M:%S"), format="isot", scale="utc").jd
    result = get_BC_vel(
        JDUTC = jd,
        ra    = ra_deg,
        dec   = dec_deg,
        pmra  = pmra,
        pmdec = pmdec,
        lat   = obs["lat"],
        longi = obs["lon"],
        alt   = obs["alt"],
        epoch = 2451545.0,
    )
    return float(result[0][0]) / 1000.0   # m/s → km/s


# ── Airmass ────────────────────────────────────────────────────────────────────
def compute_airmass(ra_deg: float, dec_deg: float,
                    utc_dt: datetime.datetime, obs: dict) -> float:
    """Return sec(z) airmass at the given UTC time from the given observatory."""
    location = EarthLocation(lat=obs["lat"] * u.deg,
                             lon=obs["lon"] * u.deg,
                             height=obs["alt"] * u.m)
    t = Time(utc_dt.strftime("%Y-%m-%dT%H:%M:%S"), format="isot", scale="utc")
    coord  = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    altaz  = coord.transform_to(AltAz(obstime=t, location=location))
    alt_deg = float(altaz.alt.deg)
    if alt_deg <= 0.0:
        return float("nan")        # target below horizon
    return float(altaz.secz)


def optimal_date_from_ra(ra_deg: float) -> str:
    """Return a rough best observing window label like 'Early-May' from RA."""
    # Around midnight, target RA ~= Sun RA + 12h.
    ra_h = (float(ra_deg) / 15.0) % 24.0
    sun_ra_h = (ra_h - 12.0) % 24.0
    day_offset = int(round((sun_ra_h / 24.0) * 365.0))
    d0 = datetime.date(2025, 3, 21) + datetime.timedelta(days=day_offset)

    if d0.day <= 10:
        phase = "Early"
    elif d0.day <= 20:
        phase = "Mid"
    else:
        phase = "Late"
    return f"{phase}-{d0.strftime('%B')}"


def _tz_from_obs(obs: dict):
    """Return a tzinfo for the observatory, preferring real IANA timezones."""
    if ZoneInfo is not None:
        tz_name = OBS_TIMEZONES.get(obs.get("name", ""))
        if tz_name:
            try:
                return ZoneInfo(tz_name)
            except Exception:
                pass
    # Fallback for custom/unknown sites: fixed offset estimated from longitude.
    lon = float(obs.get("lon", 0.0))
    offset_h = int(round(lon / 15.0))
    sign = "+" if offset_h >= 0 else "-"
    return datetime.timezone(datetime.timedelta(hours=offset_h),
                             name=f"UTC{sign}{abs(offset_h):02d}:00")


def _observes_dst(tzinfo, year: int) -> bool:
    """Best-effort check whether tzinfo has seasonal DST changes."""
    jan = datetime.datetime(year, 1, 15, tzinfo=tzinfo).utcoffset()
    jul = datetime.datetime(year, 7, 15, tzinfo=tzinfo).utcoffset()
    return jan != jul


def format_local_time(utc_dt: datetime.datetime, obs: dict) -> str:
    """Format local civil time from UTC with explicit timezone context."""
    tzinfo = _tz_from_obs(obs)
    dt_utc = utc_dt.replace(tzinfo=datetime.timezone.utc)
    dt_loc = dt_utc.astimezone(tzinfo)
    abbr = dt_loc.tzname() or "LT"
    if _observes_dst(tzinfo, dt_loc.year):
        season = "summer" if (dt_loc.dst() and dt_loc.dst() != datetime.timedelta(0)) else "winter"
        return f"{dt_loc.strftime('%H:%M:%S')} {abbr} ({season})"
    return f"{dt_loc.strftime('%H:%M:%S')} {abbr}"


# ── Add Calibration dialog ─────────────────────────────────────────────────────
class AddCalDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Add Calibration")
        self.resizable(False, False)
        self.result = None
        self._build()
        self.grab_set()
        self.wait_window()

    def _build(self):
        pad = dict(padx=10, pady=5)
        tk.Label(self, text="Type:").grid(row=0, column=0, sticky="e", **pad)
        self._type = ttk.Combobox(self, values=CAL_TYPES, width=10, state="readonly")
        self._type.set("THAR")
        self._type.grid(row=0, column=1, sticky="w", **pad)
        self._type.bind("<<ComboboxSelected>>", self._on_type)

        tk.Label(self, text="Fiber(s):").grid(row=1, column=0, sticky="e", **pad)
        self._fiber = ttk.Combobox(self, values=["rect", "oct", "both"], width=10, state="readonly")
        self._fiber.set("rect")
        self._fiber.grid(row=1, column=1, sticky="w", **pad)

        tk.Label(self, text="Exp [s]:").grid(row=2, column=0, sticky="e", **pad)
        self._exp = tk.StringVar(value="30")
        tk.Entry(self, textvariable=self._exp, width=10).grid(row=2, column=1, sticky="w", **pad)

        tk.Label(self, text="Repeats:").grid(row=3, column=0, sticky="e", **pad)
        self._rep = tk.StringVar(value="1")
        tk.Entry(self, textvariable=self._rep, width=10).grid(row=3, column=1, sticky="w", **pad)

        bf = tk.Frame(self)
        bf.grid(row=4, column=0, columnspan=2, pady=8)
        tk.Button(bf, text="Add",    command=self._ok,      width=8).pack(side="left", padx=4)
        tk.Button(bf, text="Cancel", command=self.destroy,  width=8).pack(side="left", padx=4)

    def _on_type(self, _=None):
        self._exp.set(str(int(DEFAULT_EXP.get(self._type.get(), 30))))

    def _ok(self):
        try:
            exp  = float(self._exp.get())
            reps = int(self._rep.get())
        except ValueError:
            messagebox.showerror("Input error", "Exposure and repeats must be numbers.", parent=self)
            return
        self.result = dict(obs_type=self._type.get(), fiber=self._fiber.get(),
                           exp_s=exp, repeats=reps)
        self.destroy()


# ── Add Science dialog ─────────────────────────────────────────────────────────
class AddScienceDialog(tk.Toplevel):
    def __init__(self, parent, obs: dict):
        super().__init__(parent)
        self.title("Add Science Target")
        self.resizable(False, False)
        self._obs  = obs
        self.result = None
        self._resolved = None
        self._cache_names = _list_cached_names()   # all names on disk for autocomplete
        self._build()
        self.grab_set()
        self.wait_window()

    def _build(self):
        pad = dict(padx=10, pady=4)

        tk.Label(self, text="Target name:").grid(row=0, column=0, sticky="e", **pad)
        nf = tk.Frame(self)
        nf.grid(row=0, column=1, columnspan=2, sticky="w", **pad)
        self._target_var = tk.StringVar()
        self._name_entry = ttk.Combobox(nf, textvariable=self._target_var,
                                        values=self._cache_names, width=20)
        self._name_entry.pack(side="left")
        self._name_entry.bind("<<ComboboxSelected>>", self._on_selected)
        tk.Button(nf, text="Resolve ↵", command=self._resolve).pack(side="left", padx=4)

        tk.Label(self, text="RA (°):").grid(row=1, column=0, sticky="e", **pad)
        self._ra  = tk.StringVar(value="—")
        tk.Label(self, textvariable=self._ra,  width=14, anchor="w",
                 relief="sunken", bg="#eef").grid(row=1, column=1, sticky="w", **pad)

        tk.Label(self, text="Dec (°):").grid(row=2, column=0, sticky="e", **pad)
        self._dec = tk.StringVar(value="—")
        tk.Label(self, textvariable=self._dec, width=14, anchor="w",
                 relief="sunken", bg="#eef").grid(row=2, column=1, sticky="w", **pad)

        tk.Label(self, text="RV_sys [km/s]:").grid(row=3, column=0, sticky="e", **pad)
        self._rv = tk.StringVar(value="0.0")
        tk.Entry(self, textvariable=self._rv, width=12).grid(row=3, column=1, sticky="w", **pad)

        tk.Label(self, text="Grp [mag]:").grid(row=4, column=0, sticky="e", **pad)
        self._grp = tk.StringVar(value="")
        tk.Entry(self, textvariable=self._grp, width=12).grid(row=4, column=1, sticky="w", **pad)

        tk.Label(self, text="Teff [K]:").grid(row=5, column=0, sticky="e", **pad)
        self._teff = tk.StringVar(value="")
        tk.Entry(self, textvariable=self._teff, width=12).grid(row=5, column=1, sticky="w", **pad)

        tk.Label(self, text="log g:").grid(row=6, column=0, sticky="e", **pad)
        self._logg = tk.StringVar(value="")
        tk.Entry(self, textvariable=self._logg, width=12).grid(row=6, column=1, sticky="w", **pad)

        tk.Label(self, text="vsini [km/s]:").grid(row=7, column=0, sticky="e", **pad)
        self._vsini = tk.StringVar(value="")
        tk.Entry(self, textvariable=self._vsini, width=12).grid(row=7, column=1, sticky="w", **pad)

        tk.Label(self, text="Exp [s]:").grid(row=8, column=0, sticky="e", **pad)
        self._exp = tk.StringVar(value=str(int(DEFAULT_EXP["SCIENCE"])))
        tk.Entry(self, textvariable=self._exp, width=12).grid(row=8, column=1, sticky="w", **pad)

        tk.Label(self, text="Repeats:").grid(row=9, column=0, sticky="e", **pad)
        self._rep = tk.StringVar(value="1")
        tk.Entry(self, textvariable=self._rep, width=12).grid(row=9, column=1, sticky="w", **pad)

        tk.Label(self, text="Fiber:").grid(row=10, column=0, sticky="e", **pad)
        self._fiber_var = tk.StringVar(value="rect")
        ttk.Combobox(self, textvariable=self._fiber_var,
                     values=["rect", "oct"], width=8,
                 state="readonly").grid(row=10, column=1, sticky="w", **pad)

        self._status = tk.Label(self, text="", fg="grey", width=38, anchor="w")
        self._status.grid(row=11, column=0, columnspan=3, padx=10, pady=2)

        bf = tk.Frame(self)
        bf.grid(row=12, column=0, columnspan=3, pady=8)
        tk.Button(bf, text="Add",    command=self._ok,     width=8).pack(side="left", padx=4)
        tk.Button(bf, text="Cancel", command=self.destroy, width=8).pack(side="left", padx=4)

        self._name_entry.bind("<Return>", lambda _: self._resolve())
        self._name_entry.focus_set()
        self._name_entry.bind("<KeyRelease>", self._on_name_key)

    def _set_stellar_vars_from_dict(self, d: dict):
        """Fill Grp/Teff/logg/vsini entries from dict values (blank if missing)."""
        grp = d.get("grp_mag", np.nan)
        teff = d.get("teff", np.nan)
        logg = d.get("logg", np.nan)
        vsini = d.get("vsini", np.nan)
        self._grp.set(f"{grp:.3f}" if np.isfinite(grp) else "")
        self._teff.set(f"{teff:.0f}" if np.isfinite(teff) else "")
        self._logg.set(f"{logg:.2f}" if np.isfinite(logg) else "")
        self._vsini.set(f"{vsini:.2f}" if np.isfinite(vsini) else "")

    def _missing_stellar_fields(self) -> list[str]:
        """Return list of missing required science-target fields."""
        missing = []
        if not self._grp.get().strip():
            missing.append("Grp")
        if not self._teff.get().strip():
            missing.append("Teff")
        if not self._logg.get().strip():
            missing.append("logg")
        if not self._vsini.get().strip():
            missing.append("vsini")
        return missing

    def _on_name_key(self, _=None):
        """Update dropdown suggestions and auto-fill when an exact cached match exists."""
        name = self._target_var.get().strip()
        # Update dropdown with prefix-filtered cached names
        if name:
            lname = name.lower()
            matches = [n for n in self._cache_names if n.lower().startswith(lname)]
        else:
            matches = self._cache_names
        self._name_entry["values"] = matches
        # Auto-fill if we have an exact cached match
        cached = load_target_cache(name)
        if cached is not None and self._resolved is None:
            self._resolved = cached
            self._ra.set(f"{cached['ra']:.6f}")
            self._dec.set(f"{cached['dec']:.6f}")
            if abs(cached.get('rv', 0.0)) > 0.01:
                self._rv.set(f"{cached['rv']:.3f}")
            self._set_stellar_vars_from_dict(cached)
            miss = self._missing_stellar_fields()
            if miss:
                self._status.config(
                    text=(f"☑ cached astrometry; please enter missing: {', '.join(miss)}"),
                    fg="#b36b00")
            else:
                self._status.config(
                    text=(f"☑ cached  RA={cached['ra']:.4f}°  Dec={cached['dec']:.4f}°  "
                          f"Grp={self._grp.get()}  Teff={self._teff.get()}  "
                          f"logg={self._logg.get()}  vsini={self._vsini.get()}"),
                    fg="#007700")

    def _on_selected(self, _=None):
        """User picked a name from the dropdown — resolve immediately from cache."""
        self._resolved = None   # reset so auto-fill fires again
        self._on_name_key()

    def _resolve(self):
        name = self._target_var.get().strip()
        if not name:
            return
        # Check disk cache first
        cached = load_target_cache(name)
        if cached is not None:
            self._resolved = cached
            self._ra.set(f"{cached['ra']:.6f}")
            self._dec.set(f"{cached['dec']:.6f}")
            if abs(cached.get('rv', 0.0)) > 0.01:
                self._rv.set(f"{cached['rv']:.3f}")
            self._set_stellar_vars_from_dict(cached)
            miss = self._missing_stellar_fields()
            if miss:
                self._status.config(
                    text=(f"☑ cached astrometry; please enter missing: {', '.join(miss)}"),
                    fg="#b36b00")
            else:
                self._status.config(
                    text=(f"☑ cached  RA={cached['ra']:.4f}°  Dec={cached['dec']:.4f}°  "
                          f"Grp={self._grp.get()}  Teff={self._teff.get()}  "
                          f"logg={self._logg.get()}  vsini={self._vsini.get()}"),
                    fg="#007700")
            return
        self._status.config(text="Querying SIMBAD …", fg="orange")
        self.update()
        try:
            d = resolve_simbad(name)
            self._resolved = d
            save_target_cache(name, d)
            self._ra.set(f"{d['ra']:.6f}")
            self._dec.set(f"{d['dec']:.6f}")
            if abs(d["rv"]) > 0.01:
                self._rv.set(f"{d['rv']:.3f}")
            self._set_stellar_vars_from_dict(d)
            miss = self._missing_stellar_fields()
            if miss:
                self._status.config(
                    text=(f"✓  RA={d['ra']:.4f}°  Dec={d['dec']:.4f}°  "
                          f"SIMBAD missing: {', '.join(miss)} — please enter manually"),
                    fg="#b36b00")
            else:
                self._status.config(
                    text=(f"✓  RA={d['ra']:.4f}°  Dec={d['dec']:.4f}°  "
                          f"Grp={self._grp.get()}  Teff={self._teff.get()}  "
                          f"logg={self._logg.get()}  vsini={self._vsini.get()}"),
                    fg="green")
        except Exception as exc:
            self._resolved = None
            self._ra.set("—")
            self._dec.set("—")
            self._status.config(text=f"✗  {exc}", fg="red")

    def _ok(self):
        if self._resolved is None:
            messagebox.showerror("No target",
                "Resolve a target via SIMBAD first.", parent=self)
            return
        try:
            exp  = float(self._exp.get())
            reps = int(self._rep.get())
            rv   = float(self._rv.get())
            grp_mag = float(self._grp.get())
            teff = float(self._teff.get())
            logg = float(self._logg.get())
            vsini = float(self._vsini.get())
        except ValueError:
            messagebox.showerror("Input error",
                "Exposure, repeats, RV, Grp, Teff, logg and vsini must be numbers.", parent=self)
            return
        if not np.isfinite(grp_mag):
            messagebox.showerror("Input error", "Grp must be finite.", parent=self)
            return
        if teff <= 0.0:
            messagebox.showerror("Input error", "Teff must be > 0 K.", parent=self)
            return
        if vsini < 0.0:
            messagebox.showerror("Input error", "vsini must be >= 0 km/s.", parent=self)
            return

        # Persist user-entered stellar parameters so next lookup is complete.
        merged = {
            **self._resolved,
            "grp_mag": grp_mag,
            "teff": teff,
            "logg": logg,
            "vsini": vsini,
        }
        save_target_cache(self._target_var.get().strip(), merged)
        self._resolved = merged

        self.result = dict(
            obs_type = SCI_TYPE,
            target   = self._target_var.get().strip(),
            ra       = self._resolved["ra"],
            dec      = self._resolved["dec"],
            pmra     = self._resolved["pmra"],
            pmdec    = self._resolved["pmdec"],
            rv_sys   = rv,
            grp_mag  = grp_mag,
            teff     = teff,
            logg     = logg,
            vsini    = vsini,
            exp_s    = exp,
            repeats  = reps,
            fiber    = self._fiber_var.get(),
        )
        self.destroy()


# ── Overhead (Slew / Delay) dialog ────────────────────────────────────────────
class _OverheadDialog(tk.Toplevel):
    def __init__(self, parent, title: str, row_type: str, default_dur: float):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result = None
        self._row_type = row_type
        self._default_dur = default_dur
        self._build()
        self.grab_set()
        self.wait_window()

    def _build(self):
        pad = dict(padx=10, pady=6)
        tk.Label(self, text="Duration [s]:").grid(row=0, column=0, sticky="e", **pad)
        self._dur = tk.StringVar(value=str(int(self._default_dur)))
        tk.Entry(self, textvariable=self._dur, width=10).grid(row=0, column=1, sticky="w", **pad)

        tk.Label(self, text="Note (optional):").grid(row=1, column=0, sticky="e", **pad)
        self._label = tk.StringVar()
        tk.Entry(self, textvariable=self._label, width=20).grid(row=1, column=1, sticky="w", **pad)

        bf = tk.Frame(self)
        bf.grid(row=2, column=0, columnspan=2, pady=8)
        tk.Button(bf, text="Add",    command=self._ok,     width=8).pack(side="left", padx=4)
        tk.Button(bf, text="Cancel", command=self.destroy, width=8).pack(side="left", padx=4)

    def _ok(self):
        try:
            dur = float(self._dur.get())
        except ValueError:
            messagebox.showerror("Input error", "Duration must be a number.", parent=self)
            return
        self.result = dict(dur_s=dur, label=self._label.get().strip())
        self.destroy()


# ── Main application ───────────────────────────────────────────────────────────
class NightPlannerApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("VROOMM Night Planner")
        self.minsize(960, 560)
        self._rows: list[dict] = []
        self._logo_img = None   # keep reference to avoid GC
        self._icon_img = None
        self._expert_mode_var = tk.BooleanVar(value=False)
        self._expert_temp_params: str | None = None
        self._load_logo()
        self._build_ui()
        self._refresh_table()

    # ── Logo ───────────────────────────────────────────────────────────────────
    def _load_logo(self):
        if not os.path.exists(_LOGO_PATH):
            return
        try:
            if _HAS_PIL:
                img = Image.open(_LOGO_PATH).convert("RGBA")
                target_h = 54
                w, h = img.size
                target_w = int(w * target_h / h)
                img = img.resize((target_w, target_h), Image.LANCZOS)
                self._logo_img = ImageTk.PhotoImage(img)
                # 32×32 window icon
                icon = Image.open(_LOGO_PATH).convert("RGBA").resize((32, 32), Image.LANCZOS)
                self._icon_img = ImageTk.PhotoImage(icon)
            else:
                raw = tk.PhotoImage(file=_LOGO_PATH)
                factor = max(1, raw.height() // 54)
                self._logo_img = raw.subsample(factor, factor)
                self._icon_img = raw.subsample(max(1, raw.height() // 32),
                                               max(1, raw.width()  // 32))
            self.iconphoto(True, self._icon_img)
        except Exception:
            self._logo_img = None

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Night setup row: setup panel (left) + logo box (right) ───────────
        setup_row = tk.Frame(self)
        setup_row.pack(fill="x", padx=10, pady=(4, 4))

        # ── Night setup panel ─────────────────────────────────────────────────
        top = tk.LabelFrame(setup_row, text="Night Setup", padx=8, pady=6)
        top.pack(side="left", fill="x", expand=True, padx=(0, 8))

        # ── Logo box (to the right of Night Setup) ───────────────────────────
        if self._logo_img is not None:
            logo_box = tk.LabelFrame(setup_row, text="VROOMM", padx=8, pady=6)
            logo_box.pack(side="right", anchor="n")
            tk.Label(logo_box, image=self._logo_img, bg="white").pack(padx=2, pady=2)

        tk.Label(top, text="Observatory:").grid(row=0, column=0, sticky="e", padx=4)
        self._obs_var = tk.StringVar(value="OMM")
        obs_cb = ttk.Combobox(top, textvariable=self._obs_var,
                               values=list(OBS_DEFAULTS.keys()), width=8, state="readonly")
        obs_cb.grid(row=0, column=1, sticky="w", padx=4)
        obs_cb.bind("<<ComboboxSelected>>", self._on_obs_change)

        _omm = OBS_DEFAULTS["OMM"]
        for col, (lbl, attr, w, fmt) in enumerate([
            ("Lat (°):", "lat", 9,  str(_omm["lat"])),
            ("Lon (°):", "lon", 11, str(_omm["lon"])),
            ("Alt (m):", "alt", 7,  str(_omm["alt"])),
        ], start=2):
            tk.Label(top, text=lbl).grid(row=0, column=col*2,   sticky="e", padx=4)
            var = tk.StringVar(value=fmt)
            setattr(self, f"_obs_{attr}_var", var)
            ent = tk.Entry(top, textvariable=var, width=w, state="readonly")
            ent.grid(row=0, column=col*2+1, sticky="w", padx=2)
            setattr(self, f"_obs_{attr}_entry", ent)

        tk.Label(top, text="Date (UT):").grid(row=1, column=0, sticky="e", padx=4, pady=3)
        self._date_var = tk.StringVar(value=datetime.date.today().strftime("%Y-%m-%d"))
        tk.Entry(top, textvariable=self._date_var, width=12).grid(row=1, column=1, sticky="w", padx=4)

        tk.Label(top, text="Night start (UT):").grid(row=1, column=2, sticky="e", padx=4)
        self._start_var = tk.StringVar(value="22:00:00")
        tk.Entry(top, textvariable=self._start_var, width=10).grid(row=1, column=3, sticky="w", padx=4)

        tk.Label(top, text="Default exp [s]:").grid(row=1, column=4, sticky="e", padx=4)
        self._defexp_var = tk.StringVar(value="900")
        tk.Entry(top, textvariable=self._defexp_var, width=8).grid(row=1, column=5, sticky="w", padx=4)

        tk.Label(top, text="Overhead [s]:").grid(row=1, column=6, sticky="e", padx=4)
        self._overhead_var = tk.StringVar(value="30")
        tk.Entry(top, textvariable=self._overhead_var, width=6).grid(row=1, column=7, sticky="w", padx=4)

        tk.Label(top, text="Output dir:").grid(row=1, column=8, sticky="e", padx=4)
        _today_yymmdd = datetime.date.today().strftime("%y%m%d")
        self._outdir_var = tk.StringVar(value=f"./night_output/{_today_yymmdd}")
        self._outdir_user_edited = False
        self._outdir_var.trace_add("write", self._on_outdir_edit)
        tk.Entry(top, textvariable=self._outdir_var, width=22).grid(row=1, column=9, sticky="w", padx=4)
        tk.Button(top, text="…", command=self._browse_outdir, width=2).grid(row=1, column=10, padx=2)
        # Auto-update output dir when date changes (unless user has manually edited it)
        self._date_var.trace_add("write", self._on_date_change)

        # ── Sequence table ────────────────────────────────────────────────────
        tbl_frame = tk.LabelFrame(self, text="Observation Sequence", padx=8, pady=6)
        tbl_frame.pack(fill="both", expand=True, padx=10, pady=4)

        cols   = ("#", "Time (UT)", "Local Time", "Type", "Target", "RA (°)", "Dec (°)",
                  "RV_sys\n(km/s)", "BERV\n(km/s)", "Airmass", "Exp [s]", "Fibers", "Filename",
                  "Overwrite", "Status", "Elapsed[s]")
        widths = [28, 70, 150, 56, 130, 74, 74, 66, 68, 58, 56, 46, 170, 56, 68, 62]

        self._tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=18)
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            anchor = "center" if col in ("#", "Type", "Exp [s]", "Fibers") else "w"
            self._tree.column(col, width=w, minwidth=w, anchor=anchor)

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        self._tree.tag_configure("cal",           background="#eef4ff")
        self._tree.tag_configure("science",       background="#fffff0")
        self._tree.tag_configure("overhead",      background="#e0e0e0")
        self._tree.tag_configure("running",       background="#fff3cd")
        self._tree.tag_configure("done_ok",       background="#d4edda")
        self._tree.tag_configure("run_error",     background="#f8d7da")
        self._tree.tag_configure("skipped",       background="#e8e8e8")
        self._tree.tag_configure("unschedulable", background="#ffcccc", foreground="#990000")

        self._tree.bind("<ButtonRelease-1>", self._on_tree_click)

        # ── Button bar ────────────────────────────────────────────────────────
        bf = tk.Frame(self)
        bf.pack(fill="x", padx=10, pady=(0, 4))

        tk.Button(bf, text="+ Calibration", command=self._add_cal,
                  bg="#cce5ff", width=13).pack(side="left", padx=3)
        tk.Button(bf, text="+ Science",     command=self._add_science,
                  bg="#fff3cd", width=13).pack(side="left", padx=3)
        tk.Button(bf, text="+ Slew",        command=self._add_slew,
                  bg="#dddddd", width=8).pack(side="left", padx=3)
        tk.Button(bf, text="+ Delay",       command=self._add_delay,
                  bg="#dddddd", width=8).pack(side="left", padx=3)
        tk.Button(bf, text="Delete",        command=self._delete_row,
                  width=8).pack(side="left", padx=3)
        tk.Button(bf, text="↑", command=self._move_up,   width=3).pack(side="left", padx=1)
        tk.Button(bf, text="↓", command=self._move_down, width=3).pack(side="left", padx=1)
        tk.Button(bf, text="Recompute BERV+AM", command=self._recompute_berv,
                  width=18).pack(side="left", padx=6)
        tk.Button(bf, text="Clear All", command=self._clear_all,
                  width=9).pack(side="left", padx=3)

        tk.Checkbutton(
            bf,
            text="Expert Mode",
            variable=self._expert_mode_var,
            command=self._toggle_expert_mode,
        ).pack(side="left", padx=8)

        tk.Button(bf, text="Export YAML", command=self._export_yaml,
                  bg="#d4edda", width=12).pack(side="right", padx=3)
        tk.Button(bf, text="▶ Run Night", command=self._run_night,
                  bg="#cce5ff", width=13).pack(side="right", padx=3)
        tk.Button(bf, text="✕ Exit", command=self.quit,
                  bg="#f8d7da", width=8).pack(side="right", padx=6)

        # ── Expert-mode panel (hidden by default) ───────────────────────────
        self._expert_frame = tk.LabelFrame(
            self,
            text="Expert Mode — simulate_params.yaml",
            padx=8,
            pady=6,
        )

        expert_btn_row = tk.Frame(self._expert_frame)
        expert_btn_row.pack(fill="x", pady=(0, 4))

        tk.Button(expert_btn_row, text="Reload file",
                  command=self._reload_expert_yaml, width=11).pack(side="left", padx=2)
        tk.Button(expert_btn_row, text="Validate",
                  command=self._validate_expert_yaml, width=9).pack(side="left", padx=2)
        tk.Button(expert_btn_row, text="Save to disk",
                  command=self._save_expert_yaml, width=11).pack(side="left", padx=2)

        tk.Label(
            expert_btn_row,
            text="Run Night uses this YAML text while Expert Mode is enabled.",
            fg="#444",
            anchor="w",
        ).pack(side="left", padx=10)

        self._expert_text = ScrolledText(
            self._expert_frame,
            height=12,
            wrap="none",
            font=("Menlo", 9),
        )
        self._expert_text.pack(fill="x", expand=False)
        self._reload_expert_yaml(initial=True)

        # ── Status bar ────────────────────────────────────────────────────────
        self._statusbar = tk.Label(self, text="Ready.", anchor="w",
                                   relief="sunken", bd=1, font=("Courier", 9))
        self._statusbar.pack(fill="x", padx=10, pady=(0, 2))

        # ── Embedded Run Night panel (hidden until ▶ Run Night is clicked) ────
        self._runner_sep = ttk.Separator(self, orient="horizontal")
        self._runner_frame = RunNightFrame(self, on_hide=self._hide_runner,
                                           on_row_update=self._on_run_status_update)
        # not packed yet — shown on demand

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _on_outdir_edit(self, *_):
        """Flag that the user has manually edited the output dir."""
        self._outdir_user_edited = True

    def _on_date_change(self, *_):
        """Keep output dir in sync with the night date unless user overrode it."""
        if self._outdir_user_edited:
            return
        try:
            dt = datetime.datetime.strptime(self._date_var.get().strip(), "%Y-%m-%d")
            yymmdd = dt.strftime("%y%m%d")
        except ValueError:
            return
        # Suppress the edit flag while we do the auto-update
        self._outdir_user_edited = False
        self._outdir_var.set(f"./night_output/{yymmdd}")
        self._outdir_user_edited = False

    def _set_status(self, msg: str):
        self._statusbar.config(text=msg)
        self.update_idletasks()

    def _toggle_expert_mode(self):
        if self._expert_mode_var.get():
            self._expert_frame.pack(fill="x", padx=10, pady=(0, 4))
            self._set_status("Expert Mode enabled: editing simulate_params YAML in GUI.")
        else:
            self._expert_frame.pack_forget()
            self._set_status("Expert Mode disabled.")

    def _reload_expert_yaml(self, initial: bool = False):
        try:
            with open(_SIM_PARAMS) as fh:
                text = fh.read()
        except Exception as exc:
            if not initial:
                messagebox.showerror("Expert Mode", f"Cannot read {_SIM_PARAMS}:\n{exc}")
            return
        self._expert_text.delete("1.0", "end")
        self._expert_text.insert("1.0", text)
        if not initial:
            self._set_status("Expert Mode: reloaded simulate_params.yaml from disk.")

    def _validate_expert_yaml(self) -> bool:
        text = self._expert_text.get("1.0", "end")
        try:
            doc = yaml.safe_load(text)
        except Exception as exc:
            messagebox.showerror("YAML error", f"Invalid YAML:\n{exc}")
            return False
        if not isinstance(doc, dict):
            messagebox.showerror("YAML error", "simulate_params YAML must be a mapping (dictionary).")
            return False
        self._set_status("Expert Mode: YAML is valid.")
        return True

    def _save_expert_yaml(self):
        if not self._validate_expert_yaml():
            return
        text = self._expert_text.get("1.0", "end")
        try:
            with open(_SIM_PARAMS, "w") as fh:
                fh.write(text)
        except Exception as exc:
            messagebox.showerror("Save error", f"Cannot write {_SIM_PARAMS}:\n{exc}")
            return
        self._set_status(f"Expert Mode: saved {_SIM_PARAMS}.")

    def _base_params_for_run(self) -> str | None:
        """Return the base simulate_params path to use for Run Night."""
        if not self._expert_mode_var.get():
            return _SIM_PARAMS

        if not self._validate_expert_yaml():
            return None

        text = self._expert_text.get("1.0", "end")

        # Remove stale temp file from a previous run, if any.
        if self._expert_temp_params and os.path.exists(self._expert_temp_params):
            try:
                os.unlink(self._expert_temp_params)
            except OSError:
                pass

        fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="vroomm_expert_params_")
        os.close(fd)
        with open(tmp, "w") as fh:
            fh.write(text)
        self._expert_temp_params = tmp
        return tmp

    def _browse_outdir(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self._outdir_user_edited = True
            self._outdir_var.set(d)

    def _on_obs_change(self, _=None):
        key = self._obs_var.get()
        if key in OBS_DEFAULTS:
            obs = OBS_DEFAULTS[key]
            self._obs_lat_var.set(str(obs["lat"]))
            self._obs_lon_var.set(str(obs["lon"]))
            self._obs_alt_var.set(str(obs["alt"]))
            # For predefined observatories, lock coordinates to avoid accidental edits.
            self._obs_lat_entry.config(state="readonly")
            self._obs_lon_entry.config(state="readonly")
            self._obs_alt_entry.config(state="readonly")
        else:
            # If a custom observatory value is ever used, allow manual edits.
            self._obs_lat_entry.config(state="normal")
            self._obs_lon_entry.config(state="normal")
            self._obs_alt_entry.config(state="normal")

    def _get_obs(self) -> dict:
        return dict(
            lat  = float(self._obs_lat_var.get()),
            lon  = float(self._obs_lon_var.get()),
            alt  = float(self._obs_alt_var.get()),
            name = self._obs_var.get(),
        )

    def _night_start(self) -> datetime.datetime:
        return datetime.datetime.strptime(
            f"{self._date_var.get().strip()}T{self._start_var.get().strip()}",
            "%Y-%m-%dT%H:%M:%S")

    def _get_overhead(self) -> float:
        try:
            return max(0.0, float(self._overhead_var.get()))
        except (ValueError, AttributeError):
            return READOUT_S

    def _compute_timestamps(self):
        """Assign sequential UTC start times to every row."""
        overhead = self._get_overhead()
        t = self._night_start()
        for row in self._rows:
            row["utc_start"] = t
            if row["obs_type"] in ("DELAY", "SLEW"):
                t += datetime.timedelta(seconds=row["exp_s"])
            else:
                t += datetime.timedelta(seconds=row["exp_s"] + overhead)

    def _filename(self, row: dict) -> str:
        if row["obs_type"] in ("DELAY", "SLEW"):
            return "—"
        ts = row["utc_start"].strftime("%Y%m%dT%H%M%S")
        if row["obs_type"] == SCI_TYPE:
            safe = row["target"].replace(" ", "_").replace("/", "-")
            return f"{ts}_{safe}.fits"
        return f"{ts}_{row['obs_type']}.fits"

    # ── Table rendering ────────────────────────────────────────────────────────
    def _refresh_table(self):
        self._compute_timestamps()
        for item in self._tree.get_children():
            self._tree.delete(item)
        for i, row in enumerate(self._rows, 1):
            ts     = row["utc_start"].strftime("%H:%M:%S")
            lts    = format_local_time(row["utc_start"], self._get_obs())
            target = row.get("target", "—")
            ra     = f"{row['ra']:.4f}"  if row.get("ra")  is not None else "—"
            dec    = f"{row['dec']:.4f}" if row.get("dec") is not None else "—"
            if row["obs_type"] == SCI_TYPE:
                rv_sys = f"{row.get('rv_sys', 0.0):+.2f}"
                berv   = f"{row.get('berv', float('nan')):+.3f}" \
                         if np.isfinite(row.get("berv", float("nan"))) else "—"
                am_val = row.get("airmass", float("nan"))
                airmass_str = f"{am_val:.3f}" if np.isfinite(am_val) else "—"
            elif row["obs_type"] in ("DELAY", "SLEW"):
                rv_sys = berv = airmass_str = "—"
            else:
                rv_sys = berv = airmass_str = "—"
            if row["obs_type"] == SCI_TYPE:
                am_val = row.get("airmass", float("nan"))
                if not np.isfinite(am_val) or am_val > 3.0:
                    tag = "unschedulable"
                    opt = optimal_date_from_ra(row.get("ra", 0.0))
                    airmass_str = f"❌ optimal: {opt}"
                else:
                    tag = "science"
            elif row["obs_type"] in ("DELAY", "SLEW"):
                tag = "overhead"
            else:
                tag = "cal"
            if row["obs_type"] in ("DELAY", "SLEW"):
                overwrite_cell = ""
            else:
                fname = os.path.join(self._outdir_var.get(), self._filename(row))
                if os.path.exists(fname):
                    overwrite_cell = "☑" if row.get("overwrite", False) else "☐"
                else:
                    overwrite_cell = ""
            self._tree.insert("", "end", iid=str(i - 1),
                          values=(i, ts, lts, row["obs_type"], target,
                                      ra, dec, rv_sys, berv, airmass_str,
                                      row["exp_s"], row.get("fiber", "rect"),
                                      self._filename(row), overwrite_cell, "to do", ""),
                              tags=(tag,))

    def _on_tree_click(self, event):
        """Toggle the Overwrite checkbox when user clicks the Overwrite column."""
        col = self._tree.identify_column(event.x)   # '#1', '#2', …
        iid = self._tree.identify_row(event.y)
        if not iid or col != "#14":   # column 14 = Overwrite (1-based)
            return
        idx = int(iid)
        row = self._rows[idx]
        if row["obs_type"] in ("DELAY", "SLEW"):
            return
        row["overwrite"] = not row.get("overwrite", False)
        self._refresh_table()

    def _selected_index(self) -> int | None:
        sel = self._tree.selection()
        return int(sel[0]) if sel else None

    # ── Actions ────────────────────────────────────────────────────────────────
    def _add_slew(self):
        dlg = _OverheadDialog(self, title="Add Slew", row_type="SLEW", default_dur=300.0)
        if dlg.result is None:
            return
        self._rows.append(dict(
            obs_type="SLEW", target=dlg.result["label"] or "slew",
            ra=None, dec=None, pmra=0.0, pmdec=0.0, rv_sys=0.0,
            berv=float("nan"), exp_s=dlg.result["dur_s"], fiber="—",
            utc_start=self._night_start(),
        ))
        self._refresh_table()
        self._set_status(f"Added SLEW  {dlg.result['dur_s']:.0f} s")

    def _add_delay(self):
        dlg = _OverheadDialog(self, title="Add Delay", row_type="DELAY", default_dur=60.0)
        if dlg.result is None:
            return
        self._rows.append(dict(
            obs_type="DELAY", target=dlg.result["label"] or "delay",
            ra=None, dec=None, pmra=0.0, pmdec=0.0, rv_sys=0.0,
            berv=float("nan"), exp_s=dlg.result["dur_s"], fiber="—",
            utc_start=self._night_start(),
        ))
        self._refresh_table()
        self._set_status(f"Added DELAY  {dlg.result['dur_s']:.0f} s")

    def _add_cal(self):
        dlg = AddCalDialog(self)
        if dlg.result is None:
            return
        r    = dlg.result
        reps = r.pop("repeats")
        for _ in range(reps):
            self._rows.append(dict(
                obs_type  = r["obs_type"],
                target    = "—",
                ra=None, dec=None, pmra=0.0, pmdec=0.0, rv_sys=0.0,
                berv      = float("nan"),
                exp_s     = r["exp_s"],
                fiber     = r["fiber"],
                utc_start = self._night_start(),
            ))
        self._refresh_table()
        self._set_status(f"Added {reps}× {r['obs_type']}")

    def _add_science(self):
        obs = self._get_obs()
        dlg = AddScienceDialog(self, obs)
        if dlg.result is None:
            return
        r    = dlg.result
        reps = r.pop("repeats")
        overhead = self._get_overhead()

        # Determine the start time for the first new exposure
        self._compute_timestamps()
        if self._rows:
            last = self._rows[-1]
            if last["obs_type"] in ("DELAY", "SLEW"):
                next_start = last["utc_start"] + datetime.timedelta(seconds=last["exp_s"])
            else:
                next_start = last["utc_start"] + datetime.timedelta(
                    seconds=last["exp_s"] + overhead)
        else:
            next_start = self._night_start()

        self._set_status(f"Computing BERV + airmass for {r['target']} …")
        for _ in range(reps):
            t_mid   = next_start + datetime.timedelta(seconds=r["exp_s"] / 2.0)
            berv    = compute_berv(r["ra"], r["dec"], r["pmra"], r["pmdec"], t_mid, obs)
            airmass = compute_airmass(r["ra"], r["dec"], t_mid, obs)
            self._rows.append(dict(
                obs_type  = SCI_TYPE,
                target    = r["target"],
                ra        = r["ra"],
                dec       = r["dec"],
                pmra      = r["pmra"],
                pmdec     = r["pmdec"],
                rv_sys    = r["rv_sys"],
                grp_mag   = r.get("grp_mag", np.nan),
                teff      = r.get("teff", np.nan),
                logg      = r.get("logg", np.nan),
                vsini     = r.get("vsini", np.nan),
                berv      = berv,
                airmass   = airmass,
                exp_s     = r["exp_s"],
                fiber     = "rect",
                utc_start = next_start,
            ))
            next_start += datetime.timedelta(seconds=r["exp_s"] + overhead)

        self._refresh_table()
        berv0 = self._rows[-reps]["berv"]
        am0   = self._rows[-reps]["airmass"]
        am_str = f"{am0:.3f}" if np.isfinite(am0) else "below horizon"
        self._set_status(
            f"Added {reps}× {r['target']}   BERV = {berv0:+.3f} km/s   airmass = {am_str}")

    def _delete_row(self):
        idx = self._selected_index()
        if idx is None:
            return
        del self._rows[idx]
        self._refresh_table()
        self._set_status("Row deleted.")

    def _move_up(self):
        idx = self._selected_index()
        if idx is None or idx == 0:
            return
        self._rows[idx - 1], self._rows[idx] = self._rows[idx], self._rows[idx - 1]
        self._refresh_table()
        self._tree.selection_set(str(idx - 1))

    def _move_down(self):
        idx = self._selected_index()
        if idx is None or idx >= len(self._rows) - 1:
            return
        self._rows[idx + 1], self._rows[idx] = self._rows[idx], self._rows[idx + 1]
        self._refresh_table()
        self._tree.selection_set(str(idx + 1))

    def _clear_all(self):
        if messagebox.askyesno("Clear sequence", "Remove all rows?"):
            self._rows.clear()
            self._refresh_table()
            self._set_status("Sequence cleared.")

    def _recompute_berv(self):
        obs = self._get_obs()
        self._compute_timestamps()
        n = 0
        for row in self._rows:
            if row["obs_type"] != SCI_TYPE or row.get("ra") is None:
                continue
            t_mid = row["utc_start"] + datetime.timedelta(seconds=row["exp_s"] / 2.0)
            row["berv"]    = compute_berv(
                row["ra"], row["dec"], row.get("pmra", 0.0),
                row.get("pmdec", 0.0), t_mid, obs)
            row["airmass"] = compute_airmass(row["ra"], row["dec"], t_mid, obs)
            n += 1
        self._refresh_table()
        self._set_status(f"BERV + airmass recomputed for {n} science exposure(s).")

    # ── Export ─────────────────────────────────────────────────────────────────
    def _export_yaml(self):
        if not self._rows:
            messagebox.showwarning("Empty", "No observations to export.")
            return
        self._compute_timestamps()
        out_dir = self._outdir_var.get()
        seq = []
        for row in self._rows:
            entry = dict(
                utc_start = row["utc_start"].strftime("%Y-%m-%dT%H:%M:%S"),
                obs_type  = row["obs_type"],
                dur_s     = row["exp_s"],
            )
            if row["obs_type"] in ("DELAY", "SLEW"):
                entry["note"] = row.get("target", "")
                seq.append(entry)
                continue
            fname = os.path.join(out_dir, self._filename(row))
            entry["exp_s"]  = row["exp_s"]
            entry["fiber"]  = row.get("fiber", "rect")
            entry["output"] = fname
            if row["obs_type"] == SCI_TYPE:
                am = row.get("airmass", float("nan"))
                entry.update(dict(
                    target   = row["target"],
                    ra_deg   = round(row["ra"],  6),
                    dec_deg  = round(row["dec"], 6),
                    pmra     = round(row.get("pmra",  0.0), 4),
                    pmdec    = round(row.get("pmdec", 0.0), 4),
                    rv_sys   = round(row.get("rv_sys", 0.0), 4),
                    grp_mag  = (round(float(row.get("grp_mag")), 3)
                                if np.isfinite(row.get("grp_mag", np.nan)) else None),
                    teff     = (round(float(row.get("teff")))
                                if np.isfinite(row.get("teff", np.nan)) else None),
                    logg     = (round(float(row.get("logg")), 2)
                                if np.isfinite(row.get("logg", np.nan)) else None),
                    vsini    = (round(float(row.get("vsini")), 2)
                                if np.isfinite(row.get("vsini", np.nan)) else None),
                    berv_kms = round(row.get("berv", 0.0),  5),
                    airmass  = round(am, 4) if np.isfinite(am) else None,
                ))
            else:
                entry["lamp_type"] = row["obs_type"].lower()
            seq.append(entry)

        obs = self._get_obs()
        plan = dict(
            observatory = obs,
            night_start = self._night_start().strftime("%Y-%m-%dT%H:%M:%S"),
            output_dir  = out_dir,
            sequence    = seq,
        )

        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*")],
            initialfile="night_plan.yaml",
            title="Export night plan",
        )
        if not path:
            return
        with open(path, "w") as fh:
            yaml.dump(plan, fh, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)
        self._set_status(f"Exported: {path}")
        return path

    def _run_night(self):
        """Build plan dict and load it into the embedded RunNightFrame."""
        if not self._rows:
            messagebox.showwarning("Empty", "No observations to export.")
            return
        base_params = self._base_params_for_run()
        if base_params is None:
            return
        self._compute_timestamps()
        # Build plan dict (same logic as _export_yaml but without file dialog)
        out_dir = self._outdir_var.get()
        seq = []
        for j, row in enumerate(self._rows):
            entry = dict(
                _planner_row_idx = j,
                utc_start = row["utc_start"].strftime("%Y-%m-%dT%H:%M:%S"),
                obs_type  = row["obs_type"],
                dur_s     = row["exp_s"],
            )
            if row["obs_type"] in ("DELAY", "SLEW"):
                entry["note"] = row.get("target", "")
                seq.append(entry)
                continue
            fname = os.path.join(out_dir, self._filename(row))
            entry["exp_s"]      = row["exp_s"]
            entry["fiber"]      = row.get("fiber", "rect")
            entry["output"]     = fname
            entry["overwrite"]  = row.get("overwrite", False)
            if row["obs_type"] == SCI_TYPE:
                am = row.get("airmass", float("nan"))
                entry.update(dict(
                    target   = row["target"],
                    ra_deg   = round(row["ra"],  6),
                    dec_deg  = round(row["dec"], 6),
                    pmra     = round(row.get("pmra",  0.0), 4),
                    pmdec    = round(row.get("pmdec", 0.0), 4),
                    rv_sys   = round(row.get("rv_sys", 0.0), 4),
                    grp_mag  = (round(float(row.get("grp_mag")), 3)
                                if np.isfinite(row.get("grp_mag", np.nan)) else None),
                    teff     = (round(float(row.get("teff")))
                                if np.isfinite(row.get("teff", np.nan)) else None),
                    logg     = (round(float(row.get("logg")), 2)
                                if np.isfinite(row.get("logg", np.nan)) else None),
                    vsini    = (round(float(row.get("vsini")), 2)
                                if np.isfinite(row.get("vsini", np.nan)) else None),
                    berv_kms = round(row.get("berv", 0.0),  5),
                    airmass  = round(am, 4) if np.isfinite(am) else None,
                ))
            else:
                entry["lamp_type"] = row["obs_type"].lower()
            seq.append(entry)
        obs = self._get_obs()
        plan = dict(
            observatory = obs,
            night_start = self._night_start().strftime("%Y-%m-%dT%H:%M:%S"),
            output_dir  = out_dir,
            sequence    = seq,
        )
        self._set_status(f"Starting night — {len(seq)} exposure(s).")
        self._runner_frame.set_base_params_path(base_params)
        self._runner_frame.load_plan_dict(plan)
        self._runner_sep.pack(fill="x", padx=0, pady=0)
        self._runner_frame.pack(fill="x")
        self._runner_frame._start()


    def _hide_runner(self):
        self._runner_frame.pack_forget()
        self._runner_sep.pack_forget()

    def _on_run_status_update(self, row_idx: int, status: str, elapsed_s):
        """Called by RunNightFrame to update a row's Status column in place."""
        iid = str(row_idx)
        if not self._tree.exists(iid):
            return
        vals = list(self._tree.item(iid, "values"))
        if len(vals) < 16:
            return
        if status == "running":
            vals[14] = "▶ running"
            vals[15] = ""
            status_tag = "running"
        elif status == "done":
            vals[14] = "✓ done"
            vals[15] = f"{elapsed_s:.0f}" if elapsed_s is not None else ""
            status_tag = "done_ok"
            # reveal overwrite checkbox now the file exists
            row = self._rows[row_idx]
            if not row.get("obs_type", "") in ("DELAY", "SLEW"):
                vals[13] = "☐"   # default: don't overwrite on next run
                row["overwrite"] = False
        elif status == "skipped":
            vals[14] = "⊘ skipped"
            vals[15] = ""
            status_tag = "skipped"
        elif status == "error":
            vals[14] = "✗ error"
            vals[15] = ""
            status_tag = "run_error"
        else:
            return
        base_tags = self._tree.item(iid, "tags")
        base_tag  = base_tags[0] if base_tags else ""
        self._tree.item(iid, values=vals, tags=(base_tag, status_tag))
        self._tree.see(iid)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = NightPlannerApp()
    app.mainloop()
