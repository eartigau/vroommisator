#!/usr/bin/env python3
"""
run_night.py — VROOMM batch night runner

Reads a night_plan.yaml produced by night_planner.py and simulates every
exposure (science, calibration) one by one, using simulate_detector.py.

Usage:
    python run_night.py [night_plan.yaml]
    python run_night.py                    # opens a file picker GUI

The GUI shows:
  • a progress bar
  • a live list of completed files
  • elapsed / estimated remaining time per frame and for the night
"""

import copy
import datetime
import os
import queue
import re
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import yaml

# ── locate assets relative to this file ────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_LOGO_PATH   = os.path.join(_HERE, "assets", "logo.png")
_BASE_PARAMS = os.path.join(_HERE, "simulate_params.yaml")

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ── thread-local stdout router ─────────────────────────────────────────────────
# Captures print() calls from worker threads and posts them to the GUI queue.
# When _local.queue is None (main thread / any non-worker thread) output goes
# to the real stdout unchanged.
_real_stdout = sys.stdout


class _WorkerStdout:
    """Proxy for sys.stdout that routes worker-thread output to the msg queue."""
    def __init__(self):
        self._local = threading.local()

    def write(self, text: str):
        q = getattr(self._local, 'queue', None)
        if q is not None:
            # skip bare newlines that would clutter the log
            if text and text.strip():
                q.put({"status": "log", "text": text.rstrip('\n')})
        else:
            _real_stdout.write(text)

    def flush(self):
        _real_stdout.flush()

    def isatty(self) -> bool:
        return False


_worker_stdout = _WorkerStdout()
sys.stdout = _worker_stdout


# ── thread-local stderr router (captures tqdm) ────────────────────────────────
_real_stderr = sys.stderr
_ANSI_RE     = re.compile(r'\x1b\[[0-9;]*[mKGJABCDH]')


class _WorkerStderr:
    """Captures worker-thread stderr (tqdm bars) and posts them to the queue."""
    def __init__(self):
        self._local = threading.local()

    def write(self, text: str):
        q = getattr(self._local, 'queue', None)
        if q is not None:
            clean = _ANSI_RE.sub('', text).strip('\r\n')
            if clean.strip():
                q.put({"status": "tqdm", "text": clean})
        else:
            _real_stderr.write(text)

    def flush(self):
        _real_stderr.flush()

    def isatty(self) -> bool:
        return False


_worker_stderr = _WorkerStderr()
sys.stderr = _worker_stderr

# ── import the simulator ───────────────────────────────────────────────────────
sys.path.insert(0, _HERE)
import simulate_detector as _sim   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_seconds(s: float) -> str:
    """Format a duration in seconds as h:mm:ss or mm:ss."""
    if not np.isfinite(s) or s < 0:
        return "—"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _obs_entries(plan: dict) -> list[dict]:
    """Return only entries that produce an output file (drop DELAY / SLEW)."""
    return [e for e in plan.get("sequence", [])
            if e.get("obs_type") not in ("DELAY", "SLEW")
            and e.get("output")]


def _build_temp_yaml(entry: dict, base_params_path: str) -> str:
    """
    Deep-copy the base simulate_params.yaml, override fields from the night-plan
    entry, write to a temp file, and return its path.
    """
    with open(base_params_path) as fh:
        p = yaml.safe_load(fh)

    # ── absolutize all relative paths so the temp file can live anywhere ──
    root = os.path.dirname(os.path.abspath(base_params_path))

    def _abs(val):
        if isinstance(val, str) and val and not os.path.isabs(val):
            return os.path.join(root, val)
        return val

    for key in ("psf_dir", "xy_table", "output_fits", "output_png"):
        if key in p:
            p[key] = _abs(p[key])
    for section in ("model", "sky", "telluric", "lamp", "flatfield"):
        if section in p and isinstance(p[section], dict):
            for subkey in ("cache_dir", "fits_path", "path"):
                if subkey in p[section]:
                    p[section][subkey] = _abs(p[section][subkey])
    if "spectrum_file" in p and isinstance(p["spectrum_file"], dict):
        if "path" in p["spectrum_file"]:
            p["spectrum_file"]["path"] = _abs(p["spectrum_file"]["path"])
    if "octagonal_fiber" in p and isinstance(p["octagonal_fiber"], dict):
        if "psf_dir" in p["octagonal_fiber"]:
            p["octagonal_fiber"]["psf_dir"] = _abs(p["octagonal_fiber"]["psf_dir"])

    obs_type = entry.get("obs_type", "SCIENCE")
    output   = os.path.abspath(entry["output"])
    os.makedirs(os.path.dirname(output), exist_ok=True)

    fiber_sel = str(entry.get("fiber", "both")).lower()
    rect_on = fiber_sel in ("rect", "both")
    oct_on  = fiber_sel in ("oct", "both")

    # ── output paths ──────────────────────────────────────────────────────
    p["output_fits"] = output
    base_png = os.path.splitext(output)[0] + ".png"
    p["output_png"]  = base_png

    # ── per-type overrides ────────────────────────────────────────────────
    if obs_type == "SCIENCE":
        # Populate target section from the night-plan entry so FITS OBJECT header
        # and RA/DEC etc. reflect the actual target, not the base-YAML placeholder.
        p["target"] = {
            **p.get("target", {}),
            "name"        : entry.get("target",  "Unknown"),
            "ra_deg"      : float(entry.get("ra_deg",  entry.get("ra",   0.0))),
            "dec_deg"     : float(entry.get("dec_deg", entry.get("dec",  0.0))),
            "pmra_masyr"  : float(entry.get("pmra",   0.0)),
            "pmdec_masyr" : float(entry.get("pmdec",  0.0)),
            "rv_sys_kms"  : float(entry.get("rv_sys",  0.0)),
        }

        # Doppler shift: star RV in topocentric frame = rv_sys − BERV
        rv_sys   = float(entry.get("rv_sys",   0.0))
        berv_kms = float(entry.get("berv_kms", 0.0))
        p["observation"] = p.get("observation", {})
        p["observation"]["rv_kms"] = rv_sys - berv_kms

        # Telluric airmass
        am = entry.get("airmass")
        if am is not None and np.isfinite(float(am)):
            tell = p.get("telluric", {})
            am_ref  = float(tell.get("airmass",  1.2))
            wod_ref = float(tell.get("water_od", 1.5))
            pwv_scale = wod_ref / max(am_ref, 0.01)
            tell["airmass"]  = float(am)
            tell["water_od"] = round(float(am) * pwv_scale, 4)
            p["telluric"] = tell

        # disable flat / lamp
        p["flatfield"] = {**p.get("flatfield", {}), "enabled": False}
        p["lamp"]      = {**p.get("lamp",      {}), "enabled": False}

        # If SIMBAD provided stellar parameters for this target, override model
        # settings for this exposure (rounded to practical values).
        teff = entry.get("teff")
        logg = entry.get("logg")
        vsini = entry.get("vsini")
        if teff is not None:
            try:
                teff_f = float(teff)
                if np.isfinite(teff_f) and teff_f > 0:
                    p["spectrum_mode"] = "model"
                    p["model"] = {**p.get("model", {}),
                                  "teff": int(round(teff_f / 100.0) * 100)}
            except (TypeError, ValueError):
                pass
        if logg is not None:
            try:
                logg_f = float(logg)
                if np.isfinite(logg_f):
                    p["spectrum_mode"] = "model"
                    p["model"] = {**p.get("model", {}),
                                  "logg": round(logg_f * 2.0) / 2.0}
            except (TypeError, ValueError):
                pass
        if vsini is not None:
            try:
                vsini_f = float(vsini)
                if np.isfinite(vsini_f) and vsini_f >= 0.0:
                    p["star"] = {**p.get("star", {}),
                                 "vsini_kms": round(vsini_f, 2)}
            except (TypeError, ValueError):
                pass

        # Per-target photometric normalization: use Gaia RP (Grp) magnitude
        # from SIMBAD/manual entry as the stellar brightness scalar.
        grp_mag = entry.get("grp_mag")
        if grp_mag is not None:
            try:
                grp_f = float(grp_mag)
                if np.isfinite(grp_f):
                    p["star"] = {**p.get("star", {}),
                                 "R_mag": round(grp_f, 3),
                                 "mag_band": "grp"}
            except (TypeError, ValueError):
                pass

        # Science fiber routing: the chosen fiber carries the stellar spectrum;
        # the other fiber sees sky only.  Both fibers always see sky emission.
        science_fiber = entry.get("fiber", "rect")
        p["science_fiber"] = science_fiber
        # Always enable the octagonal fiber so it receives sky
        p["octagonal_fiber"] = {
            **p.get("octagonal_fiber", {}),
            "enabled": True,
        }

    elif obs_type in ("FLAT",):
        p["flatfield"] = {**p.get("flatfield", {}),
                          "enabled": True, "rect_fiber": rect_on, "oct_fiber": oct_on}
        p["lamp"]      = {**p.get("lamp",      {}), "enabled": False}
        p["telluric"]  = {**p.get("telluric",  {}), "enabled": False}
        p["sky"]       = {**p.get("sky",       {}), "enabled": False}
        p["spectrum_mode"] = "synthetic"

    elif obs_type in ("THAR", "UNE", "FP"):
        lamp_type = "thar" if obs_type == "THAR" else ("une" if obs_type == "UNE" else "fp")
        p["lamp"]     = {**p.get("lamp", {}),
                         "enabled": True, "rect_fiber": rect_on, "oct_fiber": oct_on,
                         "type": lamp_type}
        p["flatfield"]= {**p.get("flatfield", {}), "enabled": False}
        p["telluric"] = {**p.get("telluric",  {}), "enabled": False}
        p["sky"]      = {**p.get("sky",       {}), "enabled": False}
        p["observation"] = {**p.get("observation", {}), "rv_kms": 0.0}
        p["spectrum_mode"] = "synthetic"

    elif obs_type in ("DARK", "BIAS"):
        p["flatfield"] = {**p.get("flatfield", {}), "enabled": False}
        p["lamp"]      = {**p.get("lamp",      {}), "enabled": False}
        p["telluric"]  = {**p.get("telluric",  {}), "enabled": False}
        p["sky"]       = {**p.get("sky",       {}), "enabled": False}
        p["spectrum_mode"] = "synthetic"
        p["observation"]   = {**p.get("observation", {}),
                               "R_mag": 99.0,
                               "rv_kms": 0.0}

    # ── exposure time ──────────────────────────────────────────────────────
    exp_s = float(entry.get("exp_s", entry.get("dur_s", 300.0)))
    p["observation"] = {**p.get("observation", {}), "exp_s": exp_s}

    # Temp file can live anywhere — all paths are now absolute
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="vroomm_run_")
    os.close(fd)
    with open(tmp, "w") as fh:
        yaml.dump(p, fh, allow_unicode=True, sort_keys=False,
                  default_flow_style=False)
    return tmp


def _run_one(entry: dict, base_params: str, msg_q: queue.Queue):
    """
    Simulate one exposure.  Runs in a worker thread.
    Posts status dicts to msg_q:
      {"status": "start",    "label": str}
      {"status": "done",     "output": str, "elapsed_s": float}
      {"status": "error",    "label": str,  "exc": str}
    """
    label   = (entry.get("target") or entry.get("obs_type", "?")).strip()
    row_idx = entry.get("_planner_row_idx")
    _worker_stdout._local.queue = msg_q   # route this thread's prints to GUI
    _worker_stderr._local.queue = msg_q   # route tqdm output to GUI
    msg_q.put({"status": "start", "label": label, "row_idx": row_idx})
    # Skip if output already exists and overwrite not requested
    output_path = entry.get("output", "")
    if output_path and os.path.exists(os.path.abspath(output_path)) \
            and not entry.get("overwrite", False):
        print(f"  Skipping — output already exists: {os.path.basename(output_path)}")
        msg_q.put({"status": "skipped", "label": label, "row_idx": row_idx})
        return
    tmp = None
    t0  = time.monotonic()
    try:
        tmp = _build_temp_yaml(entry, base_params)
        kwargs     = _sim.load_params(tmp)
        oct_conf   = kwargs.pop("oct_conf",  None)
        fits_meta  = kwargs.pop("fits_meta", None)
        output_fits = kwargs.get("output_fits")
        output_png  = kwargs.get("output_png")

        if oct_conf is not None:
            img     = _sim.simulate_detector(
                **{**kwargs, "output_fits": None, "output_png": None})
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
            img_oct = _sim.simulate_detector(**oct_kwargs)
            img = img + img_oct
        else:
            img = _sim.simulate_detector(
                **{**kwargs, "output_fits": None, "output_png": None})

        if output_fits:
            _sim._save_fits(img, output_fits, fits_meta)
        if output_png:
            _sim._save_preview(img, output_png)

        elapsed = time.monotonic() - t0
        msg_q.put({"status": "done",
                   "output": output_fits or output_png or "—",
                   "elapsed_s": elapsed,
                   "label": label,
                   "row_idx": row_idx})
    except Exception as exc:
        import traceback
        msg_q.put({"status": "error",
                   "label": label,
                   "exc": traceback.format_exc(),
                   "row_idx": row_idx})
    finally:
        _worker_stdout._local.queue = None   # restore normal stdout for this thread
        _worker_stderr._local.queue = None   # restore normal stderr
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# Core embeddable frame
# ─────────────────────────────────────────────────────────────────────────────

class RunNightFrame(tk.Frame):
    """
    Core run-night UI: progress bar, file list, start/stop controls.
    Can be embedded inside another window (e.g. NightPlannerApp) or used
    standalone via the thin RunNightApp wrapper.

    Parameters
    ----------
    master    : parent widget
    plan_path : optional YAML file to load immediately
    on_hide   : optional callback invoked when the "✕ Close runner" button
                is clicked (pass None to suppress that button)
    """

    def __init__(self, master, plan_path: str | None = None,
                 on_hide=None, on_row_update=None):
        super().__init__(master)

        self._on_hide       = on_hide
        self._on_row_update = on_row_update

        self._plan_path   = plan_path
        self._plan        = None
        self._entries     = []
        self._n_total     = 0
        self._n_done      = 0
        self._running     = False
        self._stop_flag   = threading.Event()
        self._msg_q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._t_night_start: float | None = None
        self._elapsed_per_frame: list[float] = []
        self._base_params_path = _BASE_PARAMS

        self._build_ui()

        if plan_path:
            self._load_plan(plan_path)

    def set_base_params_path(self, path: str):
        """Override the simulate_params YAML used for per-exposure temp files."""
        if not path or not os.path.exists(path):
            raise ValueError(f"Base params file not found: {path}")
        self._base_params_path = path

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Summary bar ───────────────────────────────────────────────────────
        self._summary_var = tk.StringVar(value="No plan loaded.")
        tk.Label(self, textvariable=self._summary_var, anchor="w",
                 font=("TkDefaultFont", 10, "italic"),
                 fg="#555").pack(fill="x", padx=12, pady=(4, 2))

        # Main body: left controls + right console (full height)
        body = tk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 2))
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        # ── Progress section ──────────────────────────────────────────────────
        prog_frame = tk.LabelFrame(left, text="Progress", padx=10, pady=6)
        prog_frame.pack(fill="x", pady=4)

        self._pbar = ttk.Progressbar(prog_frame, mode="determinate",
                                     length=500, maximum=100)
        self._pbar.pack(fill="x", pady=(0, 4))

        timing = tk.Frame(prog_frame)
        timing.pack(fill="x")
        self._status_var  = tk.StringVar(value="Idle.")
        self._elapsed_var = tk.StringVar(value="")
        self._eta_var     = tk.StringVar(value="")

        tk.Label(timing, textvariable=self._status_var, anchor="w",
                 width=50).pack(side="left")
        tk.Label(timing, text="Elapsed:").pack(side="left", padx=(12, 2))
        tk.Label(timing, textvariable=self._elapsed_var,
                 width=8, anchor="w").pack(side="left")
        tk.Label(timing, text="  ETA:").pack(side="left", padx=(8, 2))
        tk.Label(timing, textvariable=self._eta_var,
                 width=8, anchor="w").pack(side="left")

        # ── Control buttons ───────────────────────────────────────────────────
        bf = tk.Frame(left)
        bf.pack(fill="x", pady=(0, 4))

        self._btn_stop = tk.Button(bf, text="⏹  Stop",
                                   command=self._stop,
                                   bg="#f8d7da", width=10,
                                   state="disabled")
        self._btn_stop.pack(side="left", padx=4)

        if self._on_hide is not None:
            tk.Button(bf, text="✕ Close runner", command=self._on_hide,
                      bg="#f8d7da", width=14).pack(side="right", padx=4)

        # ── Console log (right pane, full height) ────────────────────────────
        log_frame = tk.LabelFrame(body, text="Console output", padx=4, pady=4)
        log_frame.grid(row=0, column=1, sticky="nsew")

        log_btn_row = tk.Frame(log_frame)
        log_btn_row.pack(fill="x")
        tk.Button(log_btn_row, text="Clear log", font=("TkDefaultFont", 8),
                  command=self._clear_log).pack(side="right")
        tk.Button(log_btn_row, text="Copy log", font=("TkDefaultFont", 8),
                  command=self._copy_log).pack(side="right", padx=4)

        self._log = ScrolledText(
            log_frame, height=10,
            bg="#1e1e1e", fg="#d4d4d4",
            font=("Menlo", 9), wrap="word",
            state="disabled", insertbackground="white",
            selectbackground="#264f78",
        )
        self._log.pack(fill="both", expand=True)

        # tqdm order-progress bar (updates in real-time during simulation)
        tqdm_row = tk.Frame(log_frame, bg="#1e1e1e")
        tqdm_row.pack(fill="x", pady=(2, 0))
        self._tqdm_bar = ttk.Progressbar(tqdm_row, mode="determinate",
                                         maximum=100, length=1)
        self._tqdm_bar.pack(fill="x", pady=(0, 1))
        self._tqdm_var = tk.StringVar(value="")
        tk.Label(tqdm_row, textvariable=self._tqdm_var, anchor="w",
                 bg="#1e1e1e", fg="#569cd6",
                 font=("Menlo", 8)).pack(fill="x")

        self._log.tag_config("sep",      foreground="#569cd6", font=("Menlo", 9, "bold"))
        self._log.tag_config("error",    foreground="#f48771")   # red
        self._log.tag_config("warning",  foreground="#ce9178")   # orange
        self._log.tag_config("ok",       foreground="#4ec9b0")   # green
        self._log.tag_config("download", foreground="#dcdcaa")   # yellow
        self._log.tag_config("indent",   foreground="#c8c8c8")   # light gray (params)
        self._log.tag_config("normal",   foreground="#d4d4d4")   # gray
        self._log.tag_config("num",      foreground="#569cd6")   # blue — always on top
        self._log.tag_raise("num")   # num wins over any base-line tag

        # ── Status bar ────────────────────────────────────────────────────────
        self._statusbar = tk.Label(left, text="Ready.", anchor="w",
                       relief="sunken", bd=1, font=("Courier", 9))
        self._statusbar.pack(fill="x", pady=(0, 4))

    # ── Plan loading ───────────────────────────────────────────────────────────
    def _load_plan(self, path: str):
        try:
            with open(path) as fh:
                plan = yaml.safe_load(fh)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return
        self._plan_path = path
        self.load_plan_dict(plan)
        self._statusbar.config(text=f"Loaded: {path}")

    def load_plan_dict(self, plan: dict):
        """Load a plan directly from a dict (no file I/O needed)."""
        self._plan    = plan
        self._entries = _obs_entries(plan)
        self._n_total = len(self._entries)
        self._n_done  = 0
        self._pbar["value"] = 0

        obs  = plan.get("observatory", {})
        date = plan.get("night_start", "")[:10]
        sci  = sum(1 for e in self._entries if e.get("obs_type") == "SCIENCE")
        cal  = self._n_total - sci
        self._summary_var.set(
            f"  {date}  |  Observatory: {obs.get('name', '?')}  |  "
            f"{self._n_total} exposure(s): {sci} science + {cal} calibration"
        )
        self._status_var.set(f"Plan loaded.  {self._n_total} exposure(s) ready.")

    # ── Controls ───────────────────────────────────────────────────────────────
    def _start(self):
        if not self._entries:
            messagebox.showwarning("No plan", "Load a night plan first.")
            return
        if self._running:
            return
        self._running = True
        self._stop_flag.clear()
        self._n_done = 0
        self._elapsed_per_frame = []
        self._t_night_start = time.monotonic()
        self._btn_stop.config(state="normal")
        self._worker = threading.Thread(target=self._run_all, daemon=True)
        self._worker.start()
        self._poll()

    def _stop(self):
        self._stop_flag.set()
        self._status_var.set("Stop requested — finishing current frame …")
        self._btn_stop.config(state="disabled")

    def _clear_list(self):
        pass  # list removed; status is reflected in the planner treeview

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    def _copy_log(self):
        text = self._log.get("1.0", "end").rstrip("\n")
        self._log.clipboard_clear()
        self._log.clipboard_append(text)
        self._log.update()   # make it available immediately

    # regex that matches standalone numbers (int / float / sci-notation / %)
    _NUM_RE = re.compile(
        r'(?<![\w.])'
        r'[-+]?\d+(?:[,.]\d+)?(?:[eE][+-]?\d+)?(?:\s*%)?'
        r'(?![\w])'
    )

    @staticmethod
    def _classify(text: str) -> str:
        """Return the base colour tag for a log line."""
        tl = text.lower().strip()
        if any(k in tl for k in ("error", "traceback", "exception", "failed",
                                  "[error", "errno")):
            return "error"
        if any(k in tl for k in ("✓", "saved", "complete", "cached",
                                  "loaded from cache", "wrote")):
            return "ok"
        if any(k in tl for k in ("fetching", "downloading", "loading",
                                  "fetch", "download", "reading",
                                  "psf library", "xy table")):
            return "download"
        if "warn" in tl or "[warn]" in tl:
            return "warning"
        if text.startswith("  ") or text.startswith("\t"):
            return "indent"
        return "normal"

    def _log_append(self, text: str, tag: str | None = None):
        """Append one richly-coloured line to the console log (main-thread only)."""
        base = tag if tag is not None else self._classify(text)
        self._log.config(state="normal")
        # Insert text in segments; overlay 'num' tag on every numeric token
        pos = 0
        for m in self._NUM_RE.finditer(text):
            s, e = m.start(), m.end()
            if s > pos:
                self._log.insert("end", text[pos:s], base)
            self._log.insert("end", text[s:e], (base, "num"))
            pos = e
        if pos < len(text):
            self._log.insert("end", text[pos:], base)
        self._log.insert("end", "\n", base)
        self._log.see("end")
        self._log.config(state="disabled")

    # ── Worker ─────────────────────────────────────────────────────────────────
    def _run_all(self):
        for i, entry in enumerate(self._entries):
            if self._stop_flag.is_set():
                self._msg_q.put({"status": "stopped", "at": i})
                return
            _run_one(entry, self._base_params_path, self._msg_q)
        self._msg_q.put({"status": "finished"})

    # ── Polling / GUI update ───────────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                msg = self._msg_q.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

        if self._running and self._t_night_start is not None:
            elapsed = time.monotonic() - self._t_night_start
            self._elapsed_var.set(_fmt_seconds(elapsed))
            if self._elapsed_per_frame and self._n_done < self._n_total:
                mean_s = np.mean(self._elapsed_per_frame)
                remain = (self._n_total - self._n_done) * mean_s
                self._eta_var.set(_fmt_seconds(remain))

        if self._running:
            self.after(200, self._poll)

    def _handle_msg(self, msg: dict):
        status = msg["status"]

        if status == "tqdm":
            text = msg["text"]
            # parse  n/total  from tqdm line
            m = re.search(r'(\d+)/(\d+)', text)
            if m:
                n, total = int(m.group(1)), int(m.group(2))
                self._tqdm_bar["value"] = int(100 * n / max(total, 1))
            # clean up block chars so the label stays readable
            label = re.sub(r'[|█░▏▎▍▌▋▊▉╸╺━ ]', ' ', text)
            label = re.sub(r'\s+', ' ', label).strip()
            self._tqdm_var.set(label)
            return

        if status == "log":
            self._log_append(msg["text"])   # auto-classified + blue numbers
            return

        if status == "start":
            self._status_var.set(f"Simulating  [{self._n_done + 1}/{self._n_total}]  "
                                 f"{msg['label']} …")
            self._statusbar.config(text=f"Running: {msg['label']}")
            # reset tqdm bar for this exposure
            self._tqdm_bar["value"] = 0
            self._tqdm_var.set("")
            # separator line in log
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._log_append(
                f"{'─'*60}  [{self._n_done+1}/{self._n_total}]  {msg['label']}  {ts}",
                "sep")
            if self._on_row_update is not None and msg.get("row_idx") is not None:
                self._on_row_update(msg["row_idx"], "running", None)

        elif status == "done":
            self._n_done += 1
            elapsed = msg["elapsed_s"]
            self._elapsed_per_frame.append(elapsed)
            pct = int(100 * self._n_done / max(self._n_total, 1))
            self._pbar["value"] = pct
            self._status_var.set(
                f"Done [{self._n_done}/{self._n_total}]  "
                f"{msg['label']}  ({elapsed:.0f} s)")
            self._log_append(f"✓ Done in {elapsed:.1f} s — {msg.get('output','')}", "ok")
            if self._on_row_update is not None and msg.get("row_idx") is not None:
                self._on_row_update(msg["row_idx"], "done", elapsed)

        elif status == "error":
            self._n_done += 1
            self._statusbar.config(text=f"ERROR on {msg['label']}: {msg['exc'][:120]}")
            self._log_append(f"✗ ERROR on {msg['label']}:", "error")
            for line in msg["exc"].splitlines():
                self._log_append("  " + line, "error")
            _real_stdout.write(f"\n[ERROR] {msg['label']}\n{msg['exc']}\n")
            if self._on_row_update is not None and msg.get("row_idx") is not None:
                self._on_row_update(msg["row_idx"], "error", None)

        elif status == "skipped":
            self._n_done += 1
            pct = int(100 * self._n_done / max(self._n_total, 1))
            self._pbar["value"] = pct
            self._log_append(
                f"⊘ Skipped (file exists) — {msg['label']}  [tick Overwrite to rerun]",
                "warning")
            if self._on_row_update is not None and msg.get("row_idx") is not None:
                self._on_row_update(msg["row_idx"], "skipped", None)

        elif status in ("finished", "stopped"):
            self._running = False
            self._btn_stop.config(state="disabled")
            self._tqdm_bar["value"] = 0
            self._tqdm_var.set("")
            total_s = (time.monotonic() - self._t_night_start
                       if self._t_night_start else 0)
            if status == "finished":
                self._pbar["value"] = 100
                msg_text = (f"Night complete — {self._n_done} exposure(s) in "
                            f"{_fmt_seconds(total_s)}")
                self._status_var.set(msg_text)
                self._eta_var.set("—")
                self._statusbar.config(text=msg_text)
            else:
                msg_text = (f"Stopped after {self._n_done}/{self._n_total} "
                            f"exposure(s)  ({_fmt_seconds(total_s)} elapsed)")
                self._status_var.set(msg_text)
                self._statusbar.config(text=msg_text)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RunNightApp(tk.Tk):
    """Thin standalone wrapper: adds logo banner + plan-file selector on top."""

    def __init__(self, plan_path: str | None = None):
        super().__init__()
        self.title("VROOMM — Run Night")
        self.minsize(900, 560)

        self._logo_img = None
        self._icon_img = None
        self._load_logo()

        # ── Logo banner ───────────────────────────────────────────────────────
        if self._logo_img is not None:
            banner = tk.Frame(self, bg="white", bd=0)
            banner.pack(fill="x")
            tk.Label(banner, image=self._logo_img, bg="white",
                     anchor="w").pack(side="left", padx=12, pady=4)

        # ── Plan selector ─────────────────────────────────────────────────────
        sel = tk.Frame(self, padx=10, pady=6)
        sel.pack(fill="x")
        tk.Label(sel, text="Night plan:").pack(side="left")
        self._plan_var = tk.StringVar(value=plan_path or "")
        tk.Entry(sel, textvariable=self._plan_var, width=60,
                 state="readonly").pack(side="left", padx=6)
        tk.Button(sel, text="Browse…", command=self._browse_plan).pack(side="left", padx=4)
        tk.Button(sel, text="Load",    command=self._on_load).pack(side="left", padx=4)

        # ── Core frame ────────────────────────────────────────────────────────
        self._runner = RunNightFrame(self, plan_path=plan_path)
        self._runner.pack(fill="both", expand=True)

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

    def _browse_plan(self):
        path = filedialog.askopenfilename(
            title="Select night plan",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*")],
        )
        if path:
            self._plan_var.set(path)
            self._runner._load_plan(path)

    def _on_load(self):
        p = self._plan_var.get().strip()
        if p:
            self._runner._load_plan(p)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plan_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = RunNightApp(plan_path)
    app.mainloop()
