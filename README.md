# VROOMM Simulator

Night planning, physically motivated detector synthesis, and batch execution for spectrograph simulation workflows.

This repository is the full local toolkit behind the VROOMM demo website in the docs folder.

## What You Can Do Here

- Build a night plan with calibrations + science targets in a GUI.
- Run the full sequence automatically and generate FITS/PNG frames.
- Simulate a detector image directly from optical design products.
- Generate the static website assets shown in docs.
- Publish docs to GitHub Pages.

## 60-Second Quick Start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Launch the planner.

~~~bash
cd vroommisator
python -m venv .venv
source .venv/bin/activate

pip install numpy scipy pyyaml matplotlib tqdm astropy astroquery barycorrpy pillow
python night_planner.py
~~~

If you only run one command day-to-day, run:

~~~bash
python night_planner.py
~~~

## Daily Workflow (Recommended)

### Step 1: Build the observing sequence

~~~bash
python night_planner.py
~~~

In the planner GUI you can:
- Set observatory and timing.
- Add calibration blocks (FLAT, THAR, UNE, FP, DARK, BIAS).
- Add science targets (with SIMBAD/Gaia enrichment when available).
- Save a night plan YAML.

### Step 2: Execute the plan

The planner can launch the runner automatically. You can also run it yourself:

~~~bash
python run_night.py path/to/night_plan.yaml
~~~

Or start without argument to open a file picker:

~~~bash
python run_night.py
~~~

### Step 3: Inspect outputs

Typical outputs:
- FITS detector frames in night_output/YYMMDD/
- PNG previews next to FITS outputs
- Optional root-level detector preview files (detector_sim.fits/.npy/.png)

## Single-Frame Simulation (Direct)

Run detector synthesis directly from parameter YAML:

~~~bash
python simulate_detector.py
~~~

Use a custom parameter file:

~~~bash
python simulate_detector.py --params simulate_params.yaml
~~~

Useful overrides:

~~~bash
python simulate_detector.py \
  --params simulate_params.yaml \
  --output-fits detector_sim.fits \
  --output-png detector_sim.png \
  --wave-step 0.002
~~~

## How 1D Spectrum Becomes 2D Detector Image

The simulator follows the same 4-step logic shown on the website:

1. Build detector-scale PSFs.
2. Build order traces from Zemax XY geometry.
3. Interpolate x(lambda), y(lambda) with cubic splines.
4. Accumulate shifted PSF stamps in a photon-conserving detector image.

The lecture plots used in docs are generated automatically by make_webpage_assets.py.

## Core Files

- night_planner.py: GUI planner and target enrichment.
- run_night.py: batch execution engine with progress UI.
- simulate_detector.py: physics-driven 2D detector generator.
- simulate_params.yaml: central simulation configuration.
- make_webpage_assets.py: builds docs/assets manifest, gallery, lecture figures.
- deploy_docs_github_pages.sh: publishes docs subtree to gh-pages.

## Configuration Summary

Most behavior is controlled in simulate_params.yaml:

- observatory: latitude, longitude, elevation.
- telescope: diameter and throughput.
- observation: exposure and time context.
- target/star: astrometry and brightness.
- spectrum_mode: synthetic, file, or model.
- sky/telluric: atmospheric emission and absorption.
- octagonal_fiber: second fiber behavior and offset.
- flatfield/lamp: calibration source injection.

## Build the Documentation Website Assets

The docs page reads from docs/assets/manifest.json and manifest.js.
Regenerate these after new simulations or screenshots:

~~~bash
python make_webpage_assets.py
~~~

This updates:
- copied assets for docs/assets
- detector statistics summary
- transmission snapshot summary
- generated gallery images
- lecture figures used by the website

## Publish Docs to GitHub Pages

~~~bash
bash deploy_docs_github_pages.sh
~~~

This script:
1. Rebuilds docs assets.
2. Ensures docs/.nojekyll exists.
3. Pushes docs as the root of gh-pages via git subtree.

## Data and Asset Layout

- assets/zemax_data: optical design XY maps and PSF files.
- assets/transmission: throughput curves and transmission utilities.
- assets/targets: per-target cached YAML metadata.
- docs: static website (index.html, styles.css, app.js, assets).
- night_output: generated nightly simulation products.

## Troubleshooting

- Missing SIMBAD/Gaia features: install astroquery and check network access.
- Missing BERV computation: install barycorrpy.
- Missing image support in GUI: install pillow.
- YAML parsing errors: install pyyaml and validate simulate_params.yaml syntax.
- Empty docs cards/gallery: run make_webpage_assets.py after generating outputs.

## Reproducibility Notes

- Keep simulate_params.yaml under version control for traceable runs.
- Keep generated night plans (YAML) alongside produced outputs.
- Include observatory, date, and calibration setup in commit messages for simulated nights.
