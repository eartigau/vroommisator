function fmtNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  const num = Number(value);
  if (Math.abs(num) >= 1000 || Math.abs(num) < 0.01) {
    return num.toExponential(3);
  }
  return num.toFixed(4);
}

function fmtSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return `${Math.round(Number(value))} s`;
}

function addStat(container, label, value) {
  const card = document.createElement("div");
  card.className = "stat";

  const l = document.createElement("div");
  l.className = "label";
  l.textContent = label;

  const v = document.createElement("div");
  v.className = "value";
  v.textContent = value;

  card.appendChild(l);
  card.appendChild(v);
  container.appendChild(card);
}

async function loadManifest() {
  const statsGrid = document.getElementById("statsGrid");
  const methodsGrid = document.getElementById("methodsGrid");
  const galleryGrid = document.getElementById("galleryGrid");
  const lectureGrid = document.getElementById("lectureGrid");
  const transmission = document.getElementById("transmissionSummary");
  const guiShot = document.getElementById("guiShot");
  const guiShotHint = document.getElementById("guiShotHint");

  guiShot.addEventListener("error", () => {
    guiShotHint.textContent = "Screenshot missing: put file at docs/assets/gui_screenshot.png";
  });

  try {
    let m = window.VROOMM_MANIFEST;
    if (!m) {
      const res = await fetch("assets/manifest.json", { cache: "no-store" });
      if (!res.ok) {
        throw new Error("manifest not found");
      }
      m = await res.json();
    }

    const d = m.detector_stats || {};
    const fits = d.fits || null;
    const statsIntro = document.getElementById("statsIntro");

    if (fits) {
      const [ny, nx] = fits.shape || [];
      addStat(statsGrid, "Detector size", ny && nx ? `${ny} × ${nx} px` : "n/a");
      addStat(statsGrid, "Total detected photons", fmtNumber(fits.sum));
      addStat(statsGrid, "Peak photons / pixel", fmtNumber(fits.max));
      addStat(statsGrid, "Illuminated pixels",
        fits.illuminated_frac != null ? `${(fits.illuminated_frac * 100).toFixed(1)}%` : "n/a");
      addStat(statsGrid, "p90 among lit pixels", fmtNumber(fits.p90));
      addStat(statsGrid, "p99 among lit pixels", fmtNumber(fits.p99));
    } else {
      addStat(statsGrid, "Detector stats", "n/a — run simulate_detector.py first");
    }

    if (statsIntro) {
      const src = (m.methods || {}).source || {};
      const obsv = (m.methods || {}).observation || {};
      const env = (m.methods || {}).environment || {};
      if (fits && (m.methods && !m.methods.error)) {
        statsIntro.textContent =
          `This is a single simulated exposure: a Teff=${src.model_teff ?? "?"} K, ` +
          `${fmtNumber(src.star_mag)} mag (${(src.star_mag_band || "R").toUpperCase()}) star, ` +
          `${fmtSeconds(obsv.exposure_s)} exposure` +
          (env.sky_enabled ? ", sky emission on" : "") +
          (env.telluric_enabled ? `, telluric absorption at airmass ${fmtNumber(env.telluric_airmass)}` : "") +
          ". Numbers below describe the resulting 4096×4096 photon-count image.";
      } else {
        statsIntro.textContent = "Run simulate_detector.py then python make_webpage_assets.py to populate this section.";
      }
    }

    const t = m.transmission_summary || {};
    if (t.error) {
      addStat(transmission, "Transmission", "error: " + t.error);
    } else {
      addStat(transmission, "Lambda range", `${fmtNumber(t.wavelength_min_nm)}-${fmtNumber(t.wavelength_max_nm)} nm`);
      addStat(transmission, "Peak", `${fmtNumber(t.transmission_max)} @ ${fmtNumber(t.peak_nm)} nm`);
      addStat(transmission, "T(400)/T(500)",
        t.samples ? fmtNumber(Number(t.samples["400"]) / Number(t.samples["500"])) : "n/a");
      addStat(transmission, "Blue sample T(380)", t.samples ? fmtNumber(t.samples["380"]) : "n/a");
      addStat(transmission, "Red sample T(900)", t.samples ? fmtNumber(t.samples["900"]) : "n/a");
    }

    const meth = m.methods || {};
    if (meth.error) {
      addStat(methodsGrid, "Methods", "error: " + meth.error);
    } else {
      const obs = meth.observatory || {};
      const tel = meth.telescope || {};
      const src = meth.source || {};
      const env = meth.environment || {};
      const smp = meth.sampling || {};

      addStat(methodsGrid, "Observatory", obs.name || "n/a");
      addStat(methodsGrid, "Coordinates", `${fmtNumber(obs.lat_deg)}, ${fmtNumber(obs.lon_deg)}`);
      addStat(methodsGrid, "Telescope", `${tel.name || "n/a"} (${fmtNumber(tel.diameter_m)} m)`);
      addStat(methodsGrid, "Peak throughput", fmtNumber(tel.peak_throughput));
      addStat(methodsGrid, "Exposure", fmtSeconds((meth.observation || {}).exposure_s));
      addStat(methodsGrid, "Spectrum mode", src.spectrum_mode || "n/a");
      addStat(methodsGrid, "Stellar model", `Teff=${src.model_teff ?? "n/a"}, logg=${src.model_logg ?? "n/a"}`);
      addStat(methodsGrid, "Magnitude", `${fmtNumber(src.star_mag)} (${(src.star_mag_band || "R").toUpperCase()})`);
      addStat(methodsGrid, "vsini [km/s]", fmtNumber(src.star_vsini_kms));
      addStat(methodsGrid, "Sampling [pix frac]", fmtNumber(smp.wave_step_pix_frac));
      addStat(methodsGrid, "Sky enabled", String(Boolean(env.sky_enabled)));
      addStat(methodsGrid, "Telluric enabled", String(Boolean(env.telluric_enabled)));
      addStat(methodsGrid, "Telluric airmass", fmtNumber(env.telluric_airmass));
    }

    const gallery = Array.isArray(m.gallery) ? m.gallery : [];
    if (!gallery.length) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = "No gallery images yet. Generate outputs and run python make_webpage_assets.py.";
      galleryGrid.appendChild(empty);
    } else {
      for (const item of gallery) {
        const card = document.createElement("figure");
        card.className = "gallery-item";

        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = item.web_path || item.dest || "";
        img.alt = item.title || "Generated detector image";

        const cap = document.createElement("figcaption");
        cap.textContent = `${item.title || "image"} (${Math.round((item.bytes || 0) / 1024)} KB)`;

        card.appendChild(img);
        card.appendChild(cap);
        galleryGrid.appendChild(card);
      }
    }

    const lecture = Array.isArray(m.lecture_plots) ? m.lecture_plots : [];
    if (!lectureGrid) {
      // Lecture is now primarily static section content in index.html.
    } else if (!lecture.length) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = "No lecture figures yet. Run python make_webpage_assets.py.";
      lectureGrid.appendChild(empty);
    } else {
      const hasStatic = document.querySelector(".static-lecture") !== null;
      if (hasStatic) {
        // Static lecture figures are already in index.html for file:// robustness.
        return;
      }
      for (const item of lecture) {
        const card = document.createElement("figure");
        card.className = "lecture-item";

        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = item.web_path || item.dest || "";
        img.alt = item.title || "Lecture plot";

        const cap = document.createElement("figcaption");
        cap.textContent = item.title || "Lecture figure";

        card.appendChild(img);
        card.appendChild(cap);
        lectureGrid.appendChild(card);
      }
    }
  } catch (err) {
    addStat(statsGrid, "Manifest", "Not found. Run python make_webpage_assets.py");
    addStat(methodsGrid, "Methods", "Not found. Run python make_webpage_assets.py");
    addStat(transmission, "Transmission", "Not found. Run python make_webpage_assets.py");

    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "Gallery unavailable without manifest.";
    galleryGrid.appendChild(empty);

    const emptyL = document.createElement("p");
    emptyL.className = "hint";
    emptyL.textContent = "Lecture figures unavailable without manifest.";
    if (lectureGrid) {
      lectureGrid.appendChild(emptyL);
    }
  }
}

function captionFor(img) {
  const figure = img.closest("figure");
  if (figure) {
    const cap = figure.querySelector("figcaption");
    if (cap && cap.textContent.trim()) {
      return cap.textContent.trim();
    }
  }
  const card = img.closest(".card");
  if (card) {
    const h2 = card.querySelector("h2");
    const hint = card.querySelector(".hint");
    const title = h2 ? h2.textContent.trim() : "";
    const desc = hint ? hint.textContent.trim() : "";
    return desc ? `${title} — ${desc}` : title;
  }
  return img.alt || "";
}

function setupLightbox() {
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightboxImg");
  const lightboxCaption = document.getElementById("lightboxCaption");
  const lightboxClose = document.getElementById("lightboxClose");
  if (!lightbox || !lightboxImg || !lightboxCaption || !lightboxClose) {
    return;
  }

  function open(img) {
    lightboxImg.src = img.currentSrc || img.src;
    lightboxImg.alt = img.alt || "";
    lightboxCaption.textContent = captionFor(img);
    lightbox.hidden = false;
  }

  function close() {
    lightbox.hidden = true;
    lightboxImg.src = "";
  }

  document.addEventListener("click", (e) => {
    const img = e.target.closest(".image-frame img, .gallery-item img, .lecture-figure img");
    if (img) {
      open(img);
    }
  });

  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) {
      close();
    }
  });
  lightboxClose.addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) {
      close();
    }
  });
}

setupLightbox();
loadManifest();
