function isFr() {
  return document.documentElement.classList.contains("lang-fr");
}

function L(en, fr) {
  return isFr() ? fr : en;
}

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

let cachedManifest = null;
let manifestLoadFailed = false;

function clearChildren(el) {
  if (el) {
    el.textContent = "";
  }
}

function renderManifest() {
  const statsGrid = document.getElementById("statsGrid");
  const methodsGrid = document.getElementById("methodsGrid");
  const galleryGrid = document.getElementById("galleryGrid");
  const lectureGrid = document.getElementById("lectureGrid");
  const transmission = document.getElementById("transmissionSummary");

  clearChildren(statsGrid);
  clearChildren(methodsGrid);
  clearChildren(galleryGrid);
  clearChildren(lectureGrid);
  clearChildren(transmission);

  const m = cachedManifest;

  if (manifestLoadFailed || !m) {
    addStat(statsGrid, L("Manifest", "Manifeste"), L("Not found. Run python make_webpage_assets.py", "Introuvable. Exécutez python make_webpage_assets.py"));
    addStat(methodsGrid, L("Methods", "Méthodes"), L("Not found. Run python make_webpage_assets.py", "Introuvable. Exécutez python make_webpage_assets.py"));
    addStat(transmission, L("Transmission", "Transmission"), L("Not found. Run python make_webpage_assets.py", "Introuvable. Exécutez python make_webpage_assets.py"));

    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = L("Gallery unavailable without manifest.", "Galerie indisponible sans manifeste.");
    galleryGrid.appendChild(empty);

    const emptyL = document.createElement("p");
    emptyL.className = "hint";
    emptyL.textContent = L("Lecture figures unavailable without manifest.", "Figures de cours indisponibles sans manifeste.");
    if (lectureGrid) {
      lectureGrid.appendChild(emptyL);
    }
    return;
  }

  {
    const d = m.detector_stats || {};
    const fits = d.fits || null;
    const statsIntro = document.getElementById("statsIntro");

    if (fits) {
      const [ny, nx] = fits.shape || [];
      addStat(statsGrid, L("Detector size", "Taille du détecteur"), ny && nx ? `${ny} × ${nx} px` : "n/a");
      addStat(statsGrid, L("Total detected photons", "Photons détectés (total)"), fmtNumber(fits.sum));
      addStat(statsGrid, L("Peak photons / pixel", "Photons max / pixel"), fmtNumber(fits.max));
      addStat(statsGrid, L("Illuminated pixels", "Pixels illuminés"),
        fits.illuminated_frac != null ? `${(fits.illuminated_frac * 100).toFixed(1)}%` : "n/a");
      addStat(statsGrid, L("p90 among lit pixels", "p90 (pixels illuminés)"), fmtNumber(fits.p90));
      addStat(statsGrid, L("p99 among lit pixels", "p99 (pixels illuminés)"), fmtNumber(fits.p99));
    } else {
      addStat(statsGrid, L("Detector stats", "Statistiques du détecteur"),
        L("n/a — run simulate_detector.py first", "n/d — exécutez d'abord simulate_detector.py"));
    }

    if (statsIntro) {
      const src = (m.methods || {}).source || {};
      const obsv = (m.methods || {}).observation || {};
      const env = (m.methods || {}).environment || {};
      if (fits && (m.methods && !m.methods.error)) {
        statsIntro.textContent = isFr()
          ? `Ceci est une exposition simulée unique : une étoile de Teff=${src.model_teff ?? "?"} K, ` +
            `magnitude ${fmtNumber(src.star_mag)} (${(src.star_mag_band || "R").toUpperCase()}), ` +
            `exposition de ${fmtSeconds(obsv.exposure_s)}` +
            (env.sky_enabled ? ", émission du ciel activée" : "") +
            (env.telluric_enabled ? `, absorption tellurique à la masse d'air ${fmtNumber(env.telluric_airmass)}` : "") +
            ". Les chiffres ci-dessous décrivent l'image de photons 4096×4096 résultante."
          : `This is a single simulated exposure: a Teff=${src.model_teff ?? "?"} K, ` +
            `${fmtNumber(src.star_mag)} mag (${(src.star_mag_band || "R").toUpperCase()}) star, ` +
            `${fmtSeconds(obsv.exposure_s)} exposure` +
            (env.sky_enabled ? ", sky emission on" : "") +
            (env.telluric_enabled ? `, telluric absorption at airmass ${fmtNumber(env.telluric_airmass)}` : "") +
            ". Numbers below describe the resulting 4096×4096 photon-count image.";
      } else {
        statsIntro.textContent = L(
          "Run simulate_detector.py then python make_webpage_assets.py to populate this section.",
          "Exécutez simulate_detector.py puis python make_webpage_assets.py pour remplir cette section."
        );
      }
    }

    const t = m.transmission_summary || {};
    if (t.error) {
      addStat(transmission, L("Transmission", "Transmission"), L("error: ", "erreur : ") + t.error);
    } else {
      addStat(transmission, L("Lambda range", "Plage de longueur d'onde"), `${fmtNumber(t.wavelength_min_nm)}-${fmtNumber(t.wavelength_max_nm)} nm`);
      addStat(transmission, L("Peak", "Pic"), `${fmtNumber(t.transmission_max)} @ ${fmtNumber(t.peak_nm)} nm`);
      addStat(transmission, "T(400)/T(500)",
        t.samples ? fmtNumber(Number(t.samples["400"]) / Number(t.samples["500"])) : "n/a");
      addStat(transmission, L("Blue sample T(380)", "Échantillon bleu T(380)"), t.samples ? fmtNumber(t.samples["380"]) : "n/a");
      addStat(transmission, L("Red sample T(900)", "Échantillon rouge T(900)"), t.samples ? fmtNumber(t.samples["900"]) : "n/a");
    }

    const meth = m.methods || {};
    if (meth.error) {
      addStat(methodsGrid, L("Methods", "Méthodes"), L("error: ", "erreur : ") + meth.error);
    } else {
      const obs = meth.observatory || {};
      const tel = meth.telescope || {};
      const src = meth.source || {};
      const env = meth.environment || {};
      const smp = meth.sampling || {};

      addStat(methodsGrid, L("Observatory", "Observatoire"), obs.name || "n/a");
      addStat(methodsGrid, L("Coordinates", "Coordonnées"), `${fmtNumber(obs.lat_deg)}, ${fmtNumber(obs.lon_deg)}`);
      addStat(methodsGrid, L("Telescope", "Télescope"), `${tel.name || "n/a"} (${fmtNumber(tel.diameter_m)} m)`);
      addStat(methodsGrid, L("Peak throughput", "Rendement maximal"), fmtNumber(tel.peak_throughput));
      addStat(methodsGrid, L("Exposure", "Exposition"), fmtSeconds((meth.observation || {}).exposure_s));
      addStat(methodsGrid, L("Spectrum mode", "Mode du spectre"), src.spectrum_mode || "n/a");
      addStat(methodsGrid, L("Stellar model", "Modèle stellaire"), `Teff=${src.model_teff ?? "n/a"}, logg=${src.model_logg ?? "n/a"}`);
      addStat(methodsGrid, L("Magnitude", "Magnitude"), `${fmtNumber(src.star_mag)} (${(src.star_mag_band || "R").toUpperCase()})`);
      addStat(methodsGrid, "vsini [km/s]", fmtNumber(src.star_vsini_kms));
      addStat(methodsGrid, L("Sampling [pix frac]", "Échantillonnage [frac. pix]"), fmtNumber(smp.wave_step_pix_frac));
      addStat(methodsGrid, L("Sky enabled", "Ciel activé"), String(Boolean(env.sky_enabled)));
      addStat(methodsGrid, L("Telluric enabled", "Tellurique activé"), String(Boolean(env.telluric_enabled)));
      addStat(methodsGrid, L("Telluric airmass", "Masse d'air tellurique"), fmtNumber(env.telluric_airmass));
    }

    const gallery = Array.isArray(m.gallery) ? m.gallery : [];
    if (!gallery.length) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = L(
        "No gallery images yet. Generate outputs and run python make_webpage_assets.py.",
        "Aucune image de galerie pour l'instant. Générez des sorties puis exécutez python make_webpage_assets.py."
      );
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
      empty.textContent = L(
        "No lecture figures yet. Run python make_webpage_assets.py.",
        "Aucune figure de cours pour l'instant. Exécutez python make_webpage_assets.py."
      );
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
  }
}

async function loadManifest() {
  const guiShot = document.getElementById("guiShot");
  const guiShotHint = document.getElementById("guiShotHint");

  guiShot.addEventListener("error", () => {
    guiShotHint.textContent = L(
      "Screenshot missing: put file at docs/assets/gui_screenshot.png",
      "Capture d'écran manquante : placez le fichier à docs/assets/gui_screenshot.png"
    );
  });

  try {
    cachedManifest = window.VROOMM_MANIFEST || null;
    if (!cachedManifest) {
      const res = await fetch("assets/manifest.json", { cache: "no-store" });
      if (!res.ok) {
        throw new Error("manifest not found");
      }
      cachedManifest = await res.json();
    }
  } catch (err) {
    manifestLoadFailed = true;
  }
  renderManifest();
}

function visibleText(el) {
  if (!el) return "";
  const langEl = el.querySelector(isFr() ? ".fr" : ".en");
  if (langEl) return langEl.textContent.trim();
  return el.textContent.trim();
}

function captionFor(img) {
  const figure = img.closest("figure");
  if (figure) {
    const cap = figure.querySelector("figcaption");
    if (cap && visibleText(cap)) {
      return visibleText(cap);
    }
  }
  const card = img.closest(".card");
  if (card) {
    const h2 = card.querySelector("h2");
    const hint = card.querySelector(".hint");
    const title = h2 ? visibleText(h2) : "";
    const desc = hint ? visibleText(hint) : "";
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
