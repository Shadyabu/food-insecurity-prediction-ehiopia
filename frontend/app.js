const LEADS = [0, 1, 2, 3, 4, 8, 12];
const LEAD_LABELS = { 0: "Nowcast", 1: "+1mo", 2: "+2mo", 3: "+3mo", 4: "+4mo", 8: "+8mo", 12: "+12mo" };

// Displayed on a 1 / 2 / 3+ scale (FEWS NET's own "Crisis or worse"
// humanitarian-response threshold, same convention as this project's
// to_ipc_class_3plus() elsewhere) -- Crisis/Emergency/Famine (raw classes
// 3/4/5) are shown as a single "3+" bucket, not distinguished.
const IPC_COLORS = { 1: "#cdf582", 2: "#ffe29a", "3+": "#fa890f" };
const IPC_NAMES = { 1: "Minimal", 2: "Stressed", "3+": "Crisis or worse" };
const IPC_NA_COLOR = "#3a3f4b";

function displayIpcClass(rawClass) {
  return rawClass >= 3 ? "3+" : rawClass;
}

const CAUSE_COLORS = {
  DROUGHT: "#b7791f",
  FLOODING: "#2b6cb0",
  CONFLICT: "#c53030",
  MARKET_SHOCK: "#6b46c1",
  COMPOUND: "#718096",
  UNCLASSIFIED: "#4a5568",
  NO_DATA: "#4a5568",
};
const CAUSE_LABELS = {
  DROUGHT: "Drought",
  FLOODING: "Flooding",
  CONFLICT: "Conflict",
  MARKET_SHOCK: "Market shock",
  COMPOUND: "Compound",
  UNCLASSIFIED: "Unclassified",
  NO_DATA: "No data",
};

const LHZ_COLORS = { pct_pastoral: "#d69e2e", pct_agropastoral: "#38a169", pct_crop_farming: "#3182ce" };
const LHZ_LABELS = { pct_pastoral: "Pastoral", pct_agropastoral: "Agropastoral", pct_crop_farming: "Crop-farming" };

const SUBJECT_LABELS = {
  xgboost: "XGBoost",
  randomforest: "RandomForest",
  champion_ensemble: "Champion ensemble (XGBoost + RandomForest + LSTM + TabICLv2)",
};

let state = {
  data: null,
  geojson: null,
  leadIndex: 0,
  model: "champion_ensemble", // overwritten from data.default_subject once loaded
  map: null,
  geoLayer: null,
  hoveredZone: null, // last zone the side panel should display (hover or click)
  highlightZone: null, // zone whose border should be highlighted -- only while the mouse is actually over it
};

async function loadData() {
  const [data, geojson] = await Promise.all([
    fetch("data/nowcast.json").then((r) => r.json()),
    fetch("data/admin2_boundaries.geojson").then((r) => r.json()),
  ]);
  return { data, geojson };
}

function zonePrediction(zoneCode) {
  const lead = LEADS[state.leadIndex];
  const zone = state.data.zones[zoneCode];
  if (!zone) return null;
  const leadEntry = zone.predictions[String(lead)];
  if (!leadEntry) return null;
  // One selected model drives BOTH the map color and the explanation panel
  // -- its own prediction, its own explanation, always self-consistent.
  return { zone, leadEntry, pred: leadEntry[state.model] };
}

function styleForFeature(feature) {
  const zoneCode = feature.properties.ADM2_PCODE;
  const info = zonePrediction(zoneCode);
  const fill = info && info.pred ? IPC_COLORS[displayIpcClass(info.pred.ipc_class)] : IPC_NA_COLOR;
  const isHighlighted = state.highlightZone === zoneCode;
  return {
    fillColor: fill,
    fillOpacity: 0.85,
    color: isHighlighted ? "#ffffff" : "#20232b",
    weight: isHighlighted ? 2 : 0.6,
  };
}

function restyleMap() {
  if (state.geoLayer) state.geoLayer.setStyle(styleForFeature);
  updateSubtitleAndTargetMonth();
  if (state.hoveredZone) renderSidePanel(state.hoveredZone);
}

function updateSubtitleAndTargetMonth() {
  const lead = LEADS[state.leadIndex];
  const anyZone = Object.values(state.data.zones)[0];
  const targetMonth = anyZone.predictions[String(lead)].target_month;
  const originLabel = state.data.origin_month;
  document.getElementById("subtitle").textContent =
    lead === 0
      ? `Nowcast — target month: ${targetMonth} (generated from data as of ${originLabel})`
      : `Forecast — target month: ${targetMonth} (${LEAD_LABELS[lead]} from ${originLabel} origin)`;
  document.getElementById("target-month-label").textContent = `Target month: ${targetMonth}`;
}

function renderSidePanel(zoneCode) {
  const info = zonePrediction(zoneCode);
  const panel = document.getElementById("side-panel");
  if (!info || !info.pred) {
    panel.innerHTML = `<div class="panel-empty">No prediction available for this zone at this lead.</div>`;
    return;
  }
  const { zone, pred } = info;

  const displayClass = displayIpcClass(pred.ipc_class);
  const ipcColor = IPC_COLORS[displayClass];
  const ipcName = IPC_NAMES[displayClass];
  const causeColor = CAUSE_COLORS[pred.predicted_cause] || CAUSE_COLORS.UNCLASSIFIED;
  const causeLabel = CAUSE_LABELS[pred.predicted_cause] || pred.predicted_cause;
  // A Minimal (phase 1) food-security situation has no crisis to attribute
  // a cause to -- only show the cause/SHAP-evidence sections at phase 2+.
  const showCause = pred.ipc_class >= 2;

  const lhzBar = ["pct_pastoral", "pct_agropastoral", "pct_crop_farming"]
    .map((k) => `<div style="flex:${Math.max(zone[k], 0.001)};background:${LHZ_COLORS[k]}"></div>`)
    .join("");
  const lhzLegend = ["pct_pastoral", "pct_agropastoral", "pct_crop_farming"]
    .map((k) => `<span>${LHZ_LABELS[k]} ${(zone[k] * 100).toFixed(0)}%</span>`)
    .join("");

  // top_shap is already sorted descending by contribution -- scale each
  // bar relative to the top driver's own value (not a fixed absolute
  // scale), since raw SHAP magnitudes here are tiny (often < 0.001) and a
  // fixed scale would render every bar as an invisible sliver.
  const maxShap = showCause ? (pred.top_shap || []).reduce((m, s) => Math.max(m, Math.abs(s.shap)), 0) : 0;
  const shapRows = showCause
    ? (pred.top_shap || [])
        .map((s) => {
          const pct = maxShap > 0 ? Math.max(4, (Math.abs(s.shap) / maxShap) * 100) : 0;
          return `<div class="shap-row">
        <span class="shap-feature" title="${s.category}">${s.feature}</span>
        <div class="shap-bar-track"><div class="shap-bar-fill" style="width:${pct}%"></div></div>
        <span class="shap-value">${s.shap.toFixed(4)}</span>
      </div>`;
        })
        .join("") || `<div class="panel-empty">No qualifying SHAP drivers for this cause.</div>`
    : "";

  const limitationNote = pred.limitation
    ? `<div class="panel-note">${pred.limitation}</div>`
    : "";

  const causeSection = showCause
    ? `
    <div class="panel-section">
      <div class="panel-section-title">Reason for prediction${pred.is_compound ? " (compound)" : ""}</div>
      <span class="cause-badge" style="background:${causeColor}">${causeLabel}</span>
      ${limitationNote}
    </div>

    <div class="panel-section">
      <div class="panel-section-title">Top SHAP drivers</div>
      ${shapRows}
    </div>`
    : `
    <div class="panel-section">
      <div class="panel-section-title">Reason for prediction</div>
      <div class="panel-empty">Not shown at phase 1 (Minimal) — no crisis to attribute.</div>
    </div>`;

  panel.innerHTML = `
    <div class="panel-zone-name">${zone.zone_name}</div>
    <div class="panel-region">${zone.region} · dominant: ${LHZ_LABELS["pct_" + zone.dominant_livelihood_zone] || zone.dominant_livelihood_zone}</div>

    <div class="panel-section">
      <div class="panel-section-title">Predicted IPC phase (${SUBJECT_LABELS[state.model] || state.model})</div>
      <span class="ipc-badge" style="background:${ipcColor}">${displayClass} — ${ipcName}</span>
      <div style="font-size:0.75rem;color:var(--text-dim);margin-top:4px">continuous score: ${pred.ipc_continuous.toFixed(2)}</div>
    </div>
    ${causeSection}
    <div class="panel-section">
      <div class="panel-section-title">Livelihood zone</div>
      <div class="lhz-bar">${lhzBar}</div>
      <div class="lhz-legend-row">${lhzLegend}</div>
    </div>
  `;
}

function buildLegend() {
  const ipcLegend = document.getElementById("ipc-legend");
  ipcLegend.innerHTML = Object.entries(IPC_COLORS)
    .map(([cls, color]) => `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${cls} ${IPC_NAMES[cls]}</span>`)
    .join("");

  const causeLegend = document.getElementById("cause-legend");
  causeLegend.innerHTML = ["DROUGHT", "FLOODING", "CONFLICT", "MARKET_SHOCK", "COMPOUND"]
    .map((c) => `<span class="legend-item"><span class="legend-swatch" style="background:${CAUSE_COLORS[c]}"></span>${CAUSE_LABELS[c]}</span>`)
    .join("");
}

function buildSliderTicks() {
  const ticks = document.getElementById("slider-ticks");
  ticks.innerHTML = LEADS.map((lead, i) => `<span class="slider-tick" data-index="${i}">${LEAD_LABELS[lead]}</span>`).join("");
  ticks.querySelectorAll(".slider-tick").forEach((el) => {
    el.addEventListener("click", () => {
      state.leadIndex = Number(el.dataset.index);
      document.getElementById("lead-slider").value = state.leadIndex;
      updateActiveTick();
      restyleMap();
    });
  });
  updateActiveTick();
}

function updateActiveTick() {
  document.querySelectorAll(".slider-tick").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.index) === state.leadIndex);
  });
}

function buildModelSelect() {
  const select = document.getElementById("model-select");
  const subjects = state.data.subjects || Object.keys(SUBJECT_LABELS);
  select.innerHTML = subjects
    .map((s) => `<option value="${s}"${s === state.model ? " selected" : ""}>${SUBJECT_LABELS[s] || s}</option>`)
    .join("");
}

async function init() {
  const { data, geojson } = await loadData();
  state.data = data;
  state.geojson = geojson;
  state.model = data.default_subject || state.model;

  buildLegend();
  buildSliderTicks();
  buildModelSelect();

  const slider = document.getElementById("lead-slider");
  slider.addEventListener("input", () => {
    state.leadIndex = Number(slider.value);
    updateActiveTick();
    restyleMap();
  });

  document.getElementById("model-select").addEventListener("change", (e) => {
    state.model = e.target.value;
    restyleMap();
  });

  state.map = L.map("map", { zoomControl: true, attributionControl: false }).setView([9.1, 40.5], 6);

  state.geoLayer = L.geoJSON(geojson, {
    style: styleForFeature,
    onEachFeature: (feature, layer) => {
      const zoneCode = feature.properties.ADM2_PCODE;
      layer.on("mouseover", () => {
        state.hoveredZone = zoneCode;
        state.highlightZone = zoneCode;
        layer.setStyle({ color: "#ffffff", weight: 2 });
        renderSidePanel(zoneCode);
      });
      layer.on("mouseout", () => {
        state.highlightZone = null;
        layer.setStyle(styleForFeature(feature));
      });
      layer.on("click", () => {
        state.hoveredZone = zoneCode;
        renderSidePanel(zoneCode);
      });
      layer.bindTooltip(feature.properties.ADM2_EN, { sticky: true, className: "leaflet-zone-tooltip" });
    },
  }).addTo(state.map);

  state.map.fitBounds(state.geoLayer.getBounds(), { padding: [10, 10] });
  updateSubtitleAndTargetMonth();
}

init();
