// Monate und Bildpfade für 2012 und 2013
const months = [];
for (let year = 2012; year <= 2013; year++) {
  for (let m = 1; m <= 12; m++) {
    const mm = m.toString().padStart(2, "0");
    months.push(`${year}-${mm}`);
  }
}
const imageOverlays = {};
const bounds = [
  [0, 0],
  [3000, 6000],
];

// Map initialisieren
const map = L.map("map", {
  crs: L.CRS.Simple,
  minZoom: -2,
  maxZoom: 2,
  zoomControl: false,
  attributionControl: false,
  dragging: true,
  boxZoom: false,
  doubleClickZoom: false,
  scrollWheelZoom: true,
  keyboard: false,
  maxBounds: bounds,
  maxBoundsViscosity: 1.0,
});
map.fitBounds(bounds);

// Overlays für alle Monate anlegen und hinzufügen
months.forEach((month, i) => {
  const imgUrl = `images/${month}_all.png`;
  const overlay = L.imageOverlay(imgUrl, bounds, {
    opacity: i === 0 ? 1 : 0,
    interactive: false,
  });
  imageOverlays[month] = overlay;
  overlay.addTo(map);
});

// Slider-Setup
const slider = document.getElementById("year-slider");
const yearLabel = document.getElementById("year-label");
slider.min = 0;
slider.max = months.length - 1;
slider.value = 0;
yearLabel.textContent = months[0];

// Slider-Logik: Sofortiges Umschalten
slider.addEventListener("input", () => {
  const idx = parseInt(slider.value, 10);
  const month = months[idx];
  yearLabel.textContent = month;
  // Opacity aller Overlays setzen
  months.forEach((m) => {
    imageOverlays[m].setOpacity(m === month ? 1 : 0);
  });
});

// Play/Pause-Button Logik
const playBtn = document.getElementById("play-btn");
let playing = false;
let playInterval = null;

function setPlayIcon(isPlaying) {
  playBtn.innerHTML = isPlaying
    ? '<i class="bi bi-pause-fill"></i>'
    : '<i class="bi bi-play-fill"></i>';
}

function playSlider() {
  if (playing) return;
  // Immer von vorne starten
  slider.value = 0;
  slider.dispatchEvent(new Event("input"));
  playing = true;
  setPlayIcon(true);
  playInterval = setInterval(() => {
    let idx = parseInt(slider.value, 10);
    if (idx < months.length - 1) {
      slider.value = idx + 1;
      slider.dispatchEvent(new Event("input"));
    } else {
      pauseSlider();
    }
  }, 200);
}

function pauseSlider() {
  playing = false;
  setPlayIcon(false);
  if (playInterval) clearInterval(playInterval);
  playInterval = null;
}

playBtn.addEventListener("click", () => {
  if (playing) {
    pauseSlider();
  } else {
    playSlider();
  }
});

// Initial Icon setzen
setPlayIcon(false);

// Stoppe Animation, wenn Slider manuell bewegt wird
slider.addEventListener("mousedown", pauseSlider);
slider.addEventListener("touchstart", pauseSlider);
