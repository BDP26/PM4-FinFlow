// Jahre und Bildpfade
const years = [
  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,
  2025,
];
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

// Overlays für alle Jahre anlegen und hinzufügen
years.forEach((year, i) => {
  const imgUrl = `images/${year}_all.png`;
  const overlay = L.imageOverlay(imgUrl, bounds, {
    opacity: i === 0 ? 1 : 0,
    interactive: false,
  });
  imageOverlays[year] = overlay;
  overlay.addTo(map);
});

// Slider-Setup
const slider = document.getElementById("year-slider");
const yearLabel = document.getElementById("year-label");
slider.min = 0;
slider.max = years.length - 1;
slider.value = 0;
yearLabel.textContent = years[0];

// Slider-Logik: Sofortiges Umschalten
slider.addEventListener("input", () => {
  const idx = parseInt(slider.value, 10);
  const year = years[idx];
  yearLabel.textContent = year;
  // Opacity aller Overlays setzen
  years.forEach((y) => {
    imageOverlays[y].setOpacity(y === year ? 1 : 0);
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
    if (idx < years.length - 1) {
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
