const REGION_LABEL = { US: "USA", UK: "UK", HK: "Hong Kong", AUTO: "Universal" };
const REGION_CHIP = { US: "us", UK: "uk", HK: "hk", AUTO: "univ" };

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const fileMeta = document.getElementById("fileMeta");
const regionSeg = document.getElementById("regionSeg");
const analyzeBtn = document.getElementById("analyzeBtn");
const uploadError = document.getElementById("uploadError");

const emptyState = document.getElementById("emptyState");
const progressState = document.getElementById("progressState");
const doneState = document.getElementById("doneState");
const progressLabel = document.getElementById("progressLabel");
const progressCount = document.getElementById("progressCount");
const progressBar = document.getElementById("progressBar");

const resultsSection = document.getElementById("resultsSection");
const resultsBody = document.getElementById("resultsBody");
const downloadBtn = document.getElementById("downloadBtn");
const thumbsEl = document.getElementById("thumbs");

const metricPlates = document.getElementById("metricPlates");
const metricConf = document.getElementById("metricConf");
const metricFrames = document.getElementById("metricFrames");
const metricRuntime = document.getElementById("metricRuntime");

let selectedFile = null;
let selectedRegion = "AUTO";
let pollHandle = null;
let jobStartedAt = null;

function formatDuration(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "--";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function formatTimestamp(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function confidenceClass(conf) {
  if (conf >= 0.85) return "high";
  if (conf >= 0.70) return "mid";
  return "low";
}

function readVideoMeta(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => {
      resolve({ width: v.videoWidth, height: v.videoHeight, duration: v.duration });
      URL.revokeObjectURL(url);
    };
    v.onerror = () => resolve(null);
    v.src = url;
  });
}

async function selectFile(file) {
  selectedFile = file;
  uploadError.classList.add("hidden");
  fileName.textContent = file.name;
  const meta = await readVideoMeta(file);
  fileMeta.textContent = meta
    ? `${meta.width}×${meta.height} · ${formatDuration(meta.duration)}`
    : "Video selected";
  analyzeBtn.disabled = false;
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) selectFile(e.target.files[0]);
});
["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) selectFile(file);
});

regionSeg.querySelectorAll(".opt").forEach((opt) => {
  opt.addEventListener("click", () => {
    regionSeg.querySelectorAll(".opt").forEach((o) => o.removeAttribute("aria-selected"));
    opt.setAttribute("aria-selected", "true");
    selectedRegion = opt.dataset.region;
  });
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  uploadError.classList.add("hidden");
  emptyState.classList.add("hidden");
  doneState.classList.add("hidden");
  resultsSection.classList.add("hidden");
  progressState.classList.remove("hidden");
  progressLabel.textContent = "Uploading…";
  progressCount.textContent = "";
  progressBar.style.width = "0%";

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("region", selectedRegion);

  jobStartedAt = Date.now();

  let resp;
  try {
    resp = await fetch("/api/jobs", { method: "POST", body: form });
  } catch (err) {
    showUploadError("Could not reach the server. Check your connection and try again.");
    return;
  }

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    showUploadError(body.detail || "Could not analyze this video.");
    return;
  }

  const { job_id } = await resp.json();
  pollJob(job_id);
});

function showUploadError(message) {
  progressState.classList.add("hidden");
  emptyState.classList.remove("hidden");
  uploadError.textContent = message;
  uploadError.classList.remove("hidden");
  analyzeBtn.disabled = false;
}

function pollJob(jobId) {
  if (pollHandle) clearInterval(pollHandle);
  pollHandle = setInterval(async () => {
    let job;
    try {
      const resp = await fetch(`/api/jobs/${jobId}`);
      job = await resp.json();
    } catch (err) {
      return;
    }

    if (job.status === "processing" || job.status === "queued") {
      const total = job.total_frames || 1;
      const done = job.frames_processed || 0;
      const pct = Math.min(100, Math.round((done / total) * 100));
      progressLabel.textContent = `Analyzing… frame ${done} of ${total}`;
      progressCount.textContent = `${done} / ${total}`;
      progressBar.style.width = `${pct}%`;
    } else if (job.status === "done") {
      clearInterval(pollHandle);
      renderDone(job, jobId);
    } else if (job.status === "error") {
      clearInterval(pollHandle);
      showUploadError(job.error || "Something went wrong while analyzing this video.");
    }
  }, 1000);
}

function renderDone(job, jobId) {
  progressState.classList.add("hidden");
  doneState.classList.remove("hidden");
  resultsSection.classList.remove("hidden");
  analyzeBtn.disabled = false;

  const plates = job.result.plates || [];
  const avgConf = plates.length
    ? plates.reduce((sum, p) => sum + (p.confidence || 0), 0) / plates.length
    : 0;
  const runtimeSec = (Date.now() - jobStartedAt) / 1000;

  metricPlates.textContent = plates.length;
  metricConf.textContent = `${Math.round(avgConf * 100)}%`;
  metricFrames.textContent = job.result.frames_processed;
  metricRuntime.textContent = formatDuration(runtimeSec);

  thumbsEl.innerHTML = "";
  (job.thumb_urls || []).forEach((url) => {
    const div = document.createElement("div");
    div.className = "thumb";
    const img = document.createElement("img");
    img.src = url;
    div.appendChild(img);
    thumbsEl.appendChild(div);
  });

  resultsBody.innerHTML = "";
  plates.forEach((p) => {
    const tr = document.createElement("tr");
    const conf = p.confidence || 0;
    tr.innerHTML = `
      <td class="bn-plate">${p.plate}</td>
      <td><span class="bn-chip ${REGION_CHIP[p.region] || "univ"}">${REGION_LABEL[p.region] || p.region}</span></td>
      <td class="bn-conf ${confidenceClass(conf)}">${Math.round(conf * 100)}%</td>
      <td>${formatTimestamp(p.first_time_sec)}</td>
    `;
    resultsBody.appendChild(tr);
  });

  downloadBtn.href = job.video_url;
}
