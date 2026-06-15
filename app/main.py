from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.vision import CameraWorker, VisionEngine

app = FastAPI(title="Echelon Vision", version="0.2.0")
vision = VisionEngine(settings)
camera_worker = CameraWorker(settings, vision)
started_at = time.time()
BRAND_LOGO_PATH = Path(__file__).resolve().parent.parent / "3rdEchelonLogo.svg"


class ModelSelectPayload(BaseModel):
    model: str = Field(..., min_length=1)


class RuntimeSettingsPayload(BaseModel):
    camera_source: str | None = None
    confidence: float | None = Field(None, ge=0.01, le=1.0)
    iou: float | None = Field(None, ge=0.01, le=1.0)
    image_size: int | None = Field(None, ge=128, le=2048)
    bbox_opacity: float | None = Field(None, ge=0.0, le=1.0)
    sahi_enabled: bool | None = None
    sahi_slice_height: int | None = Field(None, ge=64, le=2048)
    sahi_slice_width: int | None = Field(None, ge=64, le=2048)
    sahi_overlap_height_ratio: float | None = Field(None, ge=0.0, le=0.9)
    sahi_overlap_width_ratio: float | None = Field(None, ge=0.0, le=0.9)
    alpr_enabled: bool | None = None


class ClassFilterPayload(BaseModel):
    enabled_classes: list[str]


@app.on_event("startup")
def on_startup() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.faces_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    camera_worker.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    camera_worker.stop()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_index_html())


@app.get("/settings", response_class=HTMLResponse)
def settings_page() -> HTMLResponse:
    return HTMLResponse(_settings_html())


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "uptime_seconds": round(time.time() - started_at, 2)})


@app.get("/brand.svg")
def brand_logo() -> FileResponse:
    return FileResponse(BRAND_LOGO_PATH, media_type="image/svg+xml")


@app.get("/api/status")
def status() -> JSONResponse:
    data = camera_worker.get_status()
    detections = camera_worker.get_latest_detections()
    return JSONResponse(
        {
            **data,
            "detection_count": len(detections),
            "detections": [
                {
                    "kind": detection.kind,
                    "label": detection.label,
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "box": list(detection.box),
                }
                for detection in detections
            ],
        }
    )


@app.get("/api/models")
def list_models() -> JSONResponse:
    models = vision.list_models()
    runtime = vision.get_runtime_snapshot()
    return JSONResponse(
        {
            "active_model": runtime.get("active_model"),
            "models": [
                {
                    "name": entry.name,
                    "path": entry.path,
                    "format": entry.format,
                    "classes": entry.classes,
                    "selectable": entry.selectable,
                    "note": entry.note,
                }
                for entry in models
            ],
        }
    )


@app.post("/api/models/select")
def select_model(payload: ModelSelectPayload) -> JSONResponse:
    ok, message = vision.select_model(payload.model)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return JSONResponse({"ok": True, "message": message, "runtime": vision.get_runtime_snapshot()})


@app.get("/api/settings")
def get_runtime_settings() -> JSONResponse:
    return JSONResponse(vision.get_runtime_snapshot())


@app.post("/api/settings")
def update_runtime_settings(payload: RuntimeSettingsPayload) -> JSONResponse:
    updates = payload.model_dump(exclude_none=True)
    if "camera_source" in updates:
        updates["camera_source"] = str(updates["camera_source"]).strip()
    vision.update_runtime(**updates)
    return JSONResponse({"ok": True, "runtime": vision.get_runtime_snapshot()})


@app.post("/api/classes")
def update_enabled_classes(payload: ClassFilterPayload) -> JSONResponse:
    vision.set_enabled_classes(payload.enabled_classes)
    return JSONResponse({"ok": True, "enabled_classes": payload.enabled_classes})


@app.get("/api/debug/rknn")
def debug_rknn() -> JSONResponse:
    """Run one inference and return raw output shapes + decode trace."""
    import numpy as np
    with vision._lock:
        rknn_model = vision._rknn_model
        labels = dict(vision._labels)
        image_size = vision.runtime.image_size
    if rknn_model is None:
        return JSONResponse({"error": "No RKNN model loaded"})
    dummy = np.full((image_size, image_size, 3), 128, dtype=np.uint8)
    try:
      variants = []
      best_boxes = 0
      best_shapes = []
      for idx, (inp, ratio, dw, dh) in enumerate(vision._rknn_input_variants(dummy, image_size)):
        entry: dict[str, Any] = {
          "variant_index": idx,
          "input_shape": list(inp.shape),
          "input_dtype": str(inp.dtype),
          "ratio": ratio,
          "dw": dw,
          "dh": dh,
        }
        try:
          outputs = rknn_model.inference(inputs=[inp]) or []
          shapes = [
            {"index": i, "shape": list(np.asarray(o).shape), "dtype": str(np.asarray(o).dtype)}
            for i, o in enumerate(outputs)
          ]
          boxes, _, _ = vision._decode_rknn_outputs(outputs, 0.01, num_classes_hint=len(labels), input_size=image_size)
          entry["num_outputs"] = len(outputs)
          entry["shapes"] = shapes
          entry["boxes_found_at_conf_0.01"] = int(boxes.shape[0])
          if int(boxes.shape[0]) > best_boxes:
            best_boxes = int(boxes.shape[0])
            best_shapes = shapes
        except Exception as inner_exc:
          entry["error"] = str(inner_exc)
        variants.append(entry)

      return JSONResponse(
        {
          "labels_loaded": len(labels),
          "best_boxes_found_at_conf_0.01": best_boxes,
          "best_shapes": best_shapes,
          "variants": variants,
        }
      )
    except Exception as exc:
        return JSONResponse({"error": str(exc)})


@app.get("/video.mjpg")
def video_stream() -> StreamingResponse:
    async def stream() -> AsyncIterator[bytes]:
        boundary = b"--frame\r\n"
        while True:
            frame = camera_worker.get_latest_jpeg()
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n"
            yield frame
            yield b"\r\n"
            await asyncio.sleep(1.0 / max(settings.camera_fps, 1))

    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


def _index_html() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Echelon Vision</title>
  <style>
    :root {
      --bg-a: #0b0f09;
      --bg-b: #141a10;
      --panel: rgba(14, 20, 11, 0.88);
      --line: rgba(134, 170, 90, 0.28);
      --line-strong: rgba(160, 196, 100, 0.52);
      --text: #edf2df;
      --muted: #9aaa7c;
      --accent: #a3c45a;
      --ok: #6bbf72;
      --warn: #d4b86a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      font-family: "Bahnschrift", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 10% 8%, rgba(90, 120, 40, 0.22), transparent 28%),
        radial-gradient(circle at 88% 16%, rgba(60, 90, 30, 0.18), transparent 30%),
        linear-gradient(145deg, var(--bg-a), var(--bg-b));
      min-height: 100vh;
    }
    .layout {
      max-width: 1520px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(330px, 0.85fr);
      gap: 16px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 18px 46px rgba(0, 0, 0, 0.42);
      backdrop-filter: blur(12px);
    }
    .hero {
      padding: 18px 20px 14px;
      border-bottom: 1px solid rgba(134, 170, 90, 0.14);
      display: grid;
      gap: 10px;
    }
    .brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .brand-left {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }
    .brand-mark {
      width: 68px;
      height: 68px;
      object-fit: contain;
      flex: 0 0 auto;
      filter: drop-shadow(0 8px 20px rgba(0, 0, 0, 0.35));
    }
    .brand-copy {
      display: grid;
      gap: 3px;
      min-width: 0;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }
    .title {
      margin: 0;
      font-size: clamp(1.5rem, 2.2vw, 2.25rem);
      letter-spacing: -0.02em;
      line-height: 1.05;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .clock {
      font-size: 0.88rem;
      color: var(--muted);
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.2);
      white-space: nowrap;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 9px;
      align-items: center;
    }
    .pill {
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 0.81rem;
      background: rgba(0, 0, 0, 0.28);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .pill-highlight {
      color: var(--accent);
      border-color: var(--line-strong);
      background: rgba(40, 60, 14, 0.55);
    }
    .stream-wrap {
      padding: 16px 20px 20px;
      position: relative;
    }
    .stream {
      width: 100%;
      aspect-ratio: 16/9;
      object-fit: cover;
      border-radius: 14px;
      border: 1px solid var(--line-strong);
      background: #000;
      cursor: zoom-in;
    }
    .scanline {
      position: absolute;
      left: 20px;
      right: 20px;
      top: 16px;
      bottom: 20px;
      border-radius: 14px;
      pointer-events: none;
      background: repeating-linear-gradient(
        to bottom,
        rgba(255, 255, 255, 0.02) 0,
        rgba(255, 255, 255, 0.02) 1px,
        transparent 1px,
        transparent 4px
      );
    }
    .stream-fullscreen-btn {
      position: absolute;
      top: 28px;
      right: 28px;
      z-index: 2;
      border: 1px solid var(--line-strong);
      background: rgba(10, 27, 39, 0.72);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font-weight: 650;
      letter-spacing: 0.04em;
      cursor: pointer;
      backdrop-filter: blur(8px);
    }
    .stream-wrap:fullscreen {
      padding: 0;
      background: #070d12;
    }
    .stream-wrap:fullscreen .stream {
      width: 100vw;
      height: 100vh;
      object-fit: contain;
      border-radius: 0;
      border: none;
      cursor: zoom-out;
    }
    .stream-wrap:fullscreen .stream-fullscreen-btn {
      top: 16px;
      right: 16px;
    }
    .side {
      padding: 16px;
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .metric {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: rgba(0, 0, 0, 0.17);
    }
    .metric-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
    }
    .label {
      color: var(--muted);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }
    .signal {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warn);
      box-shadow: 0 0 0 4px rgba(243, 197, 111, 0.16);
    }
    .signal.online {
      background: var(--ok);
      box-shadow: 0 0 0 4px rgba(127, 212, 167, 0.2);
    }
    .value {
      font-weight: 650;
      word-break: break-word;
      line-height: 1.3;
    }
    .value-focus {
      font-size: 1.48rem;
      letter-spacing: 0.02em;
    }
    .list {
      max-height: 290px;
      overflow: auto;
      display: grid;
      gap: 6px;
      padding-right: 2px;
    }
    .small {
      font-size: 0.82rem;
      color: var(--muted);
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 4px;
    }
    button, a.button {
      border: 1px solid var(--line-strong);
      background: rgba(42, 62, 16, 0.55);
      color: var(--text);
      border-radius: 10px;
      padding: 8px 12px;
      font-weight: 600;
      letter-spacing: 0.03em;
      text-decoration: none;
      cursor: pointer;
    }
    a.button:hover, button:hover {
      border-color: var(--accent);
      transform: translateY(-1px);
    }
    @media (max-width: 1120px) {
      .layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      .layout { padding: 14px; gap: 12px; }
      .hero { padding: 14px; }
      .stream-wrap { padding: 12px 14px 14px; }
      .scanline { left: 14px; right: 14px; top: 12px; bottom: 14px; }
      .brand { align-items: flex-start; }
      .brand-mark { width: 56px; height: 56px; }
      .clock { font-size: 0.76rem; }
    }
  </style>
</head>
<body>
  <main class="layout">
    <section class="card">
      <div class="hero">
        <div class="brand">
          <div class="brand-left">
            <img class="brand-mark" src="/brand.svg" alt="Echelon Vision insignia" />
            <div class="brand-copy">
              <p class="subtitle">Surveillance Eye Vision Aid</p>
              <h1 class="title">Echelon Vision Command</h1>
            </div>
          </div>
          <div class="clock" id="ops-clock">--:--:-- UTC</div>
        </div>
        <div class="row">
          <span class="pill pill-highlight">Live Operations Feed</span>
          <span class="pill" id="active-model">Model: loading...</span>
          <span class="pill" id="camera-source">Source: loading...</span>
        </div>
      </div>
      <div class="stream-wrap" ondblclick="toggleStreamFullscreen()" title="Double-click to fullscreen">
        <button type="button" class="stream-fullscreen-btn" id="stream-fullscreen-btn" onclick="toggleStreamFullscreen()">Fullscreen</button>
        <img class="stream" src="/video.mjpg" alt="Live detection stream" />
        <div class="scanline"></div>
      </div>
    </section>
    <aside class="card side">
      <div class="metric">
        <div class="metric-header">
          <div class="label">Camera Link</div>
          <div class="signal" id="camera-signal"></div>
        </div>
        <div class="value" id="camera-status">Loading...</div>
      </div>
      <div class="metric">
        <div class="metric-header">
          <div class="label">Objects This Frame</div>
        </div>
        <div class="value value-focus" id="detection-count">0</div>
      </div>
      <div class="metric">
        <div class="metric-header">
          <div class="label">Detection Trace</div>
        </div>
        <div id="details" class="list small">Waiting for first frame...</div>
      </div>
      <div class="actions">
        <a class="button" href="/settings">Open Mission Settings</a>
      </div>
    </aside>
  </main>
<script>
async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function toggleStreamFullscreen() {
  const wrap = document.querySelector('.stream-wrap');
  if (!wrap) return;

  if (document.fullscreenElement === wrap) {
    document.exitFullscreen();
    return;
  }

  if (wrap.requestFullscreen) {
    wrap.requestFullscreen();
  }
}

function syncFullscreenButton() {
  const button = document.getElementById('stream-fullscreen-btn');
  const wrap = document.querySelector('.stream-wrap');
  if (!button || !wrap) return;
  button.textContent = document.fullscreenElement === wrap ? 'Exit Fullscreen' : 'Fullscreen';
}

function syncClock() {
  const node = document.getElementById('ops-clock');
  if (!node) return;
  const now = new Date();
  const hh = String(now.getUTCHours()).padStart(2, '0');
  const mm = String(now.getUTCMinutes()).padStart(2, '0');
  const ss = String(now.getUTCSeconds()).padStart(2, '0');
  node.textContent = `${hh}:${mm}:${ss} UTC`;
}

document.addEventListener('fullscreenchange', syncFullscreenButton);
document.addEventListener('DOMContentLoaded', syncFullscreenButton);

async function refresh() {
  try {
    const [status, runtime] = await Promise.all([
      fetchJson('/api/status'),
      fetchJson('/api/settings')
    ]);

    document.getElementById('active-model').textContent = `Model: ${runtime.active_model || 'none'}`;
    document.getElementById('camera-source').textContent = `Source: ${runtime.camera_source || 'n/a'}`;
    document.getElementById('camera-status').textContent = status.camera_open
      ? `${status.width}x${status.height} @ frame ${status.frame_count}`
      : (status.last_error || 'camera offline');
    document.getElementById('detection-count').textContent = `${status.detection_count}`;

    const signal = document.getElementById('camera-signal');
    if (signal) {
      signal.classList.toggle('online', !!status.camera_open);
    }

    const details = document.getElementById('details');
    if (!status.detections.length) {
      details.innerHTML = '<div>No detections in current frame.</div>';
    } else {
      details.innerHTML = status.detections
        .slice(0, 16)
        .map((d) => `<div>• ${d.kind}:${d.class_name} (${d.confidence.toFixed(2)})</div>`)
        .join('');
    }

  } catch {
    document.getElementById('camera-status').textContent = 'status unavailable';
    const signal = document.getElementById('camera-signal');
    if (signal) signal.classList.remove('online');
  }
}

refresh();
syncClock();
setInterval(refresh, 1400);
setInterval(syncClock, 1000);
</script>
</body>
</html>
"""


def _settings_html() -> str:
    return """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Echelon Vision Settings</title>
  <style>
    :root {
        --bg: #0d120c;
        --bg2: #191f16;
        --panel: rgba(24, 28, 20, 0.9);
        --line: rgba(196, 182, 122, 0.2);
        --text: #f3edd9;
        --muted: #b7af8c;
        --accent: #b7a15c;
        --ok: #99b07a;
        --error: #8f6c4e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      font-family: "Aptos", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 90% 12%, rgba(183, 161, 92, 0.16), transparent 28%),
        radial-gradient(circle at 8% 85%, rgba(78, 102, 59, 0.22), transparent 30%),
        linear-gradient(150deg, var(--bg), var(--bg2));
      min-height: 100vh;
    }
    main { max-width: 1060px; margin: 0 auto; padding: 24px; display: grid; gap: 14px; }
    .card {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 16px 44px rgba(0,0,0,0.42);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-mark {
      width: 70px;
      height: 70px;
      object-fit: contain;
      flex: 0 0 auto;
      filter: drop-shadow(0 12px 24px rgba(0, 0, 0, 0.34));
    }
    .brand-copy { display: grid; gap: 4px; }
    .subtitle { margin: 0; color: var(--muted); font-size: 0.88rem; letter-spacing: 0.14em; text-transform: uppercase; }
    h1 { margin: 0; letter-spacing: -0.03em; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .field { display: grid; gap: 6px; }
    label { color: var(--muted); font-size: 0.88rem; }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      background: rgba(0,0,0,0.2);
      color: var(--text);
      border-radius: 10px;
      padding: 10px;
    }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    button, a {
      border: 1px solid var(--line);
      background: rgba(255, 179, 87, 0.2);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 14px;
      text-decoration: none;
      font-weight: 650;
      cursor: pointer;
    }
    .status { font-size: 0.9rem; color: var(--muted); }
    .ok { color: var(--ok); }
    .err { color: var(--error); }
    .class-grid {
      max-height: 220px;
      overflow: auto;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: rgba(0,0,0,0.18);
    }
    .cls {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.86rem;
      color: var(--text);
    }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <section class=\"card\">
    <div class="brand">
      <img class="brand-mark" src="/brand.svg" alt="Echelon Vision insignia" />
      <div class="brand-copy">
        <p class="subtitle">Mission Control</p>
        <h1>Echelon Vision Control</h1>
      </div>
    </div>
    <p class="status" style="margin-top: 10px;">Tactical tuning for confidence, IoU, bbox opacity, SAHI slicing, and camera source.</p>
  </section>

  <section class=\"card\">
    <div class=\"grid\">
      <div class=\"field\">
        <label>Camera Source (uvc://0, rtsp://..., rtmp://...)</label>
        <input id=\"camera_source\" />
      </div>
      <div class=\"field\">
        <label>Model</label>
        <select id=\"model_select\"></select>
      </div>
      <div class=\"field\">
        <label>Confidence (0-1)</label>
        <input id=\"confidence\" type=\"number\" step=\"0.01\" min=\"0.01\" max=\"1\" />
      </div>
      <div class=\"field\">
        <label>IoU (0-1)</label>
        <input id=\"iou\" type=\"number\" step=\"0.01\" min=\"0.01\" max=\"1\" />
      </div>
      <div class=\"field\">
        <label>Image Size</label>
        <input id=\"image_size\" type=\"number\" min=\"128\" max=\"2048\" step=\"32\" />
      </div>
      <div class=\"field\">
        <label>BBox Opacity (0-1)</label>
        <input id=\"bbox_opacity\" type=\"number\" step=\"0.05\" min=\"0\" max=\"1\" />
      </div>
      <div class=\"field\">
        <label>SAHI Enabled</label>
        <select id=\"sahi_enabled\"><option value=\"false\">false</option><option value=\"true\">true</option></select>
      </div>
      <div class=\"field\">
        <label>Fast-ALPR Experimental Flag</label>
        <select id=\"alpr_enabled\"><option value=\"false\">false</option><option value=\"true\">true</option></select>
      </div>
      <div class=\"field\">
        <label>SAHI Slice Height</label>
        <input id=\"sahi_slice_height\" type=\"number\" min=\"64\" max=\"2048\" step=\"16\" />
      </div>
      <div class=\"field\">
        <label>SAHI Slice Width</label>
        <input id=\"sahi_slice_width\" type=\"number\" min=\"64\" max=\"2048\" step=\"16\" />
      </div>
      <div class=\"field\">
        <label>SAHI Overlap Height</label>
        <input id=\"sahi_overlap_height_ratio\" type=\"number\" step=\"0.05\" min=\"0\" max=\"0.9\" />
      </div>
      <div class=\"field\">
        <label>SAHI Overlap Width</label>
        <input id=\"sahi_overlap_width_ratio\" type=\"number\" step=\"0.05\" min=\"0\" max=\"0.9\" />
      </div>
    </div>

    <div class=\"actions\" style=\"margin-top:14px;\">
      <button onclick=\"saveAll()\">Save Settings</button>
      <button onclick=\"reloadRuntime()\">Reload</button>
      <a href=\"/\">Back to Live</a>
    </div>
    <div id=\"status\" class=\"status\" style=\"margin-top:10px;\">Ready.</div>
  </section>
</main>
<script>
const keys = [
  'camera_source', 'confidence', 'iou', 'image_size', 'bbox_opacity',
  'sahi_enabled', 'sahi_slice_height', 'sahi_slice_width',
  'sahi_overlap_height_ratio', 'sahi_overlap_width_ratio', 'alpr_enabled'
];

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function setStatus(text, ok=true) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = 'status ' + (ok ? 'ok' : 'err');
}

function asNumber(id) { return Number(document.getElementById(id).value); }
function asBool(id) { return document.getElementById(id).value === 'true'; }

async function reloadRuntime() {
  const [runtime, modelData] = await Promise.all([
    fetchJson('/api/settings'),
    fetchJson('/api/models')
  ]);

  for (const key of keys) {
    const node = document.getElementById(key);
    if (!node) continue;
    const value = runtime[key];
    if (value === undefined || value === null) continue;
    node.value = String(value);
  }

  const select = document.getElementById('model_select');
  const models = modelData.models || [];
  select.innerHTML = models.map((m) => {
    const disabled = m.selectable ? '' : 'disabled';
    const note = m.note ? ` (${m.note})` : '';
    return `<option value="${m.name}" ${disabled}>${m.name} [${m.format}]${note}</option>`;
  }).join('');

  if (runtime.active_model) {
    select.value = runtime.active_model;
  }

  setStatus('Settings loaded.');
}

async function saveAll() {
  try {
    const payload = {
      camera_source: document.getElementById('camera_source').value,
      confidence: asNumber('confidence'),
      iou: asNumber('iou'),
      image_size: asNumber('image_size'),
      bbox_opacity: asNumber('bbox_opacity'),
      sahi_enabled: asBool('sahi_enabled'),
      sahi_slice_height: asNumber('sahi_slice_height'),
      sahi_slice_width: asNumber('sahi_slice_width'),
      sahi_overlap_height_ratio: asNumber('sahi_overlap_height_ratio'),
      sahi_overlap_width_ratio: asNumber('sahi_overlap_width_ratio'),
      alpr_enabled: asBool('alpr_enabled')
    };

    await fetchJson('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const selectedModel = document.getElementById('model_select').value;
    if (selectedModel) {
      await fetchJson('/api/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel })
      });
    }

    setStatus('Saved settings and model successfully.');
  } catch (err) {
    setStatus('Save failed: ' + err.message, false);
  }
}

reloadRuntime();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
