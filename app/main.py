from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.vision import CameraWorker, VisionEngine

app = FastAPI(title="Echelon Vision", version="0.2.0")
vision = VisionEngine(settings)
camera_worker = CameraWorker(settings, vision)
started_at = time.time()
BRAND_LOGO_PATH = Path(__file__).resolve().parent.parent / "3rdEchelonLogo.svg"
OFFICIAL_ASSETS_RELEASE = "v8.4.0"


def _official_model_catalog() -> list[dict[str, str]]:
  families = [
    ("yolo11", "YOLO11", {
      "": ("detect", "General object detection with bounding boxes."),
      "-seg": ("segment", "Instance segmentation with masks and boxes."),
      "-pose": ("pose", "Pose and keypoint estimation."),
      "-obb": ("obb", "Oriented bounding boxes for rotated objects."),
      "-cls": ("classify", "Image classification."),
    }),
    ("yolo26", "YOLO26", {
      "": ("detect", "General object detection with bounding boxes."),
      "-seg": ("segment", "Instance segmentation with masks and boxes."),
      "-sem": ("semantic", "Semantic segmentation."),
      "-pose": ("pose", "Pose and keypoint estimation."),
      "-obb": ("obb", "Oriented bounding boxes for rotated objects."),
      "-cls": ("classify", "Image classification."),
    }),
    ("yolov8", "YOLOv8", {
      "": ("detect", "General object detection with bounding boxes."),
      "-seg": ("segment", "Instance segmentation with masks and boxes."),
      "-pose": ("pose", "Pose and keypoint estimation."),
      "-obb": ("obb", "Oriented bounding boxes for rotated objects."),
      "-cls": ("classify", "Image classification."),
    }),
  ]

  entries: list[dict[str, str]] = []
  for stem, family, tasks in families:
    for size in "nsmlx":
      for suffix, (task, description) in tasks.items():
        name = f"{stem}{size}{suffix}.pt"
        entries.append(
          {
            "name": name,
            "family": family,
            "task": task,
            "description": description,
            "url": f"https://github.com/ultralytics/assets/releases/download/{OFFICIAL_ASSETS_RELEASE}/{name}",
          }
        )
  return entries


OFFICIAL_MODEL_CATALOG = _official_model_catalog()
OFFICIAL_MODEL_NAMES = {entry["name"] for entry in OFFICIAL_MODEL_CATALOG}
PLATE_WATCHLIST_PATH = settings.data_dir / "plate_watchlist.json"
MISSION_STATE_PATH = settings.data_dir / "mission_state.json"
MISSION_CATALOG: dict[str, Any] = {
  "vision": {
    "label": "Vision (YOLO Toolkit)",
    "summary": "Full computer-vision sandbox exposing every YOLO task with adjustable parameters.",
    "modules": [
      {
        "id": "vision_detect",
        "label": "Object Detection",
        "mode": "detect",
        "summary": "Define which target objects to detect with per-object constraints inside an optional geozone.",
        "params": [
          {"key": "classes", "type": "classlist", "label": "Target Objects (Legacy)", "default": "person,car"},
          {"key": "detection_objects", "type": "object_constraints", "label": "Per-Object Constraints (New)", "default": "[]"},
          {"key": "confidence", "type": "number", "label": "Confidence", "default": "0.35", "min": "0.01", "max": "1", "step": "0.01"},
          {"key": "iou", "type": "number", "label": "IoU (NMS)", "default": "0.50", "min": "0.01", "max": "1", "step": "0.01"},
          {"key": "imgsz", "type": "select", "label": "Image Size", "default": "640", "options": ["320", "480", "640", "960", "1280"]},
          {"key": "show_bbox", "type": "bool", "label": "Show Bounding Boxes", "default": "true", "workflow": "constraint"},
          {"key": "bbox_opacity", "type": "number", "label": "BBox Opacity", "default": "0.25", "min": "0", "max": "1", "step": "0.05", "workflow": "constraint"},
          {"key": "show_masks", "type": "bool", "label": "Show Masks", "default": "false", "workflow": "constraint"},
          {"key": "mask_opacity", "type": "number", "label": "Mask Opacity", "default": "0.25", "min": "0", "max": "1", "step": "0.05", "workflow": "constraint"},
          {"key": "detect_zone", "type": "zone", "shape": "polygon", "label": "Detection Geozone", "default": ""}
        ],
      },
      {
        "id": "vision_segment",
        "label": "Instance Segmentation",
        "mode": "segment",
        "summary": "Pixel-level masks with class filtering and mask opacity control.",
        "params": [
          {"key": "classes", "type": "classlist", "label": "Target Objects", "default": "person"},
          {"key": "confidence", "type": "number", "label": "Confidence", "default": "0.35", "min": "0.01", "max": "1", "step": "0.01"},
          {"key": "min_area_px", "type": "number", "label": "Min Mask Area (px)", "default": "500", "min": "0", "step": "10"},
          {"key": "show_bbox", "type": "bool", "label": "Show Bounding Boxes", "default": "true", "workflow": "constraint"},
          {"key": "bbox_opacity", "type": "number", "label": "BBox Opacity", "default": "0.25", "min": "0", "max": "1", "step": "0.05", "workflow": "constraint"},
          {"key": "show_masks", "type": "bool", "label": "Show Masks", "default": "true", "workflow": "constraint"},
          {"key": "mask_opacity", "type": "number", "label": "Mask Opacity", "default": "0.45", "min": "0", "max": "1", "step": "0.05", "workflow": "constraint"},
          {"key": "segment_zone", "type": "zone", "shape": "polygon", "label": "Area Of Interest", "default": ""}
        ],
      },
      {
        "id": "vision_pose",
        "label": "Pose / Keypoints",
        "mode": "pose",
        "summary": "Human skeletal keypoint estimation for posture analytics.",
        "params": [
          {"key": "confidence", "type": "number", "label": "Confidence", "default": "0.35", "min": "0.01", "max": "1", "step": "0.01"},
          {"key": "kpt_conf", "type": "number", "label": "Keypoint Confidence", "default": "0.50", "min": "0.01", "max": "1", "step": "0.01"}
        ],
      },
      {
        "id": "vision_obb",
        "label": "Oriented Boxes (OBB)",
        "mode": "obb",
        "summary": "Rotated bounding boxes for aerial or angled targets.",
        "params": [
          {"key": "classes", "type": "classlist", "label": "Target Objects", "default": ""},
          {"key": "confidence", "type": "number", "label": "Confidence", "default": "0.35", "min": "0.01", "max": "1", "step": "0.01"},
          {"key": "iou", "type": "number", "label": "IoU (NMS)", "default": "0.50", "min": "0.01", "max": "1", "step": "0.01"}
        ],
      },
      {
        "id": "vision_classify",
        "label": "Classification",
        "mode": "classify",
        "summary": "Whole-frame classification with Top-K label readout.",
        "params": [
          {"key": "topk", "type": "number", "label": "Top-K Labels", "default": "3", "min": "1", "max": "10", "step": "1"},
          {"key": "confidence", "type": "number", "label": "Confidence", "default": "0.25", "min": "0.01", "max": "1", "step": "0.01"}
        ],
      },
      {
        "id": "vision_track",
        "label": "Tracking / Counting",
        "mode": "track",
        "summary": "Persistent IDs with an optional counting line drawn on the video.",
        "params": [
          {"key": "classes", "type": "classlist", "label": "Target Objects", "default": "person,car"},
          {"key": "tracker", "type": "select", "label": "Tracker", "default": "bytetrack.yaml", "options": ["bytetrack.yaml", "botsort.yaml"]},
          {"key": "persist_limit_s", "type": "number", "label": "Lost Track Timeout (s)", "default": "3", "min": "0", "step": "0.5"},
          {"key": "count_line", "type": "zone", "shape": "line", "label": "Counting Line", "default": ""}
        ],
      },
    ],
  },
  "alpha_checkpoint": {
    "label": "Mission Alpha Checkpoint",
    "summary": "Border and entry point interdiction tools.",
    "modules": [
      {
        "id": "anpr_bolo",
        "label": "ANPR License Plate Reader",
        "mode": "detect",
        "summary": "BOLO license plate cross-reference at checkpoint lanes.",
        "params": [
          {"key": "trigger_conf", "type": "number", "label": "Trigger Confidence", "default": "0.65", "min": "0.01", "max": "1", "step": "0.01"},
          {"key": "watch_plates", "type": "list", "label": "BOLO Plates", "default": "", "placeholder": "Add plate e.g. ABC-1234"},
          {"key": "lane_gate", "type": "zone", "shape": "line", "label": "Lane Trigger Line", "default": ""}
        ],
        "requires": ["plate_watchlist"],
      },
      {
        "id": "speed_vector",
        "label": "Speed Estimation Radar",
        "mode": "track",
        "summary": "Track vectors and estimate vehicle speed from lane geometry.",
        "params": [
          {"key": "meter_scale", "type": "number", "label": "Meter Per Pixel", "default": "0.06", "min": "0.001", "step": "0.001"},
          {"key": "gate_line", "type": "zone", "shape": "line", "label": "Radar Gate Line", "default": ""}
        ],
      },
      {
        "id": "suspect_face",
        "label": "Target Suspect Facial Matching",
        "mode": "pose",
        "summary": "Face watch with uploaded suspect gallery.",
        "params": [
          {"key": "face_threshold", "type": "number", "label": "Match Threshold", "default": "0.40", "min": "0.01", "max": "1", "step": "0.01"},
          {"key": "watch_names", "type": "list", "label": "Suspect Names", "default": "", "placeholder": "Add suspect name"}
        ],
        "requires": ["face_gallery"],
      },
    ],
  },
  "moving_violations": {
    "label": "Mission Moving Violations",
    "summary": "Automated traffic enforcement and safety monitoring.",
    "modules": [
      {
        "id": "lane_encroach",
        "label": "Lane Encroachment Polygon",
        "mode": "segment",
        "summary": "Detect bikes or scooters violating sidewalk or bus lane zones.",
        "params": [
          {"key": "violation_classes", "type": "classlist", "label": "Violation Objects", "default": "motorcycle,bicycle"},
          {"key": "violation_zone", "type": "zone", "shape": "polygon", "label": "Violation Zone", "default": ""}
        ],
      },
      {
        "id": "red_light",
        "label": "Red Light Violations",
        "mode": "detect",
        "summary": "Trigger when vehicles cross stop line during red phase.",
        "params": [
          {"key": "signal_source", "type": "select", "label": "Signal State Source", "default": "manual", "options": ["manual", "detector", "external"]},
          {"key": "stop_line", "type": "zone", "shape": "line", "label": "Stop Line", "default": ""}
        ],
      },
      {
        "id": "triple_riding",
        "label": "Triple Riding Counter",
        "mode": "track",
        "summary": "Spatial count of overcrowded motorcycles.",
        "params": [{"key": "min_riders", "type": "number", "label": "Minimum Riders", "default": "3", "min": "1", "max": "6", "step": "1"}],
      },
    ],
  },
  "rally_square": {
    "label": "Mission Rally Square Monitoring",
    "summary": "Tactical crowd safety and emergency posture detection.",
    "modules": [
      {
        "id": "threat_pose",
        "label": "Threat Pose Alignment",
        "mode": "pose",
        "summary": "Weapon-alignment and threat posture cueing.",
        "params": [
          {"key": "pose_alert_ratio", "type": "number", "label": "Alert Ratio", "default": "0.70", "min": "0.01", "max": "1", "step": "0.01"},
          {"key": "watch_zone", "type": "zone", "shape": "polygon", "label": "Watch Zone", "default": ""}
        ],
      },
      {
        "id": "officer_down",
        "label": "Officer Down Fall Detection",
        "mode": "pose",
        "summary": "Skeletal fall posture monitoring for rapid emergency response.",
        "params": [{"key": "fall_window_s", "type": "number", "label": "Fall Window Seconds", "default": "2.0", "min": "0.5", "step": "0.5"}],
      },
    ],
  },
}
download_state_lock = threading.Lock()
download_state: dict[str, Any] = {
  "active": False,
  "status": "idle",
  "model": None,
  "progress": 0.0,
  "downloaded_bytes": 0,
  "total_bytes": 0,
  "message": "No download in progress.",
  "error": None,
}
download_thread: threading.Thread | None = None


class ModelSelectPayload(BaseModel):
    model: str = Field(..., min_length=1)


class RuntimeSettingsPayload(BaseModel):
    camera_source: str | None = None
    inference_mode: str | None = None
    confidence: float | None = Field(None, ge=0.01, le=1.0)
    iou: float | None = Field(None, ge=0.01, le=1.0)
    image_size: int | None = Field(None, ge=128, le=2048)
    bbox_opacity: float | None = Field(None, ge=0.0, le=1.0)
    mask_opacity: float | None = Field(None, ge=0.0, le=1.0)
    show_bbox: bool | None = None
    show_masks: bool | None = None
    tracking_enabled: bool | None = None
    tracking_persist: bool | None = None
    tracking_show_ids: bool | None = None
    classification_topk: int | None = Field(None, ge=1, le=10)
    sahi_enabled: bool | None = None
    sahi_slice_height: int | None = Field(None, ge=64, le=2048)
    sahi_slice_width: int | None = Field(None, ge=64, le=2048)
    sahi_overlap_height_ratio: float | None = Field(None, ge=0.0, le=0.9)
    sahi_overlap_width_ratio: float | None = Field(None, ge=0.0, le=0.9)


class ModelDownloadPayload(BaseModel):
  model: str = Field(..., min_length=1)
  use_after_download: bool = True


def _set_download_state(**updates: Any) -> None:
  with download_state_lock:
    download_state.update(updates)


def _download_sidecar_path(model_path: Path) -> Path:
  return model_path.with_suffix(model_path.suffix + ".classes.json")


def _ensure_downloaded_model_sidecar(model_path: Path) -> None:
  from convert_model_to_json import _load_names_from_model

  sidecar = _download_sidecar_path(model_path)
  if sidecar.exists():
    return

  names = _load_names_from_model(model_path)
  sidecar.write_text(json.dumps({"names": names}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _download_model_worker(model_name: str, use_after_download: bool) -> None:
  model_path = settings.models_dir / model_name
  temp_path = model_path.with_suffix(model_path.suffix + ".part")
  model_url = next(entry["url"] for entry in OFFICIAL_MODEL_CATALOG if entry["name"] == model_name)

  try:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
      _set_download_state(
        active=True,
        status="preparing",
        model=model_name,
        progress=100.0,
        message="Model already present. Preparing it for use...",
        error=None,
      )
    else:
      _set_download_state(
        active=True,
        status="downloading",
        model=model_name,
        progress=0.0,
        downloaded_bytes=0,
        total_bytes=0,
        message=f"Downloading {model_name}...",
        error=None,
      )
      with urlrequest.urlopen(model_url) as response, open(temp_path, "wb") as output_file:
        total_bytes = int(response.headers.get("Content-Length", "0") or 0)
        _set_download_state(total_bytes=total_bytes)
        while True:
          chunk = response.read(1024 * 256)
          if not chunk:
            break
          output_file.write(chunk)
          downloaded = int(download_state.get("downloaded_bytes", 0)) + len(chunk)
          progress = (downloaded / total_bytes * 100.0) if total_bytes else 0.0
          _set_download_state(downloaded_bytes=downloaded, progress=progress)
      temp_path.replace(model_path)

    _set_download_state(status="preparing", message="Generating model metadata...", progress=100.0)
    _ensure_downloaded_model_sidecar(model_path)

    load_message = "Model downloaded successfully."
    if use_after_download:
      ok, message = vision.select_model(model_name)
      if not ok:
        raise RuntimeError(message)
      load_message = f"{message} No service restart required."

    _set_download_state(
      active=False,
      status="completed",
      model=model_name,
      progress=100.0,
      message=load_message,
      error=None,
    )
  except Exception as exc:
    if temp_path.exists():
      temp_path.unlink(missing_ok=True)
    _set_download_state(
      active=False,
      status="error",
      model=model_name,
      error=str(exc),
      message=f"Download failed: {exc}",
    )


def _start_model_download(model_name: str, use_after_download: bool) -> None:
  global download_thread
  download_thread = threading.Thread(
    target=_download_model_worker,
    args=(model_name, use_after_download),
    daemon=True,
  )
  download_thread.start()


class ClassFilterPayload(BaseModel):
    enabled_classes: list[str]


class PlateWatchlistPayload(BaseModel):
  plates: list[str]


class MissionStatePayload(BaseModel):
  mission_id: str | None = None
  module_id: str | None = None
  parameters: dict[str, Any] | None = None


def _normalize_plate(text: str) -> str:
  return "".join(ch for ch in str(text).upper() if ch.isalnum() or ch in {"-", "_"}).strip()


def _load_plate_watchlist() -> list[str]:
  if not PLATE_WATCHLIST_PATH.exists():
    return []
  try:
    data = json.loads(PLATE_WATCHLIST_PATH.read_text(encoding="utf-8"))
  except Exception:
    return []
  if not isinstance(data, list):
    return []
  seen: set[str] = set()
  values: list[str] = []
  for item in data:
    plate = _normalize_plate(str(item))
    if not plate or plate in seen:
      continue
    seen.add(plate)
    values.append(plate)
  return values


def _save_plate_watchlist(plates: list[str]) -> list[str]:
  normalized = []
  seen: set[str] = set()
  for plate in plates:
    value = _normalize_plate(plate)
    if not value or value in seen:
      continue
    seen.add(value)
    normalized.append(value)
  settings.data_dir.mkdir(parents=True, exist_ok=True)
  PLATE_WATCHLIST_PATH.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
  return normalized


def _default_mission_state() -> dict[str, Any]:
  mission_id = next(iter(MISSION_CATALOG.keys()), "")
  modules = MISSION_CATALOG.get(mission_id, {}).get("modules", [])
  module_id = modules[0]["id"] if modules else ""
  return {
    "mission_id": mission_id,
    "module_id": module_id,
    "parameters": {},
  }


def _resolve_module(mission_id: str, module_id: str) -> dict[str, Any] | None:
  mission = MISSION_CATALOG.get(mission_id)
  if not mission:
    return None
  for module in mission.get("modules", []):
    if module.get("id") == module_id:
      return module
  return None


def _load_mission_state() -> dict[str, Any]:
  state = _default_mission_state()
  if MISSION_STATE_PATH.exists():
    try:
      raw = json.loads(MISSION_STATE_PATH.read_text(encoding="utf-8"))
      if isinstance(raw, dict):
        state.update({
          "mission_id": str(raw.get("mission_id", state["mission_id"])),
          "module_id": str(raw.get("module_id", state["module_id"])),
          "parameters": raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {},
        })
    except Exception:
      pass

  if not _resolve_module(state["mission_id"], state["module_id"]):
    state = _default_mission_state()
  return state


def _save_mission_state(state: dict[str, Any]) -> None:
  settings.data_dir.mkdir(parents=True, exist_ok=True)
  MISSION_STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _mission_readiness(state: dict[str, Any]) -> dict[str, Any]:
  runtime = vision.get_runtime_snapshot()
  module = _resolve_module(state.get("mission_id", ""), state.get("module_id", "")) or {}
  required = module.get("requires", [])
  checks = [
    {"id": "camera", "label": "Camera configured", "ready": bool(runtime.get("camera_source"))},
    {"id": "model", "label": "Model selected", "ready": bool(runtime.get("active_model"))},
  ]
  if "face_gallery" in required:
    checks.append({"id": "face_gallery", "label": "Face gallery loaded", "ready": bool(getattr(vision.face_analyzer, "known_embeddings", []))})
  if "plate_watchlist" in required:
    checks.append({"id": "plate_watchlist", "label": "Plate watchlist configured", "ready": len(_load_plate_watchlist()) > 0})
  return {"checks": checks, "ready": all(item["ready"] for item in checks)}


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
    mission_state = _load_mission_state()
    return JSONResponse(
        {
            **data,
          "server_time": time.time(),
          "plate_watchlist_count": len(_load_plate_watchlist()),
          "mission_state": mission_state,
          "mission_ready": _mission_readiness(mission_state),
            "detection_count": len(detections),
            "detections": [
                {
                    "kind": detection.kind,
                    "label": detection.label,
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "box": list(detection.box),
                  "track_id": detection.track_id,
                  "attributes": detection.attributes,
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


@app.get("/api/model-catalog")
def model_catalog() -> JSONResponse:
    local_files = {path.name for path in settings.models_dir.glob("*") if path.is_file()}
    return JSONResponse(
        {
            "models": [
                {
                    **entry,
                    "downloaded": entry["name"] in local_files,
                }
                for entry in OFFICIAL_MODEL_CATALOG
            ]
        }
    )


@app.get("/api/model-download")
def model_download_status() -> JSONResponse:
    with download_state_lock:
        return JSONResponse(dict(download_state))


@app.post("/api/model-download")
def model_download(payload: ModelDownloadPayload) -> JSONResponse:
    model_name = payload.model.strip()
    if model_name not in OFFICIAL_MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown official model: {model_name}")

    with download_state_lock:
        if download_state.get("active"):
            raise HTTPException(status_code=409, detail="Another model download is already in progress")

    _start_model_download(model_name, payload.use_after_download)
    return JSONResponse({"ok": True, "message": f"Started download for {model_name}"})


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
        cam_source = str(updates["camera_source"]).strip()
        # Support RTSP, RTMP, UVC, and browser camera formats
        if cam_source:
            # Validate URL formats - support RTSP, RTMP, and local sources
            if (cam_source.startswith('rtsp://') or cam_source.startswith('rtmp://') or 
                cam_source.startswith('rtsps://') or cam_source.startswith('rtmps://') or
                cam_source.startswith('uvc://') or cam_source.startswith('browser://')):
                # Valid URL format - use as-is
                updates["camera_source"] = cam_source
            elif cam_source.isdigit():
                # Numeric camera index
                updates["camera_source"] = f"uvc://{cam_source}"
            elif cam_source == 'internal':
                # Internal camera reference
                updates["camera_source"] = "browser://webui"
            else:
                # Try to preserve the input if it looks like a valid source
                updates["camera_source"] = cam_source
        else:
            updates["camera_source"] = ""
    
    vision.update_runtime(**updates)
    return JSONResponse({"ok": True, "runtime": vision.get_runtime_snapshot()})


@app.post("/api/classes")
def update_enabled_classes(payload: ClassFilterPayload) -> JSONResponse:
    vision.set_enabled_classes(payload.enabled_classes)
    return JSONResponse({"ok": True, "enabled_classes": payload.enabled_classes})


@app.get("/api/faces")
def list_faces() -> JSONResponse:
  settings.faces_dir.mkdir(parents=True, exist_ok=True)
  people: list[dict[str, Any]] = []
  for person_dir in sorted(settings.faces_dir.iterdir()):
    if not person_dir.is_dir():
      continue
    images = [
      path.name
      for path in sorted(person_dir.iterdir())
      if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]
    people.append({"name": person_dir.name, "images": images, "count": len(images)})
  return JSONResponse({"ok": True, "people": people, "known_profiles": len(vision.face_analyzer.known_embeddings)})


@app.post("/api/faces/upload")
async def upload_face(person: str = Form(...), image: UploadFile = File(...)) -> JSONResponse:
  person_name = "".join(ch for ch in person.strip() if ch.isalnum() or ch in {"_", "-", " "}).strip()
  if not person_name:
    raise HTTPException(status_code=400, detail="Person name is required")

  suffix = Path(image.filename or "").suffix.lower()
  if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
    raise HTTPException(status_code=400, detail="Unsupported image format")

  content = await image.read()
  if not content:
    raise HTTPException(status_code=400, detail="Empty image upload")

  target_dir = settings.faces_dir / person_name
  target_dir.mkdir(parents=True, exist_ok=True)
  target_name = f"{int(time.time() * 1000)}{suffix}"
  target_path = target_dir / target_name
  target_path.write_bytes(content)

  known = vision.reload_face_gallery()
  return JSONResponse(
    {
      "ok": True,
      "saved_as": str(target_path.name),
      "person": person_name,
      "known_profiles": known,
    }
  )


@app.get("/api/watchlist/plates")
def get_plate_watchlist() -> JSONResponse:
  return JSONResponse({"ok": True, "plates": _load_plate_watchlist()})


@app.post("/api/watchlist/plates")
def set_plate_watchlist(payload: PlateWatchlistPayload) -> JSONResponse:
  plates = _save_plate_watchlist(payload.plates)
  return JSONResponse({"ok": True, "plates": plates, "count": len(plates)})


@app.get("/api/missions/catalog")
def get_mission_catalog() -> JSONResponse:
  return JSONResponse({"ok": True, "catalog": MISSION_CATALOG})


@app.get("/api/missions/state")
def get_mission_state() -> JSONResponse:
  state = _load_mission_state()
  return JSONResponse({"ok": True, "state": state, "readiness": _mission_readiness(state)})


@app.post("/api/missions/state")
def set_mission_state(payload: MissionStatePayload) -> JSONResponse:
  state = _load_mission_state()

  next_mission = payload.mission_id if payload.mission_id is not None else state.get("mission_id", "")
  next_module = payload.module_id if payload.module_id is not None else state.get("module_id", "")

  module = _resolve_module(str(next_mission), str(next_module))
  if module is None:
    raise HTTPException(status_code=400, detail="Invalid mission selection")

  state["mission_id"] = str(next_mission)
  state["module_id"] = str(next_module)
  if payload.parameters is not None:
    state["parameters"] = payload.parameters

  # Push display toggles to runtime immediately when params include them
  params = payload.parameters or {}
  runtime_patch: dict[str, Any] = {}
  if "show_bbox" in params:
    runtime_patch["show_bbox"] = (str(params["show_bbox"]).lower() in ("true", "1", "yes"))
  if "bbox_opacity" in params:
    runtime_patch["bbox_opacity"] = max(0.0, min(float(params["bbox_opacity"]), 1.0))
  if "show_masks" in params:
    runtime_patch["show_masks"] = (str(params["show_masks"]).lower() in ("true", "1", "yes"))
  if "mask_opacity" in params:
    runtime_patch["mask_opacity"] = max(0.0, min(float(params["mask_opacity"]), 1.0))
  if runtime_patch:
    vision.update_runtime(**runtime_patch)

  _save_mission_state(state)
  return JSONResponse({"ok": True, "state": state, "readiness": _mission_readiness(state)})


@app.post("/api/browser-frame")
async def upload_browser_frame(frame: UploadFile = File(...)) -> JSONResponse:
  payload = await frame.read()
  if not payload:
    raise HTTPException(status_code=400, detail="Empty frame payload")

  decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
  if decoded is None:
    raise HTTPException(status_code=400, detail="Invalid image payload")

  camera_worker.ingest_browser_frame(decoded)
  return JSONResponse({"ok": True, "width": int(decoded.shape[1]), "height": int(decoded.shape[0])})


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


@app.get("/api/debug/bbox-trace")
def debug_bbox_trace() -> JSONResponse:
    """Return a per-frame trace of the last annotate() call.

    Useful for diagnosing why bounding boxes are not appearing in the stream:
      - model_format        which backend ran (pt / onnx / rknn)
      - decode_strategy     which decoder path was chosen
      - variants_tried      how many input formats were tested (RKNN only)
      - variant_results     per-variant decode/remap counts
      - raw_detections      boxes that survived decoding + NMS + remap
      - total_detections_drawn  boxes actually drawn on the frame (after class filter)
    """
    return JSONResponse(vision.get_bbox_trace())


@app.get("/video.mjpg")
def video_stream() -> StreamingResponse:
  async def stream() -> AsyncIterator[bytes]:
    while True:
      frame = camera_worker.get_latest_jpeg()
      payload = (
        b"--frame\r\n"
        + f"Content-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\nCache-Control: no-cache\r\n\r\n".encode()
        + frame
        + b"\r\n"
      )
      yield payload
      await asyncio.sleep(1.0 / max(settings.camera_fps, 1))

  return StreamingResponse(
    stream(),
    media_type="multipart/x-mixed-replace; boundary=frame",
    headers={
      "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
      "Pragma": "no-cache",
      "Expires": "0",
      "X-Accel-Buffering": "no",
    },
  )


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
      padding-bottom: env(safe-area-inset-bottom);
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
      object-fit: contain;
      border-radius: 14px;
      border: 1px solid var(--line-strong);
      background: #000;
      cursor: zoom-in;
      position: relative;
      z-index: 1;
    }
    .stream-fullscreen-btn {
      position: absolute;
      top: 28px;
      right: 28px;
      z-index: 5;
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
    .zone-canvas {
      position: absolute;
      z-index: 4;
      border-radius: 14px;
      cursor: crosshair;
      touch-action: none;
    }
    .param-list { display: flex; flex-direction: column; gap: 6px; }
    .param-list-input { display: flex; gap: 6px; }
    .param-list-input input { flex: 1; }
    .param-chips { display: flex; flex-wrap: wrap; gap: 5px; }
    .param-chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 6px 3px 9px;
      border: 1px solid var(--line-strong);
      border-radius: 999px;
      background: rgba(40, 60, 14, 0.42);
      font-size: 0.76rem;
      letter-spacing: 0.03em;
    }
    .param-chip button {
      border: none;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 0.85rem;
      line-height: 1;
      padding: 0 2px;
    }
    .param-chip button:hover { color: #ff8f8f; }
    .param-zone-row { display: flex; gap: 6px; align-items: center; }
    .param-zone-status { flex: 1; font-size: 0.76rem; color: var(--muted); }
    .param-zone-row button {
      padding: 5px 9px;
      font-size: 0.74rem;
      border-radius: 8px;
      border: 1px solid var(--line-strong);
      background: rgba(40, 60, 14, 0.42);
      color: var(--text);
      cursor: pointer;
    }
    .param-swatch {
      width: 14px;
      height: 14px;
      border-radius: 4px;
      border: 1px solid var(--line-strong);
      display: inline-block;
    }
    /* ── Target Selection Table ── */
    .param-object-constraints { display: flex; flex-direction: column; gap: 8px; }
    .obj-table tbody tr {
      border-bottom: 1px solid var(--line);
    }
    .obj-table tbody tr:hover { background: rgba(163,196,90,0.05); }
    .obj-table tbody tr:last-child { border-bottom: none; }
    .obj-table .cls-cell {
      color: var(--accent);
      font-weight: 600;
      white-space: nowrap;
      min-width: 70px;
    }
    .obj-table input[type=number] {
      width: 72px;
      padding: 3px 5px;
      background: rgba(40,60,14,0.35);
      border: 1px solid var(--line-strong);
      border-radius: 4px;
      color: var(--text);
      font-size: 0.75rem;
    }
    .obj-table select {
      padding: 3px 5px;
      background: rgba(18,26,13,0.96);
      border: 1px solid var(--line-strong);
      border-radius: 4px;
      color: var(--text);
      font-size: 0.75rem;
      min-height: 28px;
    }
    .obj-table select option {
      background: #11180c;
      color: var(--text);
    }
    .obj-table .zone-cell button {
      padding: 2px 7px;
      font-size: 0.68rem;
    }
    .add-object-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .add-object-row select,
    .add-object-row input {
      flex: 1 1 220px;
      min-width: 220px;
      padding: 8px 10px;
      background: rgba(18, 26, 13, 0.96);
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      color: var(--text);
    }
    .add-object-row select option {
      background: #11180c;
      color: var(--text);
    }
    .add-object-row select:focus,
    .add-object-row input:focus,
    .obj-table select:focus,
    .obj-table input[type=number]:focus {
      outline: 1px solid var(--accent);
      border-color: var(--accent);
    }
    .btn-tiny {
      padding: 2px 6px;
      font-size: 0.68rem;
      border: none;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
    }
    .btn-tiny:hover { color: #ff9a9a; }
    .btn-small {
      padding: 4px 8px;
      font-size: 0.7rem;
      border: 1px solid var(--line-strong);
      background: rgba(80, 100, 40, 0.4);
      color: var(--text);
      border-radius: 4px;
      cursor: pointer;
    }
    .btn-small:hover { background: rgba(80, 100, 40, 0.6); }
    .param-bool {
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      font-size: 0.82rem;
      color: var(--text);
      user-select: none;
    }
    .param-bool input[type=checkbox] {
      width: 15px;
      height: 15px;
      accent-color: var(--accent);
      cursor: pointer;
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
    .stream-wrap:-webkit-full-screen {
      padding: 0;
      background: #070d12;
    }
    .stream-wrap:-webkit-full-screen .stream {
      width: 100vw;
      height: 100vh;
      object-fit: contain;
      border-radius: 0;
      border: none;
      cursor: zoom-out;
    }
    .stream-wrap:-webkit-full-screen .stream-fullscreen-btn {
      top: 16px;
      right: 16px;
    }
    .side {
      padding: 16px;
      display: grid;
      gap: 10px;
      align-content: start;
      position: sticky;
      top: 20px;
      max-height: calc(100vh - 40px);
      overflow: auto;
    }
    .metric {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: rgba(0, 0, 0, 0.17);
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .sidebar-stack {
      display: grid;
      gap: 10px;
      min-height: 0;
    }
    .sidebar-section {
      background: linear-gradient(180deg, rgba(10, 14, 8, 0.12), rgba(0, 0, 0, 0.2));
    }
    .sidebar-section .metric-header {
      margin-bottom: 8px;
    }
    .sidebar-note {
      color: var(--muted);
      font-size: 0.78rem;
      line-height: 1.45;
      margin-bottom: 10px;
    }
    .metric-cta {
      padding: 12px 14px;
    }
    .sidebar-cta {
      display: flex;
      justify-content: flex-start;
    }
    .sidebar-cta .button {
      width: 100%;
      justify-content: center;
    }
    .stat-box {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: rgba(0, 0, 0, 0.2);
    }
    .stat-title {
      color: var(--muted);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 1.2rem;
      font-weight: 700;
      line-height: 1.1;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .status-cell {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      padding: 8px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: rgba(60, 80, 20, 0.2);
    }
    .status-label {
      font-size: 0.7rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .status-value {
      font-size: 1.3rem;
      color: var(--accent);
      font-weight: 700;
    }
    .status-value.ok { color: var(--ok); }
    .status-value.warn { color: #d4a574; }
    .status-value.error { color: #ff9a9a; }
    .system-detail {
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px;
      background: rgba(40, 60, 14, 0.15);
    }
    .system-detail summary {
      cursor: pointer;
      font-size: 0.8rem;
      color: var(--accent);
      font-weight: 500;
      padding: 4px;
      user-select: none;
    }
    .system-detail summary:hover {
      background: rgba(60, 80, 20, 0.2);
      border-radius: 4px;
    }
    .system-header {
      font-size: 0.8rem;
      color: var(--accent);
      font-weight: 500;
      cursor: pointer;
      padding: 4px;
    }
    .system-header:hover {
      background: rgba(60, 80, 20, 0.2);
      border-radius: 4px;
    }
    .system-content {
      padding: 8px;
      margin-top: 6px;
      border-top: 1px solid var(--line);
      font-size: 0.75rem;
    }
    .tiny {
      font-size: 0.76rem;
      color: var(--muted);
      margin-top: 4px;
    }
    .inline-input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(0, 0, 0, 0.2);
      color: var(--text);
      padding: 8px;
      font-family: inherit;
      font-size: 0.82rem;
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
    .panel {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: rgba(0, 0, 0, 0.16);
      display: grid;
      gap: 8px;
    }
    .panel[hidden] { display: none; }
    .mission-controls {
      display: grid;
      gap: 8px;
    }
    .mission-grid {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .mission-grid label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .mission-grid select,
    #report_filter {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(0, 0, 0, 0.18);
      color: var(--text);
      padding: 8px;
    }
    .mission-grid option,
    #report_filter option {
      background: #121a21;
      color: var(--text);
    }
    .mission-summary {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: rgba(0, 0, 0, 0.18);
      color: var(--text);
      font-size: 0.82rem;
      line-height: 1.45;
    }
    .checklist {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      font-size: 0.78rem;
    }
    .check-item {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      background: rgba(0, 0, 0, 0.16);
      color: var(--muted);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }
    .check-item .check-light {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #c59a57;
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.18);
      flex: 0 0 auto;
    }
    .check-item .check-state {
      font-size: 0.68rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      opacity: 0.85;
    }
    .check-item.ready {
      border-color: rgba(111, 209, 132, 0.5);
      color: #bde5c6;
    }
    .check-item.ready .check-light {
      background: #6fd184;
      box-shadow: 0 0 10px rgba(111, 209, 132, 0.55);
    }
    .module-params {
      display: grid;
      gap: 10px;
    }
    .param-section {
      display: grid;
      gap: 8px;
      padding: 10px;
      border: 1px solid rgba(134, 170, 90, 0.18);
      border-radius: 12px;
      background: rgba(0, 0, 0, 0.16);
    }
    .section-header {
      color: var(--accent);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-weight: 650;
    }
    .module-params input, .module-params select, .module-params textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(0, 0, 0, 0.18);
      color: var(--text);
      padding: 8px;
    }
    .quick-grid {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .stream-report {
      margin: 0 20px 20px;
      padding-top: 14px;
      border-top: 1px solid rgba(134, 170, 90, 0.14);
      display: grid;
      gap: 8px;
    }
    .report-list {
      max-height: 240px;
      overflow: auto;
      display: grid;
      gap: 6px;
    }
    .report-item {
      border: 1px solid var(--line);
      border-left-width: 4px;
      border-radius: 9px;
      padding: 8px;
      font-size: 0.8rem;
      background: rgba(0, 0, 0, 0.2);
      color: var(--text);
    }
    .report-high { border-left-color: #6fd184; }
    .report-mid { border-left-color: #d8c26e; }
    .report-low { border-left-color: #d98974; }
    .report-critical {
      border-left-color: #ff4a4a;
      border-color: rgba(255, 80, 80, 0.45);
      background: rgba(120, 12, 12, 0.28);
      box-shadow: inset 0 0 0 1px rgba(255, 90, 90, 0.14);
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
      padding: 10px 14px;
      min-height: 44px;
      font-weight: 600;
      letter-spacing: 0.03em;
      text-decoration: none;
      cursor: pointer;
      touch-action: manipulation;
    }
    a.button:hover, button:hover {
      border-color: var(--accent);
      transform: translateY(-1px);
    }
    @media (max-width: 1120px) {
      .layout { grid-template-columns: 1fr; }
      .side {
        position: static;
        top: auto;
        max-height: none;
        overflow: visible;
      }
    }
    @media (max-width: 760px) {
      .layout { padding: 14px; gap: 12px; }
      .hero { padding: 14px; }
      .stream-wrap { padding: 12px 14px 14px; }
      .stream-report { margin: 0 14px 14px; }
      .brand { align-items: flex-start; }
      .brand-mark { width: 56px; height: 56px; }
      .clock { font-size: 0.76rem; padding: 6px 8px; }
      .title { white-space: normal; overflow: visible; text-overflow: clip; }
      .metric-grid { grid-template-columns: 1fr; }
      .stream-fullscreen-btn { top: 18px; right: 18px; }
      .actions { gap: 10px; }
      .actions a.button, .actions button { flex: 1 1 100%; justify-content: center; text-align: center; }
      .quick-grid { grid-template-columns: 1fr; }
      .mission-grid { grid-template-columns: 1fr; }
    }
    
    /* Parameter workflow sections */
    .param-section {
      margin: 8px 0;
      padding: 8px;
      border-left: 3px solid var(--accent);
      background: rgba(163, 196, 90, 0.05);
    }
    .section-header {
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .param-section > div:not(.section-header) {
      margin: 6px 0;
    }
    
    @media (hover: none) and (pointer: coarse) {
      a.button:hover, button:hover {
        transform: none;
      }
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
              <p class="subtitle">Surveillance Vision Aid</p>
              <h1 class="title">Echelon Vision Command</h1>
            </div>
          </div>
          <div class="clock" id="ops-clock">--:--:-- UTC</div>
        </div>
      </div>
      <div class="stream-wrap" ondblclick="toggleStreamFullscreen()" title="Double-click to fullscreen">
        <button type="button" class="stream-fullscreen-btn" id="stream-fullscreen-btn" onclick="toggleStreamFullscreen()">Fullscreen</button>
        <img class="stream" id="stream-img" src="/video.mjpg" alt="Live detection stream" />
        <canvas id="zone-canvas" class="zone-canvas" hidden></canvas>
      </div>

      <div class="stream-report">
        <div class="metric-header">
          <div class="label">Status Report</div>
        </div>
        <div class="actions" style="margin-bottom:2px;">
          <select id="report_filter" onchange="applyReportFilter()">
            <option value="all">All Events</option>
            <option value="alerts">Alerts Only</option>
            <option value="high">High Confidence Only</option>
          </select>
        </div>
        <div id="status-report" class="report-list">
          <div class="report-item report-low">Waiting for detection events...</div>
        </div>
      </div>
    </section>
    <aside class="card side">
      <div class="sidebar-stack">
      <div class="metric sidebar-section mission-controls">
        <div class="metric-header">
          <div class="label">Mission Workflow</div>
        </div>
        <div class="sidebar-note">Pick the mission module, tune its constraints, then apply the mission state once the readiness checklist is green.</div>
        <div class="mission-grid">
          <label>Mission<select id="mission_name"></select></label>
          <label>Module<select id="mission_module"></select></label>
        </div>
        <div id="mission-summary" class="mission-summary">Select a mission module to view parameters and readiness.</div>
        <div id="module-params" class="module-params"></div>
        <div class="actions">
          <button type="button" onclick="syncMissionState()">Apply Parameters</button>
        </div>
        <div id="mission-checklist" class="checklist"></div>

        <div id="panel_facial" class="panel" hidden>
          <div class="tiny">Upload reference photos, then review browsable known face list.</div>
          <input class="inline-input" id="face_person_name" placeholder="Person name" />
          <input class="inline-input" id="face_upload_file" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" />
          <div class="actions">
            <button type="button" onclick="uploadFaceReference()">Upload Face</button>
            <button type="button" onclick="loadKnownFaces()">Refresh Face List</button>
          </div>
          <div id="face-watchlist" class="list small">No face profiles loaded yet.</div>
        </div>

        <div id="panel_plate" class="panel" hidden>
          <div class="tiny">Enter checkpoint plate watchlist (one per line or comma separated).</div>
          <textarea id="plate_watch_input" class="inline-input" rows="4" placeholder="ABC123\nPOLICE42"></textarea>
          <div class="actions">
            <button type="button" onclick="savePlateWatchlist()">Save Plate List</button>
            <button type="button" onclick="loadPlateWatchlist()">Reload Plate List</button>
          </div>
          <div id="plate-watchlist" class="list small">No plate watchlist configured.</div>
        </div>
      </div>

      <div class="metric sidebar-section">
        <div class="metric-header">
          <div class="label">Live Operation Status</div>
          <div class="signal" id="camera-signal"></div>
        </div>
        <div class="status-grid">
          <div class="status-cell">
            <span class="status-label">Camera</span>
            <span class="status-value" id="status-camera">●</span>
          </div>
          <div class="status-cell">
            <span class="status-label">Model</span>
            <span class="status-value" id="status-model">●</span>
          </div>
          <div class="status-cell">
            <span class="status-label">Zone</span>
            <span class="status-value" id="status-zone">●</span>
          </div>
        </div>
        <details class="system-detail" style="margin-top:8px;">
          <summary class="system-header">System Details</summary>
          <div class="system-content">
            <div class="list small" id="internal-status-list">
              <div>Active model: <span id="active-model">loading...</span></div>
              <div>Source: <span id="camera-source">loading...</span></div>
              <div>Active mission: <span id="internal-mission">Loading...</span></div>
              <div>Module: <span id="internal-module">Loading...</span></div>
              <div>Focus: <span id="internal-operational-focus">Loading...</span></div>
              <div>Objects/frame: <span id="detection-count">0</span></div>
            </div>
            <div class="tiny" id="internal-status-note" style="margin-top:6px;">Runtime note: waiting...</div>
          </div>
        </details>
        <details class="system-detail" style="margin-top:6px;">
          <summary class="system-header">Browser Camera</summary>
          <div class="system-content" id="browser-status" style="margin: 0;">Phone camera source inactive.</div>
        </details>
      </div>

      <div class="metric sidebar-section">
        <div class="metric-header">
          <div class="label">Detection Feed</div>
        </div>
        <div id="details" class="list small">Waiting for first frame...</div>
      </div>

      <div class="metric metric-cta sidebar-cta">
        <a class="button" href="/settings">Open Mission Settings</a>
      </div>
      </div>
    </aside>
  </main>
  <video id="browser-capture" autoplay muted playsinline style="display:none;"></video>
  <canvas id="browser-canvas" style="display:none;"></canvas>
<script>
async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function toggleStreamFullscreen() {
  const wrap = document.querySelector('.stream-wrap');
  if (!wrap) return;

  const doc = document;
  const fsElement = doc.fullscreenElement || doc.webkitFullscreenElement || doc.msFullscreenElement;
  if (fsElement === wrap) {
    if (doc.exitFullscreen) doc.exitFullscreen();
    else if (doc.webkitExitFullscreen) doc.webkitExitFullscreen();
    else if (doc.msExitFullscreen) doc.msExitFullscreen();
    return;
  }

  if (wrap.requestFullscreen) wrap.requestFullscreen();
  else if (wrap.webkitRequestFullscreen) wrap.webkitRequestFullscreen();
  else if (wrap.msRequestFullscreen) wrap.msRequestFullscreen();
}

function syncFullscreenButton() {
  const button = document.getElementById('stream-fullscreen-btn');
  const wrap = document.querySelector('.stream-wrap');
  if (!button || !wrap) return;
  const fsElement = document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement;
  button.textContent = fsElement === wrap ? 'Exit Fullscreen' : 'Fullscreen';
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
document.addEventListener('webkitfullscreenchange', syncFullscreenButton);
document.addEventListener('MSFullscreenChange', syncFullscreenButton);
async function primeBrowserSourceOnLoad() {
  try {
    const runtime = await fetchJson('/api/settings');
    runtimeCache.camera_source = runtime.camera_source || '';
    await syncBrowserMode();
  } catch {
    // Keep default status text if runtime is temporarily unavailable.
  }
}

document.addEventListener('DOMContentLoaded', () => {
  syncFullscreenButton();
  const btn = document.getElementById('stream-fullscreen-btn');
  if (btn) {
    btn.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      toggleStreamFullscreen();
    });
  }
  const stream = document.querySelector('.stream');
  if (stream) {
    stream.addEventListener('click', toggleStreamFullscreen);
  }
  hydrateViolationClasses();
  initLiveMissionUI().catch((err) => {
    appendStatusReport(`mission ui init failed: ${err.message || err}`, 0.0);
  });
  primeBrowserSourceOnLoad().catch(() => {});
  loadKnownFaces().catch(() => {});
  loadPlateWatchlist().catch(() => {});
});

const browserFeed = {
  stream: null,
  timer: null,
  uploading: false
};
let runtimeCache = {};
let lastReportKey = '';
let knownFaceProfiles = 0;
let plateWatchCount = 0;
let plateWatchValues = [];
const encounterState = { violationClasses: [] };

const DEFAULT_MISSION_CATALOG = {
  vision: {
    label: 'Vision (YOLO Toolkit)',
    modules: [
      { id: 'vision_detect', label: 'Object Detection', mode: 'detect', summary: 'Define which target objects to detect with per-object constraints inside an optional geozone.', params: [
        { key: 'classes', type: 'classlist', label: 'Target Objects', default: 'person,car' },
        { key: 'confidence', type: 'number', label: 'Confidence', default: '0.35', min: '0.01', max: '1', step: '0.01' },
        { key: 'iou', type: 'number', label: 'IoU (NMS)', default: '0.50', min: '0.01', max: '1', step: '0.01' },
        { key: 'imgsz', type: 'select', label: 'Image Size', default: '640', options: ['320','480','640','960','1280'] },
        { key: 'detect_zone', type: 'zone', shape: 'polygon', label: 'Detection Geozone', default: '' },
      ] },
      { id: 'vision_segment', label: 'Instance Segmentation', mode: 'segment', summary: 'Pixel-level masks with class filtering and mask opacity control.', params: [
        { key: 'classes', type: 'classlist', label: 'Target Objects', default: 'person' },
        { key: 'confidence', type: 'number', label: 'Confidence', default: '0.35', min: '0.01', max: '1', step: '0.01' },
        { key: 'min_area_px', type: 'number', label: 'Min Mask Area (px)', default: '500', min: '0', step: '10' },
        { key: 'opacity', type: 'number', label: 'Mask Opacity', default: '0.45', min: '0', max: '1', step: '0.05' },
        { key: 'segment_zone', type: 'zone', shape: 'polygon', label: 'Area Of Interest', default: '' },
      ] },
      { id: 'vision_pose', label: 'Pose / Keypoints', mode: 'pose', summary: 'Human skeletal keypoint estimation for posture analytics.', params: [
        { key: 'confidence', type: 'number', label: 'Confidence', default: '0.35', min: '0.01', max: '1', step: '0.01' },
        { key: 'kpt_conf', type: 'number', label: 'Keypoint Confidence', default: '0.50', min: '0.01', max: '1', step: '0.01' },
      ] },
      { id: 'vision_obb', label: 'Oriented Boxes (OBB)', mode: 'obb', summary: 'Rotated bounding boxes for aerial or angled targets.', params: [
        { key: 'classes', type: 'classlist', label: 'Target Objects', default: '' },
        { key: 'confidence', type: 'number', label: 'Confidence', default: '0.35', min: '0.01', max: '1', step: '0.01' },
        { key: 'iou', type: 'number', label: 'IoU (NMS)', default: '0.50', min: '0.01', max: '1', step: '0.01' },
      ] },
      { id: 'vision_classify', label: 'Classification', mode: 'classify', summary: 'Whole-frame classification with Top-K label readout.', params: [
        { key: 'topk', type: 'number', label: 'Top-K Labels', default: '3', min: '1', max: '10', step: '1' },
        { key: 'confidence', type: 'number', label: 'Confidence', default: '0.25', min: '0.01', max: '1', step: '0.01' },
      ] },
      { id: 'vision_track', label: 'Tracking / Counting', mode: 'track', summary: 'Persistent IDs with an optional counting line drawn on the video.', params: [
        { key: 'classes', type: 'classlist', label: 'Target Objects', default: 'person,car' },
        { key: 'tracker', type: 'select', label: 'Tracker', default: 'bytetrack.yaml', options: ['bytetrack.yaml','botsort.yaml'] },
        { key: 'persist_limit_s', type: 'number', label: 'Lost Track Timeout (s)', default: '3', min: '0', step: '0.5' },
        { key: 'count_line', type: 'zone', shape: 'line', label: 'Counting Line', default: '' },
      ] },
    ],
  },
  alpha_checkpoint: {
    label: 'Mission Alpha Checkpoint',
    modules: [
      { id: 'anpr_bolo', label: 'ANPR License Plate Reader', mode: 'detect', summary: 'BOLO license plate cross-reference at checkpoint lanes.', requires: ['plate_watchlist'], params: [
        { key: 'trigger_conf', type: 'number', label: 'Trigger Confidence', default: '0.65', min: '0.01', max: '1', step: '0.01' },
        { key: 'watch_plates', type: 'list', label: 'BOLO Plates', default: '', placeholder: 'Add plate e.g. ABC-1234' },
        { key: 'lane_gate', type: 'zone', shape: 'line', label: 'Lane Trigger Line', default: '' },
      ] },
      { id: 'speed_vector', label: 'Speed Estimation Radar', mode: 'track', summary: 'Track vectors and estimate vehicle speed from lane geometry.', params: [
        { key: 'meter_scale', type: 'number', label: 'Meter Per Pixel', default: '0.06', min: '0.001', step: '0.001' },
        { key: 'gate_line', type: 'zone', shape: 'line', label: 'Radar Gate Line', default: '' },
      ] },
      { id: 'suspect_face', label: 'Target Suspect Facial Matching', mode: 'pose', summary: 'Face watch with uploaded suspect gallery.', requires: ['face_gallery'], params: [
        { key: 'face_threshold', type: 'number', label: 'Match Threshold', default: '0.40', min: '0.01', max: '1', step: '0.01' },
        { key: 'watch_names', type: 'list', label: 'Suspect Names', default: '', placeholder: 'Add suspect name' },
      ] },
    ],
  },
  moving_violations: {
    label: 'Mission Moving Violations',
    modules: [
      { id: 'lane_encroach', label: 'Lane Encroachment Polygon', mode: 'segment', summary: 'Detect bikes or scooters violating sidewalk or bus lane zones.', params: [
        { key: 'violation_classes', type: 'classlist', label: 'Violation Objects', default: 'motorcycle,bicycle' },
        { key: 'violation_zone', type: 'zone', shape: 'polygon', label: 'Violation Zone', default: '' },
      ] },
      { id: 'red_light', label: 'Red Light Violations', mode: 'detect', summary: 'Trigger when vehicles cross stop line during red phase.', params: [
        { key: 'signal_source', type: 'select', label: 'Signal State Source', default: 'manual', options: ['manual','detector','external'] },
        { key: 'stop_line', type: 'zone', shape: 'line', label: 'Stop Line', default: '' },
      ] },
      { id: 'triple_riding', label: 'Triple Riding Counter', mode: 'track', summary: 'Spatial count of overcrowded motorcycles.', params: [
        { key: 'min_riders', type: 'number', label: 'Minimum Riders', default: '3', min: '1', max: '6', step: '1' },
      ] },
    ],
  },
  rally_square: {
    label: 'Mission Rally Square Monitoring',
    modules: [
      { id: 'threat_pose', label: 'Threat Pose Alignment', mode: 'pose', summary: 'Weapon-alignment and threat posture cueing.', params: [
        { key: 'pose_alert_ratio', type: 'number', label: 'Alert Ratio', default: '0.70', min: '0.01', max: '1', step: '0.01' },
        { key: 'watch_zone', type: 'zone', shape: 'polygon', label: 'Watch Zone', default: '' },
      ] },
      { id: 'officer_down', label: 'Officer Down Fall Detection', mode: 'pose', summary: 'Skeletal fall posture monitoring for rapid emergency response.', params: [
        { key: 'fall_window_s', type: 'number', label: 'Fall Window Seconds', default: '2.0', min: '0.5', step: '0.5' },
      ] },
    ],
  },
};
let missionCatalog = DEFAULT_MISSION_CATALOG;
const COLOR_OPTIONS = ['any','black','white','gray','red','orange','yellow','green','blue','purple','brown'];
const COLOR_SWATCHES = { any:'#888', black:'#111', white:'#eee', gray:'#888', red:'#e04545', orange:'#e08a2b', yellow:'#e0d22b', green:'#3fae5a', blue:'#3f7fd0', purple:'#9b59b6', brown:'#8b5a2b' };
const paramValues = {};

function getMissionSelection() {
  const missionId = document.getElementById('mission_name')?.value || '';
  const moduleId = document.getElementById('mission_module')?.value || '';
  const mission = missionCatalog[missionId];
  const module = mission?.modules?.find((m) => m.id === moduleId);
  return { missionId, moduleId, mission, module };
}

async function loadMissionCatalog() {
  try {
    const data = await fetchJson('/api/missions/catalog');
    if (data && data.catalog && Object.keys(data.catalog).length) {
      missionCatalog = data.catalog;
    }
  } catch {
    missionCatalog = DEFAULT_MISSION_CATALOG;
  }
}

function populateMissionNames() {
  const node = document.getElementById('mission_name');
  if (!node) return;
  node.innerHTML = Object.entries(missionCatalog)
    .map(([id, m]) => `<option value="${id}">${m.label}</option>`)
    .join('');
  if (node.options.length && !node.value) {
    node.selectedIndex = 0;
  }
}

function populateMissionModules() {
  const missionId = document.getElementById('mission_name')?.value;
  const node = document.getElementById('mission_module');
  if (!missionId || !node) return;
  const modules = missionCatalog[missionId]?.modules || [];
  node.innerHTML = modules.map((m) => `<option value="${m.id}">${m.label}</option>`).join('');
  if (node.options.length && !node.value) {
    node.selectedIndex = 0;
  }
}

function paramKey(missionId, moduleId, key) {
  return `${missionId}::${moduleId}::${key}`;
}

function getParamValue(p, def) {
  const { missionId, moduleId } = getMissionSelection();
  const stored = paramValues[paramKey(missionId, moduleId, p.key)];
  return stored !== undefined ? stored : (def !== undefined ? def : (p.default || ''));
}

function setParamValue(key, value) {
  const { missionId, moduleId } = getMissionSelection();
  paramValues[paramKey(missionId, moduleId, key)] = value;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function classOptionsList() {
  const classes = runtimeCache.available_classes || runtimeCache.model_classes || [];
  return Array.isArray(classes) ? classes : Object.values(classes || {});
}

function renderListChips(p) {
  const items = String(getParamValue(p, p.default || '') || '').split(',').map((s) => s.trim()).filter(Boolean);
  return items.map((it) => `<span class="param-chip">${escapeHtml(it)}<button type="button" onclick="paramListRemove('${p.key}','${encodeURIComponent(it)}')">&times;</button></span>`).join('');
}

function paramListAdd(key) {
  const input = document.getElementById(`list-input-${key}`);
  if (!input) return;
  const raw = (input.value || '').trim();
  if (!raw) return;
  const { module } = getMissionSelection();
  const p = (module?.params || []).find((x) => x.key === key);
  const items = String(getParamValue(p, p.default || '') || '').split(',').map((s) => s.trim()).filter(Boolean);
  raw.split(',').map((s) => s.trim()).filter(Boolean).forEach((token) => {
    if (!items.some((i) => i.toLowerCase() === token.toLowerCase())) items.push(token);
  });
  setParamValue(key, items.join(','));
  input.value = '';
  renderMissionModule();
  syncMissionState().catch(() => {});
}

function paramListRemove(key, encoded) {
  const value = decodeURIComponent(encoded);
  const { module } = getMissionSelection();
  const p = (module?.params || []).find((x) => x.key === key);
  const items = String(getParamValue(p, p.default || '') || '').split(',').map((s) => s.trim()).filter(Boolean).filter((i) => i !== value);
  setParamValue(key, items.join(','));
  renderMissionModule();
  syncMissionState().catch(() => {});
}

function paramClassToggle(key) {
  const sel = document.getElementById(`class-select-${key}`);
  if (!sel || !sel.value) return;
  paramListAddToken(key, sel.value);
  sel.value = '';
}

function paramListAddToken(key, token) {
  const { module } = getMissionSelection();
  const p = (module?.params || []).find((x) => x.key === key);
  const items = String(getParamValue(p, p.default || '') || '').split(',').map((s) => s.trim()).filter(Boolean);
  if (!items.some((i) => i.toLowerCase() === token.toLowerCase())) items.push(token);
  setParamValue(key, items.join(','));
  renderMissionModule();
  syncMissionState().catch(() => {});
}

function paramSimpleChange(key, value) {
  setParamValue(key, value);
  syncMissionState().catch(() => {});
}

function renderParam(p) {
  const fieldLabel = escapeHtml(p.label || p.key || 'Parameter');
  const val = getParamValue(p);
  if (p.type === 'zone') {
    const count = String(val || '').split(',').filter(Boolean).length / 2;
    const status = count >= 2 ? `${count} pts set` : 'not set';
    const drawing = zoneDraw.active && zoneDraw.key === p.key;
    return `<div class="param-zone"><div class="tiny">${fieldLabel} (${p.shape || 'polygon'})</div><div class="param-zone-row"><span class="param-zone-status" id="zone-status-${p.key}">${status}</span><button type="button" onclick="zoneStartDraw('${p.key}')">${drawing ? 'Done' : 'Draw'}</button><button type="button" onclick="zoneClear('${p.key}')">Clear</button></div></div>`;
  }
  if (p.type === 'list') {
    return `<div class="param-list"><div class="tiny">${fieldLabel}</div><div class="param-chips" id="list-chips-${p.key}">${renderListChips(p)}</div><div class="param-list-input"><input id="list-input-${p.key}" type="text" placeholder="${escapeHtml(p.placeholder || 'Add item')}" /><button type="button" onclick="paramListAdd('${p.key}')">Add</button></div></div>`;
  }
  if (p.type === 'classlist') {
    const opts = classOptionsList();
    const select = opts.length
      ? `<select id="class-select-${p.key}" onchange="paramClassToggle('${p.key}')"><option value="">+ add object…</option>${opts.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')}</select>`
      : `<input id="list-input-${p.key}" type="text" placeholder="Add object class" />` + `<button type="button" onclick="paramListAdd('${p.key}')">Add</button>`;
    return `<div class="param-list"><div class="tiny">${fieldLabel}${opts.length ? '' : ' (model not loaded — type manually)'}</div><div class="param-chips" id="list-chips-${p.key}">${renderListChips(p)}</div><div class="param-list-input">${select}</div></div>`;
  }
  if (p.type === 'color') {
    const opts = (p.options && p.options.length) ? p.options : COLOR_OPTIONS;
    const cur = String(val || 'any');
    return `<label>${fieldLabel}<span class="param-zone-row"><span class="param-swatch" style="background:${COLOR_SWATCHES[cur] || '#888'}"></span><select data-param-key="${p.key}" onchange="paramSimpleChange('${p.key}', this.value); renderMissionModule();">${opts.map((o) => `<option value="${escapeHtml(o)}" ${String(o) === cur ? 'selected' : ''}>${escapeHtml(o)}</option>`).join('')}</select></span></label>`;
  }
  if (p.type === 'select' || (Array.isArray(p.options) && p.options.length)) {
    const opts = p.options || [];
    return `<label>${fieldLabel}<select data-param-key="${p.key}">${opts.map((o) => `<option value="${escapeHtml(o)}" ${String(o) === String(val) ? 'selected' : ''}>${escapeHtml(o)}</option>`).join('')}</select></label>`;
  }
  if (p.type === 'number') {
    const min = p.min !== undefined ? ` min="${p.min}"` : '';
    const max = p.max !== undefined ? ` max="${p.max}"` : '';
    const step = p.step !== undefined ? p.step : '0.01';
    return `<label>${fieldLabel}<input data-param-key="${p.key}" type="number" step="${step}"${min}${max} value="${escapeHtml(val)}" /></label>`;
  }
  if (p.type === 'object_constraints') {
    return renderObjectConstraints(p);
  }
  if (p.type === 'bool') {
    const checked = String(val) === 'true' || val === true ? 'checked' : '';
    return `<label class="param-bool"><input type="checkbox" data-param-key="${p.key}" ${checked} onchange="paramBoolChange('${p.key}', this.checked)" /><span>${fieldLabel}</span></label>`;
  }
  return `<label>${fieldLabel}<input data-param-key="${p.key}" type="text" value="${escapeHtml(val)}" /></label>`;
}

// ============ Per-Object Constraint Management ============
function parseObjectConstraints(val) {
  if (!val) return [];
  try {
    const arr = JSON.parse(String(val || '[]'));
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function stringifyObjectConstraints(arr) {
  return JSON.stringify(arr || []);
}

function paramBoolChange(key, checked) {
  setParamValue(key, checked ? 'true' : 'false');
  // Immediately push to runtime for instant visual feedback
  const runtimeKey = key === 'show_bbox' ? 'show_bbox' : key === 'show_masks' ? 'show_masks' : null;
  if (runtimeKey) {
    fetchJson('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [runtimeKey]: checked }),
    }).catch(() => {});
  }
  syncMissionState().catch(() => {});
}

function renderObjectConstraints(p) {
  const val = getParamValue(p);
  const objects = parseObjectConstraints(val);
  const availableClasses = classOptionsList();
  const COLOR_OPTS = ['any','black','white','red','orange','yellow','green','blue','purple','grey'];

  // One table row per tracked object — columns: Class | Conf ≥ | Area ≥ px² | Color | Zone | ×
  const rows = objects.map((obj, idx) => {
    const cls  = escapeHtml(obj.class || '?');
    const conf = obj.confidence !== undefined && obj.confidence !== '' ? obj.confidence : '';
    const area = obj.area_min  !== undefined && obj.area_min  !== '' ? obj.area_min  : '';
    const col  = obj.color || 'any';
    const zoneSet = obj.zone_pts && String(obj.zone_pts).split(',').filter(Boolean).length >= 4;
    const zoneLabel = zoneSet ? '✓ set' : '—';
    const colOpts = COLOR_OPTS.map((o) => `<option value="${o}"${o===col?' selected':''}>${o}</option>`).join('');
    return `<tr data-obj-idx="${idx}">
      <td class="cls-cell">${cls}</td>
      <td><input type="number" min="0" max="1" step="0.01" value="${escapeHtml(String(conf))}" placeholder="0.35"
            onchange="objectConstraintSetField('${p.key}',${idx},'confidence',+this.value)" /></td>
      <td><input type="number" min="0" step="10" value="${escapeHtml(String(area))}" placeholder="—"
            onchange="objectConstraintSetField('${p.key}',${idx},'area_min',+this.value)" /></td>
      <td><select onchange="objectConstraintSetField('${p.key}',${idx},'color',this.value)">${colOpts}</select></td>
      <td class="zone-cell"><button type="button" onclick="objectConstraintDrawZone('${p.key}',${idx})">${zoneLabel}</button></td>
      <td><button type="button" class="btn-tiny" title="Remove" onclick="objectConstraintRemove('${p.key}',${idx})">✕</button></td>
    </tr>`;
  }).join('');

  const emptyRow = objects.length === 0
    ? `<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:10px 0;font-size:0.75rem;">No target objects defined — add one below</td></tr>`
    : '';

  const classOptions = availableClasses.length
    ? `<select id="new-obj-class-${p.key}"><option value="">Select class…</option>${availableClasses.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')}</select>`
    : `<input id="new-obj-class-${p.key}" type="text" placeholder="Class name (e.g. person)" />`;

  return `<div class="param-object-constraints">
    <table class="obj-table">
      <thead><tr>
        <th>Class</th><th>Conf ≥</th><th>Area ≥ px²</th><th>Color</th><th>Zone</th><th></th>
      </tr></thead>
      <tbody>${rows}${emptyRow}</tbody>
    </table>
    <div class="add-object-row">
      ${classOptions}
      <button type="button" onclick="objectConstraintAdd('${p.key}')">+ Add Target</button>
    </div>
  </div>`;
}

function objectConstraintAdd(key) {
  const classInput = document.getElementById(`new-obj-class-${key}`);
  if (!classInput || !classInput.value) return;
  const cls = classInput.value.trim();
  if (!cls) return;
  const { module } = getMissionSelection();
  const p = (module?.params || []).find((x) => x.key === key);
  const objects = parseObjectConstraints(getParamValue(p));
  // Prevent duplicate class entries
  if (objects.some((o) => String(o.class).toLowerCase() === cls.toLowerCase())) {
    classInput.value = '';
    return;
  }
  objects.push({ class: cls, confidence: '', area_min: '', color: 'any', zone_pts: '' });
  setParamValue(key, stringifyObjectConstraints(objects));
  classInput.value = '';
  renderMissionModule();
  syncMissionState().catch(() => {});
}

function objectConstraintRemove(key, idx) {
  const { module } = getMissionSelection();
  const p = (module?.params || []).find((x) => x.key === key);
  const objects = parseObjectConstraints(getParamValue(p));
  objects.splice(idx, 1);
  setParamValue(key, stringifyObjectConstraints(objects));
  renderMissionModule();
  syncMissionState().catch(() => {});
}

function objectConstraintSetField(key, idx, field, value) {
  const { module } = getMissionSelection();
  const p = (module?.params || []).find((x) => x.key === key);
  const objects = parseObjectConstraints(getParamValue(p));
  if (!objects[idx]) return;
  objects[idx][field] = value;
  setParamValue(key, stringifyObjectConstraints(objects));
  syncMissionState().catch(() => {});
}

function objectConstraintDrawZone(key, idx) {
  // Reuse the zone drawing canvas; store result back into the object's zone_pts
  // For now open a simple prompt — full canvas integration is a follow-up
  const { module } = getMissionSelection();
  const p = (module?.params || []).find((x) => x.key === key);
  const objects = parseObjectConstraints(getParamValue(p));
  if (!objects[idx]) return;
  // Clear zone_pts to signal "draw requested" — canvas integration can hook here
  objects[idx].zone_pts = '';
  setParamValue(key, stringifyObjectConstraints(objects));
  renderMissionModule();
}

function renderMissionModule() {
  const { module } = getMissionSelection();
  const summary = document.getElementById('mission-summary');
  const paramsNode = document.getElementById('module-params');
  if (!summary || !paramsNode) return;
  if (!module) {
    summary.textContent = 'Select a mission module to view parameters and readiness.';
    paramsNode.innerHTML = '';
    return;
  }

  summary.textContent = `${module.label} | Mode ${String(module.mode || 'detect').toUpperCase()} | ${module.summary || 'No module summary available.'}`;
  
  const params = module.params || [];
  const hasObjectConstraints = params.some((p) => p.type === 'object_constraints');
  const targetParams = params.filter((p) => p.workflow === 'target' || p.type === 'object_constraints' || (p.type === 'classlist' && !hasObjectConstraints));
  const constraintParams = params.filter((p) => p.workflow === 'constraint' || ['confidence'].includes(p.key));
  const validationParams = params.filter((p) => p.workflow === 'validation' || p.type === 'zone');
  const actionParams = params.filter((p) => p.workflow === 'action' || ['watch_plates', 'watch_names', 'violation_classes'].includes(p.key));
  const otherParams = params.filter((p) => !targetParams.includes(p) && !constraintParams.includes(p) && !validationParams.includes(p) && !actionParams.includes(p) && !(hasObjectConstraints && p.type === 'classlist'));
  
  let html = '';
  if (targetParams.length) {
    html += `<div class="param-section"><div class="section-header">◆ Target Selection</div>${targetParams.map(renderParam).join('')}</div>`;
  }
  if (constraintParams.length) {
    html += `<div class="param-section"><div class="section-header">◆ Detection Constraints</div>${constraintParams.map(renderParam).join('')}</div>`;
  }
  if (validationParams.length) {
    html += `<div class="param-section"><div class="section-header">◆ Monitoring Zone</div>${validationParams.map(renderParam).join('')}</div>`;
  }
  if (actionParams.length) {
    html += `<div class="param-section"><div class="section-header">◆ Watch List</div>${actionParams.map(renderParam).join('')}</div>`;
  }
  if (otherParams.length) {
    html += `<div class="param-section">${otherParams.map(renderParam).join('')}</div>`;
  }
  
  paramsNode.innerHTML = html || params.map(renderParam).join('');

  const zoneKeys = new Set(params.filter((p) => p.type === 'zone').map((p) => p.key));
  if (!zoneKeys.size || (zoneDraw.active && !zoneKeys.has(zoneDraw.key))) {
    zoneHideCanvas();
  }

  if (module.mode) {
    quickSetMode(module.mode).catch(() => {});
  }
  updateMissionChecklist();
  setMissionPanel();
}

function updateMissionChecklist() {
  const node = document.getElementById('mission-checklist');
  if (!node) return;
  const { module } = getMissionSelection();
  if (!module) {
    node.innerHTML = '';
    return;
  }

  const checks = [
    { label: 'Camera', ready: !!runtimeCache.camera_open || !!runtimeCache.camera_source },
    { label: 'Model', ready: !!runtimeCache.active_model },
  ];
  if ((module.requires || []).includes('face_gallery')) {
    checks.push({ label: 'Faces', ready: knownFaceProfiles > 0 });
  }
  if ((module.requires || []).includes('plate_watchlist')) {
    checks.push({ label: 'Plates', ready: plateWatchCount > 0 });
  }

  node.innerHTML = checks
    .map((c) => `<div class="check-item ${c.ready ? 'ready' : ''}"><span class="check-light"></span><span>${c.label}</span><span class="check-state">${c.ready ? 'ready' : 'wait'}</span></div>`)
    .join('');
}

function collectModuleParams() {
  const { missionId, moduleId, module } = getMissionSelection();
  const out = {};
  // Seed from stored values (lists, classlists, zones, colors).
  for (const p of (module?.params || [])) {
    const stored = paramValues[paramKey(missionId, moduleId, p.key)];
    if (stored !== undefined) {
      out[p.key] = stored;
    } else if (['list', 'classlist', 'zone', 'color', 'select'].includes(p.type)) {
      out[p.key] = p.default || '';
    }
  }
  // Overlay direct input fields (text, number, select).
  const node = document.getElementById('module-params');
  if (node) {
    for (const field of node.querySelectorAll('[data-param-key]')) {
      const key = field.getAttribute('data-param-key');
      if (!key) continue;
      const val = field.type === 'checkbox' ? String(field.checked) : field.value;
      out[key] = val;
      setParamValue(key, val);
    }
  }
  return out;
}

async function syncMissionState(extra = {}) {
  const { missionId, moduleId } = getMissionSelection();
  const payload = {
    mission_id: missionId,
    module_id: moduleId,
    parameters: collectModuleParams(),
    ...extra,
  };
  const data = await fetchJson('/api/missions/state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  appendStatusReport(`mission state synced: ${moduleId || 'none'}`, 0.55);
  return data;
}

async function loadMissionState() {
  const data = await fetchJson('/api/missions/state');
  const state = data.state || {};
  const mission = document.getElementById('mission_name');
  const module = document.getElementById('mission_module');
  if (mission && state.mission_id && missionCatalog[state.mission_id]) {
    mission.value = state.mission_id;
  }
  populateMissionModules();
  if (module && state.module_id) {
    module.value = state.module_id;
  }
  if (state.parameters && typeof state.parameters === 'object') {
    for (const [key, value] of Object.entries(state.parameters)) {
      setParamValue(key, value);
    }
  }
  renderMissionModule();
  if (data.readiness && Array.isArray(data.readiness.checks)) {
    appendStatusReport(`mission ready: ${data.readiness.ready ? 'yes' : 'no'}`, data.readiness.ready ? 0.8 : 0.3);
  }
}

function readViolationClasses() {
  const input = document.getElementById('violation-classes');
  if (!input) return [];
  return input.value
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}

function hydrateViolationClasses() {
  const input = document.getElementById('violation-classes');
  if (!input) return;
  const saved = localStorage.getItem('echelon.violation.classes');
  if (saved) input.value = saved;
  encounterState.violationClasses = readViolationClasses();
  input.addEventListener('change', () => {
    localStorage.setItem('echelon.violation.classes', input.value);
    encounterState.violationClasses = readViolationClasses();
  });
}

function setMissionPanel() {
  const mode = getMissionSelection().module?.id || '';
  const panelFacial = document.getElementById('panel_facial');
  const panelPlate = document.getElementById('panel_plate');
  if (panelFacial) panelFacial.hidden = !mode.includes('face');
  if (panelPlate) panelPlate.hidden = !mode.includes('plate') && !mode.includes('anpr');

  if (!panelFacial?.hidden) {
    loadKnownFaces().catch(() => {});
  } else if (!panelPlate?.hidden) {
    loadPlateWatchlist().catch(() => {});
  }
}

function appendStatusReport(text, confidence=0.0, options={}) {
  const report = document.getElementById('status-report');
  if (!report) return;
  const item = document.createElement('div');
  const critical = !!options.critical;
  const levelClass = critical ? 'report-critical' : (confidence >= 0.75 ? 'report-high' : (confidence >= 0.45 ? 'report-mid' : 'report-low'));
  item.className = `report-item ${levelClass}`;
  item.dataset.severity = critical ? 'critical' : (confidence >= 0.75 ? 'high' : (confidence >= 0.45 ? 'mid' : 'low'));
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  const tag = critical ? 'MATCH' : 'EVENT';
  item.textContent = `${ts} | ${tag} | ${text} | conf ${confidence.toFixed(2)}`;
  report.prepend(item);
  while (report.childElementCount > 28) {
    report.removeChild(report.lastElementChild);
  }
  applyReportFilter();
}

function normalizePlateToken(value) {
  return String(value || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
}

function getPlateTokens(det) {
  const attrs = det?.attributes && typeof det.attributes === 'object' ? det.attributes : {};
  const fields = [det?.label, attrs.plate, attrs.plate_text, attrs.text, attrs.ocr, attrs.value];
  return fields
    .map((v) => normalizePlateToken(v))
    .filter((v) => v.length >= 4);
}

function toLowerList(value) {
  return String(value || '')
    .split(',')
    .map((v) => v.trim().toLowerCase())
    .filter(Boolean);
}

function activeMissionParams(status, module) {
  const defaults = {};
  for (const p of (module?.params || [])) {
    defaults[p.key] = p.default || '';
  }
  const stateParams = status?.mission_state?.parameters && typeof status.mission_state.parameters === 'object'
    ? status.mission_state.parameters
    : {};
  return { ...defaults, ...stateParams };
}

function detectionColorToken(det) {
  const attrs = det?.attributes && typeof det.attributes === 'object' ? det.attributes : {};
  return String(attrs.color || attrs.colour || attrs.dominant_color || '').toLowerCase().trim();
}

function detectionNameTokens(det) {
  const attrs = det?.attributes && typeof det.attributes === 'object' ? det.attributes : {};
  return [det?.label, attrs.name, attrs.person, attrs.identity]
    .map((v) => String(v || '').trim().toLowerCase())
    .filter(Boolean);
}

function confidenceThresholdFor(moduleId, params) {
  const keys = ['trigger_conf', 'confidence', 'kpt_conf', 'face_threshold'];
  for (const key of keys) {
    const v = Number(params?.[key]);
    if (!Number.isNaN(v) && v > 0 && v <= 1) return v;
  }
  return moduleId === 'anpr_bolo' ? 0.65 : 0.35;
}

function evaluateMissionMatches(moduleId, detections, params) {
  // Check if per-object constraints are enabled
  let perObjectObjs = [];
  if (params?.detection_objects) {
    try {
      const parsed = JSON.parse(String(params.detection_objects || '[]'));
      if (Array.isArray(parsed) && parsed.length) {
        perObjectObjs = parsed;
      }
    } catch {
      // Fall through to legacy mode
    }
  }

  // === PER-OBJECT MODE ===
  if (perObjectObjs.length) {
    const matched = [];
    const matchedPlates = [];
    const matchedNames = [];

    for (const objDef of perObjectObjs) {
      const targetClass = String(objDef.class || '').toLowerCase();

      for (const det of detections) {
        const cls = String(det?.class_name || '').toLowerCase();
        if (cls !== targetClass) continue;

        // Flat constraint fields: confidence, area_min, color
        const minConf = objDef.confidence !== '' && objDef.confidence !== undefined ? Number(objDef.confidence) : null;
        const minArea = objDef.area_min   !== '' && objDef.area_min   !== undefined ? Number(objDef.area_min)   : null;
        const colorFilter = String(objDef.color || 'any').toLowerCase();

        if (minConf !== null && Number(det?.confidence || 0) < minConf) continue;
        if (minArea !== null) {
          const area = (det?.box?.width || 0) * (det?.box?.height || 0);
          if (area < minArea) continue;
        }
        if (colorFilter && colorFilter !== 'any') {
          const detColor = detectionColorToken(det);
          if (!detColor.includes(colorFilter)) continue;
        }

        matched.push(det);

        // Check for plates
        for (const token of getPlateTokens(det)) {
          if (!matchedPlates.includes(token)) matchedPlates.push(token);
        }

        // Check for names
        for (const token of detectionNameTokens(det)) {
          if (!matchedNames.includes(token)) matchedNames.push(token);
        }
      }
    }

    return {
      threshold: 0,
      matched,
      matchedPlates,
      matchedNames,
    };
  }

  // === LEGACY GLOBAL MODE ===
  const threshold = confidenceThresholdFor(moduleId, params);
  const targetClasses = new Set([
    ...toLowerList(params?.classes),
    ...toLowerList(params?.violation_classes),
  ]);
  const colorFilter = String(params?.color_filter || '').trim().toLowerCase();
  const watchPlateSet = new Set(toLowerList(params?.watch_plates).map((p) => normalizePlateToken(p)).filter(Boolean));
  const watchNameSet = new Set(toLowerList(params?.watch_names));

  const matched = [];
  for (const det of detections) {
    const conf = Number(det?.confidence || 0);
    if (conf < threshold) continue;
    const cls = String(det?.class_name || '').toLowerCase();
    const clsOk = !targetClasses.size || targetClasses.has(cls);
    if (!clsOk) continue;

    const color = detectionColorToken(det);
    const colorOk = !colorFilter || colorFilter === 'any' || color.includes(colorFilter);
    if (!colorOk) continue;

    matched.push(det);
  }

  const matchedPlates = [];
  if (watchPlateSet.size) {
    for (const det of detections) {
      for (const token of getPlateTokens(det)) {
        if (watchPlateSet.has(token)) matchedPlates.push(token);
      }
    }
  }

  const matchedNames = [];
  if (watchNameSet.size) {
    for (const det of detections) {
      for (const token of detectionNameTokens(det)) {
        if (watchNameSet.has(token)) matchedNames.push(token);
      }
    }
  }

  return {
    threshold,
    matched,
    matchedPlates: [...new Set(matchedPlates)],
    matchedNames: [...new Set(matchedNames)],
  };
}

function buildMissionReportEvent(status) {
  const { moduleId, module } = getMissionSelection();
  const detections = Array.isArray(status?.detections) ? status.detections : [];
  if (!moduleId || !detections.length) return null;
  const params = activeMissionParams(status, module);
  const evaluated = evaluateMissionMatches(moduleId, detections, params);

  const vehicles = countByClass(detections, ['car', 'truck', 'bus', 'motorcycle']);
  const people = countByClass(detections, ['person']);
  const plateDetections = detections.filter((d) => {
    const cls = String(d.class_name || '').toLowerCase();
    return cls.includes('plate') || cls === 'license_plate' || cls === 'license-plate';
  });
  const faceDetections = detections.filter((d) => String(d.kind || '').toLowerCase() === 'face');
  const faceMatches = faceDetections.filter((d) => String(d.label || '').toLowerCase() !== 'unknown');
  const poseDetections = detections.filter((d) => String(d.kind || '').toLowerCase() === 'pose');

  if (moduleId === 'anpr_bolo') {
    const watch = new Set(plateWatchValues.map((p) => normalizePlateToken(p)).filter(Boolean));
    const hits = [];
    let hitConf = 0;
    for (const det of plateDetections) {
      for (const token of getPlateTokens(det)) {
        if (watch.has(token)) {
          hits.push(token);
          hitConf = Math.max(hitConf, Number(det.confidence || 0));
        }
      }
    }
    if (hits.length) {
      const uniqueHits = [...new Set(hits)].slice(0, 3);
      return {
        key: `anpr-hit:${uniqueHits.join(',')}`,
        text: `ANPR watchlist HIT: ${uniqueHits.join(', ')}`,
        confidence: Math.max(0.92, hitConf),
        critical: true,
      };
    }
    if (evaluated.matchedPlates.length) {
      return {
        key: `anpr-param-match:${evaluated.matchedPlates.slice(0, 3).join(',')}`,
        text: `ANPR parameter match on watch_plates: ${evaluated.matchedPlates.slice(0, 3).join(', ')}`,
        confidence: 0.96,
        critical: true,
      };
    }
    if (plateDetections.length) {
      const top = plateDetections[0];
      return {
        key: `anpr-read:${plateDetections.length}:${Math.round((top.confidence || 0) * 100)}`,
        text: `ANPR read ${plateDetections.length} plate candidate(s)` + (vehicles ? ` with ${vehicles} vehicle cue(s)` : ''),
        confidence: Number(top.confidence || 0),
      };
    }
    return null;
  }

  if (moduleId === 'speed_vector' && vehicles > 0) {
    const top = detections[0];
    return {
      key: `speed-vector:${vehicles}:${Math.round((top.confidence || 0) * 100)}`,
      text: `Speed vector tracking on ${vehicles} vehicle candidate(s)`,
      confidence: Number(top.confidence || 0),
    };
  }

  if (moduleId === 'suspect_face' && faceDetections.length) {
    const top = faceDetections[0];
    if (evaluated.matchedNames.length) {
      return {
        key: `face-name-match:${evaluated.matchedNames.slice(0, 2).join(',')}`,
        text: `Face parameter match on watch_names: ${evaluated.matchedNames.slice(0, 2).join(', ')}`,
        confidence: Math.max(0.9, Number(top.confidence || 0)),
        critical: true,
      };
    }
    if (faceMatches.length) {
      const names = [...new Set(faceMatches.map((d) => String(d.label || '').trim()).filter(Boolean))].slice(0, 2);
      return {
        key: `face-hit:${names.join(',')}`,
        text: `Face match candidate: ${names.join(', ')}`,
        confidence: Math.max(0.85, Number(top.confidence || 0)),
        critical: true,
      };
    }
    return {
      key: `face-scan:${faceDetections.length}:${Math.round((top.confidence || 0) * 100)}`,
      text: `Face scan active: ${faceDetections.length} face(s) in frame`,
      confidence: Number(top.confidence || 0),
    };
  }

  if (moduleId === 'lane_encroach' && vehicles > 0) {
    return {
      key: `lane-encroach:${vehicles}`,
      text: `Lane encroachment monitor flagged ${vehicles} zone candidate(s)`,
      confidence: 0.6,
    };
  }

  if (moduleId === 'red_light' && vehicles > 0) {
    return {
      key: `red-light:${vehicles}`,
      text: `Red-light monitor tracking ${vehicles} crossing candidate(s)`,
      confidence: 0.62,
    };
  }

  if (moduleId === 'triple_riding' && (people > 0 || vehicles > 0)) {
    return {
      key: `triple-riding:${people}:${vehicles}`,
      text: `Rider load cues: ${people} person(s), ${vehicles} vehicle/motorcycle object(s)`,
      confidence: 0.58,
    };
  }

  if (moduleId === 'threat_pose' && poseDetections.length) {
    return {
      key: `threat-pose:${poseDetections.length}`,
      text: `Threat posture analysis on ${poseDetections.length} pose track(s)`,
      confidence: 0.66,
    };
  }

  if (moduleId === 'officer_down' && poseDetections.length) {
    return {
      key: `officer-down:${poseDetections.length}`,
      text: `Officer-down fall monitor evaluating ${poseDetections.length} pose track(s)`,
      confidence: 0.64,
    };
  }

  if (evaluated.matched.length) {
    const sample = evaluated.matched[0];
    const cls = String(sample.class_name || 'object');
    const c = detectionColorToken(sample);
    const colorText = c ? `, color ${c}` : '';
    
    // Check if per-object constraints are active (more specific reporting)
    let perObjectObjs = [];
    if (params?.detection_objects) {
      try {
        const parsed = JSON.parse(String(params.detection_objects || '[]'));
        if (Array.isArray(parsed) && parsed.length) {
          perObjectObjs = parsed;
        }
      } catch {}
    }
    
    if (perObjectObjs.length) {
      // Per-object constraint mode: report matched objects with constraints applied
      const constraintCount = perObjectObjs.reduce((sum, obj) => {
        let n = 0;
        if (obj.confidence !== '' && obj.confidence !== undefined) n++;
        if (obj.area_min   !== '' && obj.area_min   !== undefined) n++;
        if (obj.color && obj.color !== 'any') n++;
        return sum + n;
      }, 0);
      return {
        key: `obj-match:${moduleId}:${cls}:${constraintCount}:${evaluated.matched.length}`,
        text: `Object constraint match: ${evaluated.matched.length} ${cls} detected with ${constraintCount} constraint rule(s) active`,
        confidence: Math.max(0.88, Number(sample.confidence || 0)),
        critical: true,
      };
    }
    
    // Legacy mode: report parameter matches
    return {
      key: `param-match:${moduleId}:${cls}:${Math.round((sample.confidence || 0) * 100)}:${evaluated.matched.length}`,
      text: `Parameter match: ${evaluated.matched.length} ${cls} target(s) met threshold ${evaluated.threshold.toFixed(2)}${colorText}`,
      confidence: Math.max(0.88, Number(sample.confidence || 0)),
      critical: true,
    };
  }

  const top = detections[0];
  return {
    key: `generic:${moduleId}:${top.kind}:${top.class_name}:${Math.round((top.confidence || 0) * 100)}`,
    text: `Detection activity: ${top.class_name} (${top.kind}) @ ${top.confidence.toFixed(2)} confidence`,
    confidence: Number(top.confidence || 0),
  };
}

function applyReportFilter() {
  const filter = document.getElementById('report_filter')?.value || 'all';
  const items = Array.from(document.querySelectorAll('#status-report .report-item'));
  for (const item of items) {
    const severity = item.dataset.severity || 'low';
    const show =
      filter === 'all' ||
      (filter === 'alerts' && severity !== 'low') ||
      (filter === 'high' && (severity === 'high' || severity === 'critical'));
    item.style.display = show ? '' : 'none';
  }
}

async function quickSetCamera(source) {
  runtimeCache.camera_source = source;
  await syncBrowserMode();
  await fetchJson('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ camera_source: source })
  });
  await refresh();
}

async function quickSetMode(mode) {
  await fetchJson('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inference_mode: mode })
  });
  await refresh();
}

async function loadKnownFaces() {
  const data = await fetchJson('/api/faces');
  const node = document.getElementById('face-watchlist');
  knownFaceProfiles = Number(data.known_profiles || 0);
  if (!node) return;
  const people = data.people || [];
  if (!people.length) {
    node.textContent = 'No face profiles loaded yet.';
    return;
  }
  node.innerHTML = people.map((p) => `<div>• ${p.name} (${p.count} image)</div>`).join('');
  updateMissionChecklist();
}

async function uploadFaceReference() {
  const person = document.getElementById('face_person_name');
  const fileInput = document.getElementById('face_upload_file');
  if (!person || !fileInput || !fileInput.files || !fileInput.files.length) {
    appendStatusReport('Face upload skipped: missing person name or image file', 0.0);
    return;
  }
  const form = new FormData();
  form.append('person', person.value || 'Unknown');
  form.append('image', fileInput.files[0]);
  await fetchJson('/api/faces/upload', { method: 'POST', body: form });
  await loadKnownFaces();
  appendStatusReport(`Face profile uploaded for ${person.value || 'Unknown'}`, 0.99);
}

async function loadPlateWatchlist() {
  const data = await fetchJson('/api/watchlist/plates');
  const node = document.getElementById('plate-watchlist');
  const input = document.getElementById('plate_watch_input');
  const plates = data.plates || [];
  plateWatchValues = plates.slice();
  plateWatchCount = plates.length;
  if (input) input.value = plates.join('\\n');
  if (!node) return;
  if (!plates.length) {
    node.textContent = 'No plate watchlist configured.';
    return;
  }
  node.innerHTML = plates.map((plate) => `<div>• ${plate}</div>`).join('');
  updateMissionChecklist();
}

async function savePlateWatchlist() {
  const input = document.getElementById('plate_watch_input');
  if (!input) return;
  const values = input.value
    .split(/[\\n,]/)
    .map((v) => v.trim())
    .filter(Boolean);
  const data = await fetchJson('/api/watchlist/plates', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plates: values })
  });
  const node = document.getElementById('plate-watchlist');
  if (node) {
    const plates = data.plates || [];
    plateWatchValues = plates.slice();
    plateWatchCount = plates.length;
    node.innerHTML = plates.length ? plates.map((plate) => `<div>• ${plate}</div>`).join('') : 'No plate watchlist configured.';
  }
  appendStatusReport(`Saved checkpoint plate list (${(data.plates || []).length} entries)`, 0.99);
  updateMissionChecklist();
}

function countByClass(detections, classes) {
  const wanted = new Set(classes.map((c) => String(c || '').toLowerCase()));
  return detections.filter((d) => wanted.has(String(d.class_name || '').toLowerCase())).length;
}

function moduleSignalSummary(moduleId, status, module) {
  const detections = Array.isArray(status?.detections) ? status.detections : [];
  const people = countByClass(detections, ['person']);
  const vehicles = countByClass(detections, ['car', 'truck', 'bus', 'motorcycle']);
  const plates = detections.filter((d) => {
    const cls = String(d.class_name || '').toLowerCase();
    return cls.includes('plate') || cls === 'license_plate' || cls === 'license-plate';
  }).length;
  const faceDetections = detections.filter((d) => String(d.kind || '').toLowerCase() === 'face').length;
  const faceMatches = detections.filter((d) => String(d.kind || '').toLowerCase() === 'face' && String(d.label || '').toLowerCase() !== 'unknown').length;
  const poseDetections = detections.filter((d) => String(d.kind || '').toLowerCase() === 'pose').length;

  if (moduleId === 'anpr_bolo') {
    const ready = plateWatchCount > 0 ? 'watchlist ready' : 'watchlist missing';
    return {
      focus: `ANPR checkpoint scan (${ready})`,
      signal: `${plates} plate read(s), ${vehicles} vehicle target(s) this frame`,
    };
  }
  if (moduleId === 'speed_vector') {
    return {
      focus: 'Vehicle speed vector estimation lanes',
      signal: `${vehicles} tracked vehicle candidate(s) this frame`,
    };
  }
  if (moduleId === 'suspect_face') {
    const gallery = knownFaceProfiles > 0 ? `${knownFaceProfiles} profile(s) loaded` : 'gallery missing';
    return {
      focus: `Suspect face verification (${gallery})`,
      signal: `${faceDetections} face(s), ${faceMatches} match candidate(s)`,
    };
  }
  if (moduleId === 'crowd_density') {
    return {
      focus: 'Crowd density and bottleneck watch',
      signal: `${people} person detection(s) in current frame`,
    };
  }
  if (moduleId === 'lane_encroach') {
    return {
      focus: 'Lane and sidewalk encroachment watch',
      signal: `${vehicles} vehicle/bike candidate(s) in zone view`,
    };
  }
  if (moduleId === 'red_light') {
    return {
      focus: 'Stop-line crossing during red phase',
      signal: `${vehicles} crossing candidate(s) awaiting phase confirmation`,
    };
  }
  if (moduleId === 'triple_riding') {
    return {
      focus: 'Overloaded motorcycle rider counting',
      signal: `${people} person cue(s), ${vehicles} motorcycle/vehicle cue(s)`,
    };
  }
  if (moduleId === 'threat_pose') {
    return {
      focus: 'Threat posture alignment and escalation cues',
      signal: `${poseDetections} pose track(s) under posture analysis`,
    };
  }
  if (moduleId === 'officer_down') {
    return {
      focus: 'Officer down fall posture monitoring',
      signal: `${poseDetections} pose track(s) under fall window checks`,
    };
  }

  const modeText = String(module?.mode || 'detect').toUpperCase();
  return {
    focus: module?.summary || `${modeText} mission monitoring`,
    signal: `${detections.length} detection(s) in current frame`,
  };
}

function updateOperationalCounters(status) {
  const cameraNode = document.getElementById('internal-camera-status');
  if (cameraNode) {
    cameraNode.textContent = status.camera_open
      ? `${status.width}x${status.height} @ frame ${status.frame_count}`
      : (status.last_error || 'camera offline');
  }

  const { missionId, moduleId, mission, module } = getMissionSelection();
  const missionNode = document.getElementById('internal-mission');
  if (missionNode) {
    missionNode.textContent = mission?.label || missionId || 'n/a';
  }
  const moduleNode = document.getElementById('internal-module');
  if (moduleNode) {
    moduleNode.textContent = module?.label || moduleId || 'n/a';
  }

  const summary = moduleSignalSummary(moduleId, status, module);
  const focusNode = document.getElementById('internal-operational-focus');
  if (focusNode) {
    focusNode.textContent = summary.focus;
  }
  const signalNode = document.getElementById('internal-mission-signal');
  if (signalNode) {
    signalNode.textContent = summary.signal;
  }
}

function setBrowserStatus(text) {
  const node = document.getElementById('browser-status');
  if (node) node.textContent = text;
}

function isBrowserSource(source) {
  return String(source || '').startsWith('browser://');
}

async function startBrowserCameraUpload() {
  if (browserFeed.stream) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setBrowserStatus('Browser camera API unavailable. Use a browser with camera support.');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' } },
      audio: false
    });
    browserFeed.stream = stream;
    const video = document.getElementById('browser-capture');
    video.srcObject = stream;
    await video.play();

    browserFeed.timer = setInterval(pushBrowserFrame, 180);
    setBrowserStatus('Phone camera connected. Uploading frames to server...');
  } catch (err) {
    setBrowserStatus('Camera permission failed: ' + (err.message || String(err)));
  }
}

function stopBrowserCameraUpload() {
  if (browserFeed.timer) {
    clearInterval(browserFeed.timer);
    browserFeed.timer = null;
  }
  if (browserFeed.stream) {
    for (const track of browserFeed.stream.getTracks()) track.stop();
    browserFeed.stream = null;
  }
}

async function pushBrowserFrame() {
  if (browserFeed.uploading) return;
  const video = document.getElementById('browser-capture');
  const canvas = document.getElementById('browser-canvas');
  if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  browserFeed.uploading = true;
  canvas.toBlob(async (blob) => {
    try {
      if (!blob) return;
      const form = new FormData();
      form.append('frame', blob, 'frame.jpg');
      await fetch('/api/browser-frame', { method: 'POST', body: form });
    } catch {
      setBrowserStatus('Frame upload failed. Check network/API status.');
    } finally {
      browserFeed.uploading = false;
    }
  }, 'image/jpeg', 0.72);
}

async function syncBrowserMode() {
  if (isBrowserSource(runtimeCache.camera_source)) {
    await startBrowserCameraUpload();
  } else {
    stopBrowserCameraUpload();
    setBrowserStatus('Phone camera source inactive. Set camera source to browser://webui to enable.');
  }
}

function updateStatusIndicators(status, runtime) {
  // Camera status
  const cameraOk = !!status?.camera_open;
  const cameraEl = document.getElementById('status-camera');
  if (cameraEl) {
    cameraEl.textContent = cameraOk ? '✓' : '✗';
    cameraEl.className = `status-value ${cameraOk ? 'ok' : 'error'}`;
  }

  // Model status
  const modelOk = !!runtime?.active_model;
  const modelEl = document.getElementById('status-model');
  if (modelEl) {
    modelEl.textContent = modelOk ? '✓' : '✗';
    modelEl.className = `status-value ${modelOk ? 'ok' : 'error'}`;
  }

  // Zone status (has zone if current module has zone params)
  const { module } = getMissionSelection();
  const hasZoneParam = (module?.params || []).some((p) => p.type === 'zone');
  const zoneOk = hasZoneParam && (Object.values(paramValues).some((v) => String(v).includes(',')) || false);
  const zoneEl = document.getElementById('status-zone');
  if (zoneEl) {
    zoneEl.textContent = hasZoneParam ? (zoneOk ? '✓' : '−') : '−';
    zoneEl.className = `status-value ${zoneOk ? 'ok' : 'warn'}`;
  }
}

async function refresh() {
  try {
    const [status, runtime] = await Promise.all([
      fetchJson('/api/status'),
      fetchJson('/api/settings')
    ]);
    runtimeCache = runtime;
    runtimeCache.camera_open = !!status.camera_open;
    if (typeof status.plate_watchlist_count === 'number') {
      plateWatchCount = status.plate_watchlist_count;
    }

    updateStatusIndicators(status, runtime);

    document.getElementById('active-model').textContent = runtime.active_model || 'none';
    document.getElementById('camera-source').textContent = runtime.camera_source || 'n/a';
    const internal = document.getElementById('internal-status-note');
    if (internal) {
      const { module } = getMissionSelection();
      const runtimeNote = runtime.status_note || 'nominal';
      const mode = String(runtime.inference_mode || module?.mode || 'auto').toUpperCase();
      internal.textContent = `Runtime: ${runtimeNote} | Active mode: ${mode}`;
    }
    document.getElementById('detection-count').textContent = `${status.detection_count}`;
    updateOperationalCounters(status);

    const signal = document.getElementById('camera-signal');
    if (signal) {
      signal.classList.toggle('online', !!status.camera_open);
    }

    const details = document.getElementById('details');
    if (!status.detections.length) {
      const note = runtime.status_note ? `<div>Status: ${runtime.status_note}</div>` : '';
      details.innerHTML = `${note}<div>No detections in current frame.</div>`;
    } else {
      const classCounts = {};
      for (const det of status.detections) {
        const key = String(det.class_name || 'unknown');
        classCounts[key] = (classCounts[key] || 0) + 1;
      }
      const summary = Object.entries(classCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6)
        .map(([cls, count]) => `<div>Class ${cls}: ${count}</div>`)
        .join('');

      details.innerHTML = status.detections
        .slice(0, 16)
        .map((d) => `<div>• ${d.kind}:${d.class_name} (${d.confidence.toFixed(2)}) box=[${d.box.join(', ')}]</div>`)
        .join('') + summary;

      const event = buildMissionReportEvent(status);
      if (event && event.key !== lastReportKey) {
        appendStatusReport(event.text, Number(event.confidence || 0), { critical: !!event.critical });
        lastReportKey = event.key;
      }
    }

    await syncBrowserMode();
    updateMissionChecklist();

  } catch {
    const cameraNode = document.getElementById('internal-camera-status');
    if (cameraNode) cameraNode.textContent = 'status unavailable';
    const focusNode = document.getElementById('internal-operational-focus');
    if (focusNode) focusNode.textContent = 'status unavailable';
    const signalNode = document.getElementById('internal-mission-signal');
    if (signalNode) signalNode.textContent = 'status unavailable';
    const signal = document.getElementById('camera-signal');
    if (signal) signal.classList.remove('online');
  }
}

async function initLiveMissionUI() {
  await loadMissionCatalog();
  populateMissionNames();
  populateMissionModules();
  renderMissionModule();

  const mission = document.getElementById('mission_name');
  const module = document.getElementById('mission_module');

  if (mission) {
    mission.addEventListener('change', () => {
      populateMissionModules();
      renderMissionModule();
      syncMissionState().catch(() => {});
    });
  }
  if (module) {
    module.addEventListener('change', () => {
      renderMissionModule();
      syncMissionState().catch(() => {});
    });
  }

  const params = document.getElementById('module-params');
  if (params) {
    params.addEventListener('change', () => {
      syncMissionState().catch(() => {});
    });
  }

  await loadMissionState().catch(() => {});
  setMissionPanel();
}

const zoneDraw = { active: false, key: '', shape: 'polygon', points: [] };

function parseZonePoints(value) {
  const nums = String(value || '').split(',').map((s) => parseFloat(s.trim())).filter((n) => !Number.isNaN(n));
  const pts = [];
  for (let i = 0; i + 1 < nums.length; i += 2) pts.push({ x: nums[i], y: nums[i + 1] });
  return pts;
}

function serializeZonePoints(points) {
  return points.map((p) => `${p.x.toFixed(3)},${p.y.toFixed(3)}`).join(',');
}

function syncZoneCanvas() {
  const img = document.getElementById('stream-img');
  const wrap = document.querySelector('.stream-wrap');
  const canvas = document.getElementById('zone-canvas');
  if (!img || !wrap || !canvas) return;
  const wrapRect = wrap.getBoundingClientRect();
  const imgRect = img.getBoundingClientRect();
  canvas.style.left = (imgRect.left - wrapRect.left) + 'px';
  canvas.style.top = (imgRect.top - wrapRect.top) + 'px';
  canvas.style.width = imgRect.width + 'px';
  canvas.style.height = imgRect.height + 'px';
  canvas.width = Math.max(1, Math.round(imgRect.width));
  canvas.height = Math.max(1, Math.round(imgRect.height));
}

function zoneRedraw() {
  const canvas = document.getElementById('zone-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const pts = zoneDraw.points;
  if (!pts.length) return;
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#9ae65a';
  ctx.fillStyle = 'rgba(154, 230, 90, 0.18)';
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = p.x * w;
    const y = p.y * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  if (zoneDraw.shape === 'polygon' && pts.length >= 3) {
    ctx.closePath();
    ctx.fill();
  }
  ctx.stroke();
  ctx.fillStyle = '#d6ff9c';
  pts.forEach((p) => {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function zoneStartDraw(key) {
  const { module } = getMissionSelection();
  const p = (module?.params || []).find((x) => x.key === key);
  if (!p || p.type !== 'zone') return;
  if (zoneDraw.active && zoneDraw.key === key) {
    zoneHideCanvas();
    renderMissionModule();
    return;
  }
  zoneDraw.active = true;
  zoneDraw.key = key;
  zoneDraw.shape = p.shape || 'polygon';
  zoneDraw.points = parseZonePoints(getParamValue(p));
  const canvas = document.getElementById('zone-canvas');
  if (canvas) canvas.hidden = false;
  syncZoneCanvas();
  zoneRedraw();
  renderMissionModule();
}

function zoneCanvasClick(event) {
  if (!zoneDraw.active) return;
  const canvas = document.getElementById('zone-canvas');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  const y = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
  if (zoneDraw.shape === 'line' && zoneDraw.points.length >= 2) {
    zoneDraw.points = [];
  }
  zoneDraw.points.push({ x, y });
  setParamValue(zoneDraw.key, serializeZonePoints(zoneDraw.points));
  zoneRedraw();
  if (zoneDraw.shape === 'line' && zoneDraw.points.length >= 2) {
    zoneHideCanvas();
    renderMissionModule();
    syncMissionState().catch(() => {});
  }
}

function zoneClear(key) {
  setParamValue(key, '');
  if (zoneDraw.active && zoneDraw.key === key) zoneHideCanvas();
  renderMissionModule();
  syncMissionState().catch(() => {});
}

function zoneHideCanvas() {
  zoneDraw.active = false;
  zoneDraw.key = '';
  zoneDraw.points = [];
  const canvas = document.getElementById('zone-canvas');
  if (canvas) { canvas.hidden = true; const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height); }
}

(function bindZoneCanvas() {
  const attach = () => {
    const canvas = document.getElementById('zone-canvas');
    if (canvas) canvas.addEventListener('click', zoneCanvasClick);
    window.addEventListener('resize', () => { if (zoneDraw.active) { syncZoneCanvas(); zoneRedraw(); } });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attach); else attach();
})();

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
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Echelon Vision — Settings</title>
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
      --error: #d98974;
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
      padding-bottom: env(safe-area-inset-bottom);
    }
    main { max-width: 1060px; margin: 0 auto; padding: 24px; display: grid; gap: 14px; }
    .card {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 46px rgba(0,0,0,0.42);
      backdrop-filter: blur(12px);
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand-mark { width: 60px; height: 60px; object-fit: contain; flex: 0 0 auto; filter: drop-shadow(0 8px 20px rgba(0,0,0,0.35)); }
    .brand-copy { display: grid; gap: 3px; }
    .subtitle { margin: 0; color: var(--muted); font-size: 0.78rem; letter-spacing: 0.16em; text-transform: uppercase; }
    h1 { margin: 0; font-size: 1.5rem; letter-spacing: -0.02em; }
    .section-label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .field { display: grid; gap: 6px; }
    label { color: var(--muted); font-size: 0.86rem; }
    input[type="text"], input[type="number"], input[type="url"], select {
      width: 100%; border: 1px solid var(--line); background: rgba(0,0,0,0.2); color: var(--text);
      border-radius: 10px; padding: 10px 12px; min-height: 42px; font-size: 0.9rem; font-family: inherit;
    }
    select option { background: #0e140b; color: var(--text); }
    .hint { font-size: 0.78rem; color: var(--muted); line-height: 1.45; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
    button, a.btn {
      border: 1px solid var(--line-strong); background: rgba(42, 62, 16, 0.55); color: var(--text);
      border-radius: 10px; padding: 10px 14px; min-height: 44px; font-weight: 600; font-size: 0.88rem;
      letter-spacing: 0.03em; text-decoration: none; cursor: pointer; touch-action: manipulation;
    }
    button:hover, a.btn:hover { border-color: var(--accent); transform: translateY(-1px); }
    .param-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .param-card { border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: rgba(0,0,0,0.16); display: grid; gap: 8px; }
    .param-title { font-size: 0.84rem; font-weight: 600; color: var(--text); letter-spacing: 0.02em; }
    .slider-row { display: grid; grid-template-columns: 1fr 80px; gap: 8px; align-items: center; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .toggle-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; font-size: 0.84rem; color: var(--text); }
    input[type="checkbox"] { width: 16px; height: 16px; min-height: 0; accent-color: var(--accent); margin: 0; flex: 0 0 auto; }
    .class-grid {
      max-height: 200px; overflow: auto; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px; border: 1px solid var(--line); border-radius: 10px; padding: 8px; background: rgba(0,0,0,0.14);
    }
    .cls { display: flex; align-items: center; gap: 7px; font-size: 0.84rem; color: var(--text); line-height: 1.2; }
    progress { width: 100%; height: 12px; border-radius: 999px; overflow: hidden; }
    progress::-webkit-progress-bar { background: rgba(0,0,0,0.25); border-radius: 999px; }
    progress::-webkit-progress-value { background: linear-gradient(90deg, var(--accent), var(--ok)); border-radius: 999px; }
    .status-line { font-size: 0.86rem; color: var(--muted); }
    .status-line.ok { color: var(--ok); }
    .status-line.err { color: var(--error); }
    .sticky-bar {
      position: sticky; bottom: 10px; z-index: 2; border: 1px solid var(--line-strong); border-radius: 14px;
      padding: 12px 14px; background: rgba(10, 16, 8, 0.96); box-shadow: 0 10px 24px rgba(0,0,0,0.4);
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    }
    .sticky-bar .actions { margin: 0; }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
      .param-grid { grid-template-columns: 1fr; }
      main { padding: 14px; }
      .class-grid { grid-template-columns: 1fr; }
    }
    @media (hover: none) and (pointer: coarse) { button:hover, a.btn:hover { transform: none; } }
  </style>
</head>
<body>
<main>
  <!-- Header -->
  <section class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
      <div class="brand">
        <img class="brand-mark" src="/brand.svg" alt="Echelon Vision insignia" />
        <div class="brand-copy">
          <p class="subtitle">System Configuration</p>
          <h1>Echelon Vision Settings</h1>
        </div>
      </div>
      <a class="btn" href="/">&#8592; Back to Live</a>
    </div>
  </section>

  <!-- Camera & Model -->
  <section class="card">
    <div class="section-label">Camera &amp; Model</div>
    <div class="grid">
      <div class="field">
        <label>Camera Source</label>
        <select id="camera_source_preset">
          <option value="uvc://0">USB Camera 0</option>
          <option value="uvc://1">USB Camera 1</option>
          <option value="browser://webui">Browser / Phone Camera</option>
          <option value="rtsp://">RTSP Stream</option>
          <option value="rtmp://">RTMP Stream</option>
          <option value="">Custom URI (manual)</option>
        </select>
        <input id="camera_source" type="text" placeholder="Examples: rtsp://user:pass@ip:554/live | rtmp://ip:1935/live | uvc://0 | browser://webui" />
        <div class="hint">Supported formats: RTSP (rtsp:// or rtsps://), RTMP (rtmp:// or rtmps://), USB cameras (uvc://0-9), browser camera. For credentials, use: protocol://user:pass@host:port/path</div>
      </div>
      <div class="field">
        <label>Active Model</label>
        <select id="model_select"></select>
        <div class="hint">Models loaded from <code>/models</code>. Mission module controls the inference task automatically.</div>
      </div>
    </div>
  </section>

  <!-- Engine Defaults -->
  <section class="card">
    <div class="section-label">Engine Defaults</div>
    <div class="hint" style="margin-bottom:12px;">Global inference defaults. Mission module parameters override these per-session.</div>
    <div class="param-grid">
      <div class="param-card">
        <div class="param-title">Confidence Threshold</div>
        <div class="slider-row">
          <input id="confidence_slider" type="range" step="0.01" min="0.01" max="1" />
          <input id="confidence" type="number" step="0.01" min="0.01" max="1" />
        </div>
      </div>
      <div class="param-card">
        <div class="param-title">IoU (NMS)</div>
        <div class="slider-row">
          <input id="iou_slider" type="range" step="0.01" min="0.01" max="1" />
          <input id="iou" type="number" step="0.01" min="0.01" max="1" />
        </div>
      </div>
      <div class="param-card">
        <div class="param-title">Image Size</div>
        <div class="slider-row">
          <input id="image_size_slider" type="range" min="128" max="2048" step="32" />
          <input id="image_size" type="number" min="128" max="2048" step="32" />
        </div>
      </div>
      <div class="param-card">
        <div class="param-title">BBox Opacity</div>
        <div class="slider-row">
          <input id="bbox_opacity_slider" type="range" step="0.05" min="0" max="1" />
          <input id="bbox_opacity" type="number" step="0.05" min="0" max="1" />
        </div>
      </div>
      <div class="param-card">
        <div class="param-title">Mask Opacity</div>
        <div class="slider-row">
          <input id="mask_opacity_slider" type="range" step="0.05" min="0" max="1" />
          <input id="mask_opacity" type="number" step="0.05" min="0" max="1" />
        </div>
      </div>
    </div>
  </section>

  <!-- Tracking & SAHI -->
  <section class="card">
    <div class="section-label">Tracking &amp; SAHI</div>
    <div class="param-grid">
      <div class="param-card">
        <div class="param-title">Object Tracking</div>
        <div class="toggle-row"><span>Tracking Enabled</span><input id="tracking_enabled" type="checkbox" /></div>
        <div class="toggle-row"><span>Persist IDs across frames</span><input id="tracking_persist" type="checkbox" /></div>
        <div class="toggle-row"><span>Show Track IDs on stream</span><input id="tracking_show_ids" type="checkbox" /></div>
      </div>
      <div class="param-card">
        <div class="param-title">SAHI Sliced Inference</div>
        <div class="toggle-row"><span>SAHI Enabled</span><input id="sahi_enabled" type="checkbox" /></div>
        <div class="slider-row">
          <input id="sahi_slice_height_slider" type="range" min="64" max="2048" step="16" />
          <input id="sahi_slice_height" type="number" min="64" max="2048" step="16" />
        </div>
        <div class="slider-row">
          <input id="sahi_slice_width_slider" type="range" min="64" max="2048" step="16" />
          <input id="sahi_slice_width" type="number" min="64" max="2048" step="16" />
        </div>
        <div class="slider-row">
          <input id="sahi_overlap_height_ratio_slider" type="range" step="0.05" min="0" max="0.9" />
          <input id="sahi_overlap_height_ratio" type="number" step="0.05" min="0" max="0.9" />
        </div>
        <div class="slider-row">
          <input id="sahi_overlap_width_ratio_slider" type="range" step="0.05" min="0" max="0.9" />
          <input id="sahi_overlap_width_ratio" type="number" step="0.05" min="0" max="0.9" />
        </div>
      </div>
    </div>
  </section>

  <!-- Class Filters -->
  <section class="card">
    <div class="section-label">Enabled Detection Classes</div>
    <div class="hint">Filter which object classes the engine reports globally. Per-module class targets are set in the mission panel.</div>
    <div class="actions">
      <button type="button" onclick="setAllClassFilters(true)">Select All</button>
      <button type="button" onclick="setAllClassFilters(false)">Clear All</button>
    </div>
    <div id="class-grid" class="class-grid" style="margin-top:10px;"></div>
  </section>

  <!-- Model Library -->
  <section class="card">
    <div class="section-label">Model Library</div>
    <div class="grid">
      <div class="field">
        <label>Model to Download</label>
        <select id="official_model_select"></select>
        <div class="hint">Download official Ultralytics models into <code>/models</code>.</div>
      </div>
      <div class="field">
        <label>Description</label>
        <div id="official_model_description" class="hint">Select a model to view details.</div>
      </div>
    </div>
    <div class="actions" style="margin-top:10px;">
      <button type="button" onclick="downloadSelectedModel()">Download &amp; Use</button>
      <button type="button" onclick="refreshModelLibrary()">Refresh Library</button>
    </div>
    <progress id="model_download_progress" value="0" max="100" style="margin-top:10px;"></progress>
    <div id="model_download_status" class="status-line" style="margin-top:6px;">No model download in progress.</div>
  </section>

  <!-- Sticky save bar -->
  <div class="sticky-bar">
    <div class="actions">
      <button onclick="saveAll()">Save Settings</button>
      <button onclick="reloadRuntime()">Reload</button>
      <a class="btn" href="/">&#8592; Back to Live</a>
    </div>
    <div id="status" class="status-line" style="flex:1;">Ready.</div>
  </div>
</main>

<script>
const keys = [
  'camera_source', 'confidence', 'iou', 'image_size', 'bbox_opacity', 'mask_opacity',
  'tracking_enabled', 'tracking_persist', 'tracking_show_ids',
  'sahi_enabled', 'sahi_slice_height', 'sahi_slice_width',
  'sahi_overlap_height_ratio', 'sahi_overlap_width_ratio'
];
let officialModelCatalog = [];
let modelDownloadPoll = null;

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function setStatus(text, ok = true) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = 'status-line ' + (ok ? 'ok' : 'err');
}

function asNumber(id) { return Number(document.getElementById(id)?.value || 0); }
function asBool(id) {
  const node = document.getElementById(id);
  if (!node) return false;
  return node.type === 'checkbox' ? !!node.checked : node.value === 'true';
}

const sliderPairs = [
  ['confidence_slider', 'confidence'],
  ['iou_slider', 'iou'],
  ['image_size_slider', 'image_size'],
  ['bbox_opacity_slider', 'bbox_opacity'],
  ['mask_opacity_slider', 'mask_opacity'],
  ['sahi_slice_height_slider', 'sahi_slice_height'],
  ['sahi_slice_width_slider', 'sahi_slice_width'],
  ['sahi_overlap_height_ratio_slider', 'sahi_overlap_height_ratio'],
  ['sahi_overlap_width_ratio_slider', 'sahi_overlap_width_ratio'],
];

function bindSliderPairs() {
  for (const [sliderId, inputId] of sliderPairs) {
    const slider = document.getElementById(sliderId);
    const input = document.getElementById(inputId);
    if (!slider || !input) continue;
    slider.addEventListener('input', () => { input.value = slider.value; });
    input.addEventListener('input', () => { slider.value = input.value; });
  }
}

function syncSlidersFromInputs() {
  for (const [sliderId, inputId] of sliderPairs) {
    const slider = document.getElementById(sliderId);
    const input = document.getElementById(inputId);
    if (!slider || !input) continue;
    slider.value = input.value;
  }
}

function syncSahiFields() {
  const enabled = asBool('sahi_enabled');
  const ids = ['sahi_slice_height', 'sahi_slice_width', 'sahi_overlap_height_ratio', 'sahi_overlap_width_ratio'];
  for (const id of ids) {
    const node = document.getElementById(id);
    const slider = document.getElementById(id + '_slider');
    if (node) { node.disabled = !enabled; node.style.opacity = enabled ? '1' : '0.5'; }
    if (slider) { slider.disabled = !enabled; slider.style.opacity = enabled ? '1' : '0.5'; }
  }
}

function syncCameraSourcePreset() {
  const input = document.getElementById('camera_source');
  const preset = document.getElementById('camera_source_preset');
  if (!input || !preset) return;
  const value = String(input.value || '');
  const options = Array.from(preset.options).map((o) => o.value).filter(Boolean);
  preset.value = options.includes(value) ? value : '';
}

function renderClassGrid(availableClasses, enabledClasses) {
  const grid = document.getElementById('class-grid');
  if (!grid) return;
  if (!availableClasses || !availableClasses.length) {
    grid.innerHTML = '<div class="hint">No class list available — load a model first.</div>';
    return;
  }
  const enabled = new Set((enabledClasses || []).map((n) => String(n)));
  grid.innerHTML = availableClasses.map((name) => {
    const checked = enabled.size === 0 || enabled.has(name) ? 'checked' : '';
    const safe = String(name).replace(/"/g, '&quot;');
    return `<label class="cls"><input type="checkbox" class="class-filter" value="${safe}" ${checked}/> ${safe}</label>`;
  }).join('');
}

function getSelectedClassFilters() {
  return Array.from(document.querySelectorAll('.class-filter')).filter((n) => n.checked).map((n) => n.value);
}

function setAllClassFilters(checked) {
  for (const node of document.querySelectorAll('.class-filter')) node.checked = checked;
}

function renderOfficialCatalog() {
  const select = document.getElementById('official_model_select');
  if (!select) return;
  select.innerHTML = officialModelCatalog.map((model) => {
    const dl = model.downloaded ? ' [downloaded]' : '';
    return `<option value="${model.name}">${model.name} | ${model.family} | ${model.task}${dl}</option>`;
  }).join('');
  updateOfficialModelDescription();
}

function updateOfficialModelDescription() {
  const select = document.getElementById('official_model_select');
  const target = document.getElementById('official_model_description');
  if (!select || !target) return;
  const model = officialModelCatalog.find((e) => e.name === select.value);
  if (!model) { target.textContent = 'Select a model to view details.'; return; }
  const local = model.downloaded ? 'Already in /models.' : 'Will download into /models.';
  target.textContent = `${model.family} ${model.task}: ${model.description} ${local}`;
}

function updateDownloadState(state) {
  const bar = document.getElementById('model_download_progress');
  const status = document.getElementById('model_download_status');
  if (bar) bar.value = Number(state.progress || 0);
  if (status) status.textContent = state.message || 'No model download in progress.';
  if (!state.active && modelDownloadPoll) {
    clearInterval(modelDownloadPoll);
    modelDownloadPoll = null;
  }
}

async function pollModelDownload() {
  const state = await fetchJson('/api/model-download');
  updateDownloadState(state);
  if (!state.active) {
    await refreshModelLibrary();
    await reloadRuntime();
  }
}

async function downloadSelectedModel() {
  const select = document.getElementById('official_model_select');
  if (!select || !select.value) { setStatus('Select a model to download.', false); return; }
  try {
    await fetchJson('/api/model-download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: select.value, use_after_download: true }),
    });
    setStatus('Downloading ' + select.value + '...', true);
    if (modelDownloadPoll) clearInterval(modelDownloadPoll);
    modelDownloadPoll = setInterval(() => {
      pollModelDownload().catch((err) => setStatus('Poll failed: ' + err.message, false));
    }, 1000);
    await pollModelDownload();
  } catch (err) {
    setStatus('Download failed: ' + err.message, false);
  }
}

async function refreshModelLibrary() {
  const [catalogData, downloadData, modelData] = await Promise.all([
    fetchJson('/api/model-catalog'),
    fetchJson('/api/model-download'),
    fetchJson('/api/models'),
  ]);
  officialModelCatalog = catalogData.models || [];
  renderOfficialCatalog();
  updateDownloadState(downloadData);
  const select = document.getElementById('model_select');
  select.innerHTML = (modelData.models || []).map((m) => {
    const disabled = m.selectable ? '' : 'disabled';
    const note = m.note ? ` (${m.note})` : '';
    return `<option value="${m.name}" ${disabled}>${m.name} [${m.format}]${note}</option>`;
  }).join('');
}

async function reloadRuntime() {
  const [runtime, modelData, catalogData, downloadData] = await Promise.all([
    fetchJson('/api/settings'),
    fetchJson('/api/models'),
    fetchJson('/api/model-catalog'),
    fetchJson('/api/model-download'),
  ]);
  for (const key of keys) {
    const node = document.getElementById(key);
    if (!node) continue;
    const value = runtime[key];
    if (value === undefined || value === null) continue;
    if (node.type === 'checkbox') node.checked = !!value;
    else node.value = String(value);
  }
  const select = document.getElementById('model_select');
  select.innerHTML = (modelData.models || []).map((m) => {
    const disabled = m.selectable ? '' : 'disabled';
    const note = m.note ? ` (${m.note})` : '';
    return `<option value="${m.name}" ${disabled}>${m.name} [${m.format}]${note}</option>`;
  }).join('');
  if (runtime.active_model) select.value = runtime.active_model;
  renderClassGrid(runtime.available_classes || [], runtime.enabled_classes || []);
  syncSlidersFromInputs();
  syncSahiFields();
  officialModelCatalog = catalogData.models || [];
  renderOfficialCatalog();
  updateDownloadState(downloadData);
  syncCameraSourcePreset();
  setStatus('Settings loaded.');
}

async function saveAll() {
  try {
    const camSourceInput = document.getElementById('camera_source').value.trim();
    
    // Validate and normalize camera source URL
    let camSource = camSourceInput;
    if (camSource) {
      // Support RTSP, RTMP, and UVC formats
      if (camSource.startsWith('rtsp://') || camSource.startsWith('rtmp://') || camSource.startsWith('uvc://') || camSource.startsWith('browser://')) {
        // URLs are valid - allow them to pass through
        // Note: Special characters like @ and : are allowed in URLs
        camSource = camSource;
      } else if (camSource.match(/^\d+$/) || camSource === 'internal') {
        // Numeric camera index - convert to uvc:// format
        camSource = `uvc://${camSource}`;
      } else if (!camSource.includes('://')) {
        // No protocol specified - assume it's a local file or index
        const asNum = parseInt(camSource);
        if (!isNaN(asNum)) {
          camSource = `uvc://${asNum}`;
        }
      }
    }
    
    const payload = {
      camera_source: camSource,
      confidence: asNumber('confidence'),
      iou: asNumber('iou'),
      image_size: asNumber('image_size'),
      bbox_opacity: asNumber('bbox_opacity'),
      mask_opacity: asNumber('mask_opacity'),
      tracking_enabled: asBool('tracking_enabled'),
      tracking_persist: asBool('tracking_persist'),
      tracking_show_ids: asBool('tracking_show_ids'),
      sahi_enabled: asBool('sahi_enabled'),
      sahi_slice_height: asNumber('sahi_slice_height'),
      sahi_slice_width: asNumber('sahi_slice_width'),
      sahi_overlap_height_ratio: asNumber('sahi_overlap_height_ratio'),
      sahi_overlap_width_ratio: asNumber('sahi_overlap_width_ratio'),
    };
    
    const res = await fetchJson('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    
    // Verify camera source was saved
    if (res?.runtime?.camera_source) {
      const savedCamSource = res.runtime.camera_source;
      setStatus(`Settings saved. Camera source: ${savedCamSource}`);
    } else {
      setStatus('Settings saved.');
    }
    
    const selectedModel = document.getElementById('model_select').value;
    if (selectedModel) {
      await fetchJson('/api/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel }),
      });
    }
    await fetchJson('/api/classes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled_classes: getSelectedClassFilters() }),
    });
  } catch (err) {
    setStatus('Save failed: ' + (err.message || String(err)), false);
    console.error('Settings save error:', err);
  }
}

reloadRuntime();
bindSliderPairs();
document.getElementById('camera_source_preset')?.addEventListener('change', (e) => {
  if (e.target.value) {
    const input = document.getElementById('camera_source');
    if (input) input.value = e.target.value;
    syncCameraSourcePreset();
  }
});
document.getElementById('camera_source')?.addEventListener('input', syncCameraSourcePreset);
document.getElementById('sahi_enabled')?.addEventListener('change', syncSahiFields);
document.getElementById('official_model_select')?.addEventListener('change', updateOfficialModelDescription);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
