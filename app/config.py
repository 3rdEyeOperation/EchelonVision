from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _to_int(os.getenv("PORT"), 3000)
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))
    models_dir: Path = Path(os.getenv("MODELS_DIR", "/models"))
    camera_source: str = os.getenv("CAMERA_SOURCE", f"uvc://{os.getenv('CAMERA_INDEX', '0')}")
    camera_width: int = _to_int(os.getenv("CAMERA_WIDTH"), 1280)
    camera_height: int = _to_int(os.getenv("CAMERA_HEIGHT"), 720)
    camera_fps: int = _to_int(os.getenv("CAMERA_FPS"), 15)
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    yolo_confidence: float = _to_float(os.getenv("YOLO_CONFIDENCE"), 0.35)
    yolo_iou: float = _to_float(os.getenv("YOLO_IOU"), 0.50)
    yolo_image_size: int = _to_int(os.getenv("YOLO_IMAGE_SIZE"), 640)
    bbox_opacity: float = _to_float(os.getenv("BBOX_OPACITY"), 0.35)
    sahi_enabled: bool = _to_bool(os.getenv("SAHI_ENABLED"), False)
    sahi_slice_height: int = _to_int(os.getenv("SAHI_SLICE_HEIGHT"), 512)
    sahi_slice_width: int = _to_int(os.getenv("SAHI_SLICE_WIDTH"), 512)
    sahi_overlap_height_ratio: float = _to_float(os.getenv("SAHI_OVERLAP_HEIGHT"), 0.2)
    sahi_overlap_width_ratio: float = _to_float(os.getenv("SAHI_OVERLAP_WIDTH"), 0.2)
    faces_dir: Path = Path(os.getenv("FACES_DIR", "/data/faces"))
    face_match_threshold: float = _to_float(os.getenv("FACE_MATCH_THRESHOLD"), 0.35)
    face_model_name: str = os.getenv("FACE_MODEL_NAME", "buffalo_l")
    face_recognition_enabled: bool = _to_bool(os.getenv("FACE_RECOGNITION_ENABLED"), False)
    alpr_enabled: bool = _to_bool(os.getenv("ALPR_ENABLED"), False)


settings = Settings()
