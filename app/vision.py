from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from app.config import Settings


@dataclass
class Detection:
    label: str
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]
    kind: str


@dataclass
class RuntimeOptions:
    camera_source: str
    confidence: float
    iou: float
    image_size: int
    bbox_opacity: float
    enabled_classes: set[str] = field(default_factory=set)
    sahi_enabled: bool = False
    sahi_slice_height: int = 512
    sahi_slice_width: int = 512
    sahi_overlap_height_ratio: float = 0.2
    sahi_overlap_width_ratio: float = 0.2
    alpr_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["enabled_classes"] = sorted(self.enabled_classes)
        return data


@dataclass
class ModelCatalogEntry:
    name: str
    path: str
    format: str
    classes: list[str]
    selectable: bool
    note: str | None = None


class CameraError(RuntimeError):
    pass


class FaceAnalyzer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = False
        self.error: str | None = None
        self._model = None
        self.known_embeddings: list[tuple[str, np.ndarray]] = []

        if not settings.face_recognition_enabled:
            self.error = "Face recognition disabled by FACE_RECOGNITION_ENABLED"
            return

        try:
            from insightface.app import FaceAnalysis

            self._model = FaceAnalysis(
                name=settings.face_model_name,
                providers=["CPUExecutionProvider"],
            )
            self._model.prepare(ctx_id=0, det_size=(640, 640))
            self.enabled = True
            self.known_embeddings = self._load_gallery(settings.faces_dir)
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            self.error = str(exc)

    def _load_gallery(self, faces_dir: Path) -> list[tuple[str, np.ndarray]]:
        if not faces_dir.exists():
            return []

        gallery: list[tuple[str, np.ndarray]] = []
        for person_dir in sorted(faces_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            for image_path in sorted(person_dir.glob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    continue
                image = cv2.imread(str(image_path))
                if image is None or self._model is None:
                    continue
                faces = self._model.get(image)
                if not faces:
                    continue
                best_face = max(faces, key=lambda face: float(face.bbox[2] - face.bbox[0]))
                embedding = np.asarray(best_face.embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    continue
                gallery.append((person_name, embedding / norm))
                break
        return gallery

    def recognize(self, image: np.ndarray) -> list[Detection]:
        if not self.enabled or self._model is None:
            return []

        faces = self._model.get(image)
        detections: list[Detection] = []

        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.int32)
            x1, y1, x2, y2 = bbox.tolist()
            embedding = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            normalized = embedding / norm
            name, score = self._match(normalized)
            label = name if name is not None else "Unknown"
            confidence = score if score is not None else 0.0
            detections.append(
                Detection(
                    label=label,
                    class_name="face",
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                    kind="face",
                )
            )
        return detections

    def _match(self, embedding: np.ndarray) -> tuple[str | None, float | None]:
        if not self.known_embeddings:
            return None, None

        best_name: str | None = None
        best_score = -1.0
        for name, known_embedding in self.known_embeddings:
            score = float(np.dot(embedding, known_embedding))
            if score > best_score:
                best_name = name
                best_score = score

        if best_name is None or best_score < self.settings.face_match_threshold:
            return None, best_score
        return best_name, best_score


class VisionEngine:
    SUPPORTED_SUFFIXES = {".pt", ".onnx", ".rknn"}
    COCO80_CLASSES = (
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.face_analyzer = FaceAnalyzer(settings)

        self._lock = threading.Lock()
        self._model = None
        self._active_model_name = ""
        self._active_model_path = ""
        self._active_model_format = ""
        self._labels: dict[int, str] = {}
        self._sahi_model = None
        self._sahi_available: bool | None = None
        self._rknn_runtime_available: bool | None = None
        self._rknn_model = None
        self._status_note: str | None = None
        self._last_bbox_trace: dict[str, Any] = {}

        self.runtime = RuntimeOptions(
            camera_source=settings.camera_source,
            confidence=settings.yolo_confidence,
            iou=settings.yolo_iou,
            image_size=settings.yolo_image_size,
            bbox_opacity=max(0.0, min(settings.bbox_opacity, 1.0)),
            enabled_classes=set(),
            sahi_enabled=settings.sahi_enabled,
            sahi_slice_height=settings.sahi_slice_height,
            sahi_slice_width=settings.sahi_slice_width,
            sahi_overlap_height_ratio=settings.sahi_overlap_height_ratio,
            sahi_overlap_width_ratio=settings.sahi_overlap_width_ratio,
            alpr_enabled=settings.alpr_enabled,
        )

        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap_default_model()

    def _bootstrap_default_model(self) -> None:
        preferred = self.settings.yolo_model
        default_model_hint = self.settings.models_dir / ".default_model"
        if (not preferred or preferred == "yolov8n.pt") and default_model_hint.exists():
            try:
                hinted = default_model_hint.read_text(encoding="utf-8").strip()
                if hinted:
                    preferred = hinted
            except Exception:
                pass
        model_candidates = self.list_models()

        if preferred:
            ok, message = self.select_model(preferred)
            if ok:
                return
            self._status_note = message

        if model_candidates:
            ok, message = self.select_model(model_candidates[0].name)
            if not ok:
                self._status_note = message

    def list_models(self) -> list[ModelCatalogEntry]:
        entries: list[ModelCatalogEntry] = []
        models_dir = self.settings.models_dir

        if models_dir.exists():
            for model_path in sorted(models_dir.rglob("*")):
                if not model_path.is_file() or model_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                    continue
                classes = self._load_sidecar_classes(model_path)
                model_format = model_path.suffix.lower().lstrip(".")
                selectable, note = self._is_selectable(model_format)
                entries.append(
                    ModelCatalogEntry(
                        name=model_path.name,
                        path=str(model_path),
                        format=model_format,
                        classes=classes,
                        selectable=selectable,
                        note=note,
                    )
                )

        return entries

    def _is_selectable(self, model_format: str) -> tuple[bool, str | None]:
        if model_format in {"pt", "onnx"}:
            return True, None
        if model_format == "rknn":
            if self._has_rknn_runtime():
                return True, None
            return False, "RKNN runtime unavailable. Install rknn-toolkit-lite2 on Banana Pi host."
        return False, f"Unsupported model format: {model_format}"

    def _has_rknn_runtime(self) -> bool:
        if self._rknn_runtime_available is not None:
            return self._rknn_runtime_available
        try:
            from rknnlite.api import RKNNLite  # noqa: F401

            self._rknn_runtime_available = True
        except Exception:
            self._rknn_runtime_available = False
        return self._rknn_runtime_available

    def _load_sidecar_classes(self, model_path: Path) -> list[str]:
        sidecar_candidates = [
            model_path.with_suffix(model_path.suffix + ".classes.json"),
            model_path.with_suffix(".classes.json"),
            model_path.with_suffix(".json"),
            model_path.with_suffix(".labels.txt"),
        ]

        for sidecar in sidecar_candidates:
            if not sidecar.exists():
                continue
            if sidecar.suffix == ".txt":
                labels = [line.strip() for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
                if labels:
                    return labels
                continue

            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                continue

            if isinstance(payload, list):
                return [str(item) for item in payload]

            if isinstance(payload, dict):
                names = payload.get("names") or payload.get("classes") or payload.get("labels")
                if isinstance(names, dict):
                    ordered = [str(names[key]) for key in sorted(names.keys(), key=lambda key: int(key) if str(key).isdigit() else str(key))]
                    return ordered
                if isinstance(names, list):
                    return [str(item) for item in names]

        return []

    def select_model(self, model_ref: str) -> tuple[bool, str]:
        target_path = self._resolve_model_path(model_ref)
        model_input = str(target_path) if target_path is not None else model_ref

        if target_path is not None:
            suffix = target_path.suffix.lower().lstrip(".")
            selectable, note = self._is_selectable(suffix)
            if not selectable:
                return False, note or "Model is not selectable"
        else:
            suffix = Path(model_ref).suffix.lower().lstrip(".") or "pt"

        if suffix == "rknn":
            if target_path is None:
                return False, "RKNN model path not found"
            classes = self._load_sidecar_classes(target_path)
            return self._load_rknn_model(target_path, classes)

        try:
            from ultralytics import YOLO

            loaded = YOLO(model_input)
            names = getattr(loaded, "names", {})
            labels: dict[int, str] = {}
            if isinstance(names, dict):
                labels = {int(key): str(value) for key, value in names.items()}
            elif isinstance(names, list):
                labels = {idx: str(name) for idx, name in enumerate(names)}

            with self._lock:
                self._model = loaded
                self._active_model_name = target_path.name if target_path is not None else model_ref
                self._active_model_path = str(target_path) if target_path is not None else model_ref
                self._active_model_format = suffix
                self._labels = labels
                self._rknn_model = None
                self._sahi_model = None
                self._status_note = None
                self.runtime.enabled_classes = set(labels.values())
            return True, f"Loaded model: {self._active_model_name}"
        except Exception as exc:
            return False, f"Failed to load model {model_ref}: {exc}"

    def _load_rknn_model(self, model_path: Path, classes: list[str]) -> tuple[bool, str]:
        if not self._has_rknn_runtime():
            return False, "RKNN runtime unavailable. Install rknn-toolkit-lite2 on Banana Pi host."
        try:
            from rknnlite.api import RKNNLite

            runtime = RKNNLite()
            status = runtime.load_rknn(str(model_path))
            if status != 0:
                return False, f"RKNN load failed with code {status}"

            status = runtime.init_runtime()
            if status != 0:
                return False, f"RKNN init_runtime failed with code {status}"

            labels = {idx: name for idx, name in enumerate(classes)}
            with self._lock:
                self._model = None
                self._rknn_model = runtime
                self._active_model_name = model_path.name
                self._active_model_path = str(model_path)
                self._active_model_format = "rknn"
                self._labels = labels
                self._sahi_model = None
                self._status_note = "RKNN model loaded on NPU runtime"
                self.runtime.enabled_classes = set(classes)
            return True, f"Loaded RKNN model: {model_path.name}"
        except Exception as exc:
            return False, f"Failed to load RKNN model {model_path.name}: {exc}"

    def _resolve_model_path(self, model_ref: str) -> Path | None:
        path = Path(model_ref)
        if path.exists() and path.is_file():
            return path

        candidate = self.settings.models_dir / model_ref
        if candidate.exists() and candidate.is_file():
            return candidate

        matches = [entry for entry in self.list_models() if entry.name == model_ref]
        if matches:
            return Path(matches[0].path)

        if path.suffix in self.SUPPORTED_SUFFIXES:
            return None

        return None

    def set_enabled_classes(self, class_names: list[str]) -> None:
        with self._lock:
            requested = {str(name) for name in class_names if str(name).strip()}
            self.runtime.enabled_classes = requested or set(self._labels.values())

    def update_runtime(self, **updates: Any) -> None:
        with self._lock:
            for key, value in updates.items():
                if not hasattr(self.runtime, key) or value is None:
                    continue
                if key == "bbox_opacity":
                    value = max(0.0, min(float(value), 1.0))
                setattr(self.runtime, key, value)

    def get_runtime_snapshot(self) -> dict[str, Any]:
        with self._lock:
            runtime = self.runtime.as_dict()
            runtime.update(
                {
                    "active_model": self._active_model_name,
                    "active_model_path": self._active_model_path,
                    "active_model_format": self._active_model_format,
                    "available_classes": [self._labels[idx] for idx in sorted(self._labels.keys())],
                    "status_note": self._status_note,
                    "rknn_runtime_available": self._has_rknn_runtime(),
                    "known_faces": len(self.face_analyzer.known_embeddings),
                    "face_recognition_ready": self.face_analyzer.enabled,
                    "face_recognition_error": self.face_analyzer.error,
                }
            )
            return runtime

    def annotate(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        with self._lock:
            model = self._model
            rknn_model = self._rknn_model
            labels = dict(self._labels)
            active_format = self._active_model_format
            runtime = RuntimeOptions(**self.runtime.as_dict())
            runtime.enabled_classes = set(runtime.enabled_classes)

        annotated = frame.copy()
        detections: list[Detection] = []
        bbox_trace: dict[str, Any] = {
            "model_format": active_format,
            "model": self._active_model_name,
            "conf": runtime.confidence,
            "iou": runtime.iou,
            "image_size": runtime.image_size,
            "enabled_classes_count": len(runtime.enabled_classes),
            "labels_count": len(labels),
        }

        if active_format == "rknn" and rknn_model is not None:
            obj_detections, rknn_trace = self._predict_rknn(rknn_model, frame, labels, runtime)
            bbox_trace.update(rknn_trace)
            for detection in obj_detections:
                if runtime.enabled_classes and len(runtime.enabled_classes) < len(labels) and detection.class_name not in runtime.enabled_classes:
                    continue
                detections.append(detection)
                self._draw_box(
                    annotated,
                    detection.box,
                    f"object:{detection.class_name}",
                    detection.confidence,
                    self._color_for_label(detection.class_name),
                    runtime.bbox_opacity,
                )
        elif model is not None:
            if runtime.sahi_enabled and self._can_use_sahi():
                obj_detections = self._predict_with_sahi(frame, runtime)
                bbox_trace["decode_strategy"] = "sahi"
            else:
                obj_detections = self._predict_direct(model, frame, labels, runtime)
                bbox_trace["decode_strategy"] = "direct"
            bbox_trace["raw_detections"] = len(obj_detections)

            for detection in obj_detections:
                if runtime.enabled_classes and len(runtime.enabled_classes) < len(labels) and detection.class_name not in runtime.enabled_classes:
                    continue
                detections.append(detection)
                self._draw_box(
                    annotated,
                    detection.box,
                    f"object:{detection.class_name}",
                    detection.confidence,
                    self._color_for_label(detection.class_name),
                    runtime.bbox_opacity,
                )

        face_detections = self.face_analyzer.recognize(frame)
        for face in face_detections:
            detections.append(face)
            self._draw_box(
                annotated,
                face.box,
                f"face:{face.label}",
                face.confidence,
                (52, 152, 219),
                runtime.bbox_opacity,
            )

        bbox_trace["total_detections_drawn"] = len(detections)
        self._last_bbox_trace = bbox_trace
        return annotated, detections

    def get_bbox_trace(self) -> dict[str, Any]:
        return dict(self._last_bbox_trace)

    def _predict_direct(
        self,
        model: Any,
        frame: np.ndarray,
        labels: dict[int, str],
        runtime: RuntimeOptions,
    ) -> list[Detection]:
        detections: list[Detection] = []

        results = model.predict(
            source=frame,
            conf=runtime.confidence,
            iou=runtime.iou,
            imgsz=runtime.image_size,
            verbose=False,
        )
        if not results:
            return detections

        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = self._resolve_class_name(class_id, labels)
            detections.append(
                Detection(
                    label=class_name,
                    class_name=class_name,
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                    kind="object",
                )
            )

        return detections

    def _predict_rknn(
        self,
        runtime_model: Any,
        frame: np.ndarray,
        labels: dict[int, str],
        runtime: RuntimeOptions,
    ) -> tuple[list[Detection], dict[str, Any]]:
        detections: list[Detection] = []
        trace: dict[str, Any] = {
            "decode_strategy": "rknn",
            "variants_tried": 0,
            "variant_results": [],
            "raw_detections": 0,
            "remapped_detections": 0,
            "rknn_error": None,
        }
        try:
            image_size = max(128, int(runtime.image_size))
            best_candidate: tuple[list[tuple[int, int, int, int]], list[float], list[int], float, float, float] | None = None
            best_candidate_count = 0

            for variant_idx, (input_tensor, ratio, dw, dh) in enumerate(self._rknn_input_variants(frame, image_size)):
                trace["variants_tried"] += 1
                variant_info: dict[str, Any] = {"variant": variant_idx, "input_shape": list(input_tensor.shape)}
                outputs = runtime_model.inference(inputs=[input_tensor])
                boxes, scores, class_ids = self._decode_rknn_outputs(
                    outputs or [],
                    runtime.confidence,
                    num_classes_hint=len(labels),
                    input_size=image_size,
                )
                variant_info["decoded_boxes"] = int(boxes.shape[0])
                variant_info["output_shapes"] = [list(np.asarray(o).shape) for o in (outputs or [])]
                if boxes.size == 0:
                    trace["variant_results"].append(variant_info)
                    continue

                keep_indices = self._nms_indices_numpy(boxes, scores, runtime.iou)
                variant_info["after_nms"] = int(keep_indices.size)
                if keep_indices.size == 0:
                    trace["variant_results"].append(variant_info)
                    continue

                candidate_boxes: list[tuple[int, int, int, int]] = []
                candidate_scores: list[float] = []
                candidate_class_ids: list[int] = []

                for idx in keep_indices.tolist():
                    remapped = self._remap_rknn_box_to_frame(
                        boxes[idx],
                        ratio=ratio,
                        dw=dw,
                        dh=dh,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )
                    if remapped is None:
                        continue

                    x1, y1, x2, y2 = remapped

                    candidate_boxes.append((x1, y1, x2, y2))
                    candidate_scores.append(float(scores[idx]))
                    candidate_class_ids.append(int(class_ids[idx]))

                variant_info["remapped_boxes"] = len(candidate_boxes)
                trace["variant_results"].append(variant_info)
                if len(candidate_boxes) > best_candidate_count:
                    best_candidate_count = len(candidate_boxes)
                    best_candidate = (candidate_boxes, candidate_scores, candidate_class_ids, ratio, dw, dh)

            if best_candidate is None:
                trace["raw_detections"] = 0
                return detections, trace

            boxes_list, scores_list, class_ids_list, ratio, dw, dh = best_candidate
            trace["raw_detections"] = len(boxes_list)
            trace["remapped_detections"] = len(boxes_list)

            for (x1, y1, x2, y2), score, class_id_i in zip(boxes_list, scores_list, class_ids_list):
                class_name = self._resolve_class_name(class_id_i, labels)
                detections.append(
                    Detection(
                        label=class_name,
                        class_name=class_name,
                        confidence=float(score),
                        box=(x1, y1, x2, y2),
                        kind="object",
                    )
                )
        except Exception as exc:
            self._status_note = f"RKNN inference failed: {exc}"
            trace["rknn_error"] = str(exc)
        return detections, trace

    @staticmethod
    def _remap_rknn_box_to_frame(
        raw_box: np.ndarray,
        ratio: float,
        dw: float,
        dh: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        vals = np.asarray(raw_box, dtype=np.float32).reshape(-1)
        if vals.size < 4 or not np.isfinite(vals[:4]).all():
            return None

        x1_raw, y1_raw, x2_raw, y2_raw = [float(v) for v in vals[:4]]

        candidates = [
            (x1_raw, y1_raw, x2_raw, y2_raw),
            (x1_raw - x2_raw / 2.0, y1_raw - y2_raw / 2.0, x1_raw + x2_raw / 2.0, y1_raw + y2_raw / 2.0),
            (x1_raw, y1_raw, x1_raw + x2_raw, y1_raw + y2_raw),
            (y1_raw, x1_raw, y2_raw, x2_raw),
            (y1_raw - y2_raw / 2.0, x1_raw - x2_raw / 2.0, y1_raw + y2_raw / 2.0, x1_raw + x2_raw / 2.0),
            (y1_raw, x1_raw, y1_raw + y2_raw, x1_raw + x2_raw),
        ]

        inv_ratio = 1.0 / max(ratio, 1e-6)
        best_box: tuple[int, int, int, int] | None = None
        # Pick the interpretation with the smallest clipping fraction — that is, the one
        # whose pre-clamp coordinates fit most naturally inside the frame.  The correct
        # coordinate format (e.g. xyxy in model pixel space) should need very little or
        # no clamping, while wrong formats (e.g. treating xyxy as xywh) will produce
        # coordinates that extend far outside the frame.
        best_clip_penalty = float("inf")

        for cx1, cy1, cx2, cy2 in candidates:
            rx1 = (cx1 - dw) * inv_ratio
            ry1 = (cy1 - dh) * inv_ratio
            rx2 = (cx2 - dw) * inv_ratio
            ry2 = (cy2 - dh) * inv_ratio

            if rx2 <= rx1 or ry2 <= ry1:
                continue

            box_w = rx2 - rx1
            box_h = ry2 - ry1

            # Fraction of the box that lies outside the frame on each axis.
            clip_x = max(0.0, -rx1) + max(0.0, rx2 - frame_width)
            clip_y = max(0.0, -ry1) + max(0.0, ry2 - frame_height)
            clip_penalty = clip_x / max(box_w, 1.0) + clip_y / max(box_h, 1.0)

            ix1 = max(0, min(int(round(rx1)), frame_width - 1))
            iy1 = max(0, min(int(round(ry1)), frame_height - 1))
            ix2 = max(0, min(int(round(rx2)), frame_width - 1))
            iy2 = max(0, min(int(round(ry2)), frame_height - 1))

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            if clip_penalty < best_clip_penalty:
                best_clip_penalty = clip_penalty
                best_box = (ix1, iy1, ix2, iy2)

        return best_box

    @staticmethod
    def _letterbox_for_rknn(frame: np.ndarray, target_size: int) -> tuple[np.ndarray, float, float, float]:
        h, w = frame.shape[:2]
        ratio = min(target_size / max(h, 1), target_size / max(w, 1))
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        dw = (target_size - new_w) / 2.0
        dh = (target_size - new_h) / 2.0
        top = int(round(dh - 0.1))
        left = int(round(dw - 0.1))
        canvas[top : top + new_h, left : left + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)
        chw = np.expand_dims(chw, axis=0)
        return chw, ratio, float(left), float(top)

    @staticmethod
    def _letterbox_for_rknn_nhwc(frame: np.ndarray, target_size: int) -> tuple[np.ndarray, float, float, float]:
        h, w = frame.shape[:2]
        ratio = min(target_size / max(h, 1), target_size / max(w, 1))
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        dw = (target_size - new_w) / 2.0
        dh = (target_size - new_h) / 2.0
        top = int(round(dh - 0.1))
        left = int(round(dw - 0.1))
        canvas[top : top + new_h, left : left + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        nhwc = np.expand_dims(rgb.astype(np.uint8), axis=0)
        return nhwc, ratio, float(left), float(top)

    @staticmethod
    def _rknn_input_variants(frame: np.ndarray, target_size: int) -> list[tuple[np.ndarray, float, float, float]]:
        nchw_uint8, ratio, dw, dh = VisionEngine._letterbox_for_rknn(frame, target_size)
        nhwc_uint8, ratio2, dw2, dh2 = VisionEngine._letterbox_for_rknn_nhwc(frame, target_size)

        return [
            (nchw_uint8, ratio, dw, dh),
            (nhwc_uint8, ratio2, dw2, dh2),
        ]

    @staticmethod
    def _decode_rknn_outputs(
        outputs: list[Any],
        conf_threshold: float,
        num_classes_hint: int = 0,
        input_size: int = 640,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not outputs:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        flat_boxes, flat_scores, flat_class_ids = VisionEngine._decode_rknn_single_flat_output(
            outputs,
            conf_threshold,
            num_classes_hint=num_classes_hint,
            input_size=input_size,
            iou_threshold=0.45,
        )
        if flat_boxes.size > 0:
            return flat_boxes, flat_scores, flat_class_ids

        # Prefer DFL-style head decoding when applicable (common YOLOv8 RKNN export).
        dfl_boxes, dfl_scores, dfl_class_ids = VisionEngine._decode_rknn_dfl_heads(
            outputs,
            conf_threshold,
            num_classes_hint=num_classes_hint,
            input_size=input_size,
            iou_threshold=0.45,
        )
        if dfl_boxes.size > 0:
            return dfl_boxes, dfl_scores, dfl_class_ids

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_class_ids: list[np.ndarray] = []

        for output_item in outputs:
            matrix = VisionEngine._normalize_rknn_output_matrix(np.asarray(output_item))
            if matrix is None or matrix.shape[1] < 6:
                continue

            boxes_xywh = matrix[:, :4].astype(np.float32)
            channels = matrix.shape[1]

            mode_candidates: list[bool] = []
            if num_classes_hint > 0:
                if channels == num_classes_hint + 5:
                    mode_candidates = [True]
                elif channels == num_classes_hint + 4:
                    mode_candidates = [False]
                else:
                    mode_candidates = [False, True]
            else:
                # Unknown class count: test both interpretations for ambiguous channel counts.
                mode_candidates = [False, True] if channels >= 7 else [False]

            best_keep = np.empty((0,), dtype=np.int32)
            best_scores = np.empty((0,), dtype=np.float32)
            best_class_ids = np.empty((0,), dtype=np.int32)

            for has_objectness in mode_candidates:
                if has_objectness:
                    objectness = matrix[:, 4].astype(np.float32)
                    cls_scores = matrix[:, 5:].astype(np.float32)
                else:
                    objectness = None
                    cls_scores = matrix[:, 4:].astype(np.float32)

                if cls_scores.size == 0:
                    continue

                if np.min(cls_scores) < 0.0 or np.max(cls_scores) > 1.0:
                    cls_scores = 1.0 / (1.0 + np.exp(-np.clip(cls_scores, -60.0, 60.0)))

                class_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
                cls_conf = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

                if objectness is not None:
                    if np.min(objectness) < 0.0 or np.max(objectness) > 1.0:
                        objectness = 1.0 / (1.0 + np.exp(-np.clip(objectness, -60.0, 60.0)))
                    scores = objectness * cls_conf
                else:
                    scores = cls_conf

                keep = np.where(scores >= conf_threshold)[0]
                if keep.size > best_keep.size:
                    best_keep = keep.astype(np.int32)
                    best_scores = scores[keep].astype(np.float32)
                    best_class_ids = class_ids[keep].astype(np.int32)

            if best_keep.size == 0:
                continue

            boxes_xywh = boxes_xywh[best_keep]
            scores = best_scores
            class_ids = best_class_ids

            xyxy_from_xywh = np.empty((boxes_xywh.shape[0], 4), dtype=np.float32)
            xyxy_from_xywh[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
            xyxy_from_xywh[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
            xyxy_from_xywh[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
            xyxy_from_xywh[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

            xyxy_from_xyxy = boxes_xywh.copy()

            # Some exports emit normalized coordinates in [0,1]. Scale to model input space.
            if np.max(xyxy_from_xywh) <= 2.5:
                xyxy_from_xywh *= float(input_size)
            if np.max(xyxy_from_xyxy) <= 2.5:
                xyxy_from_xyxy *= float(input_size)

            valid_xywh = np.isfinite(xyxy_from_xywh).all(axis=1) & (xyxy_from_xywh[:, 2] > xyxy_from_xywh[:, 0]) & (xyxy_from_xywh[:, 3] > xyxy_from_xywh[:, 1])
            valid_xyxy = np.isfinite(xyxy_from_xyxy).all(axis=1) & (xyxy_from_xyxy[:, 2] > xyxy_from_xyxy[:, 0]) & (xyxy_from_xyxy[:, 3] > xyxy_from_xyxy[:, 1])

            xyxy = xyxy_from_xywh if int(np.sum(valid_xywh)) >= int(np.sum(valid_xyxy)) else xyxy_from_xyxy

            all_boxes.append(xyxy)
            all_scores.append(scores.astype(np.float32))
            all_class_ids.append(class_ids.astype(np.int32))

        if not all_boxes:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        return (
            np.concatenate(all_boxes, axis=0),
            np.concatenate(all_scores, axis=0),
            np.concatenate(all_class_ids, axis=0),
        )

    @staticmethod
    def _decode_rknn_single_flat_output(
        outputs: list[Any],
        conf_threshold: float,
        num_classes_hint: int,
        input_size: int,
        iou_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(outputs) != 1:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        matrix = VisionEngine._normalize_rknn_output_matrix(np.asarray(outputs[0]))
        if matrix is None or matrix.shape[1] < 6:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        if num_classes_hint > 0 and matrix.shape[1] < num_classes_hint + 4:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        boxes_raw = matrix[:, :4].astype(np.float32)
        class_scores = matrix[:, 4:].astype(np.float32)
        if class_scores.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        if np.min(class_scores) < 0.0 or np.max(class_scores) > 1.0:
            class_scores = 1.0 / (1.0 + np.exp(-np.clip(class_scores, -60.0, 60.0)))

        if num_classes_hint > 0 and class_scores.shape[1] > num_classes_hint:
            class_scores = class_scores[:, :num_classes_hint]

        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids].astype(np.float32)

        if np.max(boxes_raw) <= 2.5:
            boxes_raw = boxes_raw * float(input_size)

        best_boxes = np.empty((0, 4), dtype=np.float32)
        best_scores = np.empty((0,), dtype=np.float32)
        best_class_ids = np.empty((0,), dtype=np.int32)
        best_score_sum = -1.0
        best_count = 0

        for interpret_as_xywh in (True, False):
            if interpret_as_xywh:
                boxes = np.empty_like(boxes_raw)
                boxes[:, 0] = boxes_raw[:, 0] - boxes_raw[:, 2] / 2.0
                boxes[:, 1] = boxes_raw[:, 1] - boxes_raw[:, 3] / 2.0
                boxes[:, 2] = boxes_raw[:, 0] + boxes_raw[:, 2] / 2.0
                boxes[:, 3] = boxes_raw[:, 1] + boxes_raw[:, 3] / 2.0
            else:
                boxes = boxes_raw.copy()

            valid = np.isfinite(boxes).all(axis=1)
            valid &= boxes[:, 2] > boxes[:, 0]
            valid &= boxes[:, 3] > boxes[:, 1]
            valid &= scores >= conf_threshold
            keep = np.where(valid)[0]
            if keep.size == 0:
                continue

            selected_boxes = boxes[keep]
            selected_scores = scores[keep]
            selected_class_ids = class_ids[keep]

            final_indices: list[int] = []
            for class_id in np.unique(selected_class_ids):
                class_mask = np.where(selected_class_ids == class_id)[0]
                class_keep = VisionEngine._nms_indices_numpy(
                    selected_boxes[class_mask],
                    selected_scores[class_mask],
                    iou_threshold,
                )
                final_indices.extend(class_mask[class_keep].tolist())

            if not final_indices:
                continue

            final_indices_array = np.asarray(final_indices, dtype=np.int32)
            final_boxes = selected_boxes[final_indices_array]
            final_scores = selected_scores[final_indices_array]
            final_class_ids = selected_class_ids[final_indices_array]

            if np.max(final_boxes) <= 2.5:
                final_boxes = final_boxes * float(input_size)

            score_sum = float(np.sum(final_scores))
            if final_boxes.shape[0] > best_count or (final_boxes.shape[0] == best_count and score_sum > best_score_sum):
                best_boxes = final_boxes
                best_scores = final_scores
                best_class_ids = final_class_ids
                best_score_sum = score_sum
                best_count = final_boxes.shape[0]

        if best_boxes.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        return best_boxes.astype(np.float32), best_scores.astype(np.float32), best_class_ids.astype(np.int32)

    @staticmethod
    def _decode_rknn_dfl_heads(
        outputs: list[Any],
        conf_threshold: float,
        num_classes_hint: int,
        input_size: int,
        iou_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        box_heads: list[np.ndarray] = []
        cls_heads: list[np.ndarray] = []

        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 3:
                continue

            ch = int(arr.shape[0])
            if ch >= 64 and ch % 4 == 0:
                box_heads.append(arr.astype(np.float32))
            elif ch >= 1:
                cls_heads.append(arr.astype(np.float32))

        if not box_heads or not cls_heads:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_class_ids: list[np.ndarray] = []

        # Pair heads by feature map size.
        for box in box_heads:
            _, h, w = box.shape
            cls = next((c for c in cls_heads if c.shape[1] == h and c.shape[2] == w), None)
            if cls is None:
                continue

            reg_max = box.shape[0] // 4
            if reg_max <= 1:
                continue

            # DFL decode: [4*reg_max, H, W] -> [H, W, 4]
            d = box.reshape(4, reg_max, h, w)
            d = np.transpose(d, (2, 3, 0, 1))
            d = d - np.max(d, axis=3, keepdims=True)
            exp_d = np.exp(np.clip(d, -50.0, 50.0))
            prob = exp_d / np.sum(exp_d, axis=3, keepdims=True)
            bins = np.arange(reg_max, dtype=np.float32)
            dist = np.sum(prob * bins, axis=3)

            # Classification: [C, H, W] -> [H*W, C]
            cls_scores = np.transpose(cls, (1, 2, 0)).reshape(-1, cls.shape[0])
            if np.min(cls_scores) < 0.0 or np.max(cls_scores) > 1.0:
                cls_scores = 1.0 / (1.0 + np.exp(-np.clip(cls_scores, -60.0, 60.0)))

            class_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

            if num_classes_hint > 0 and cls_scores.shape[1] > num_classes_hint:
                cls_scores = cls_scores[:, :num_classes_hint]
                class_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
                scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

            keep = scores >= conf_threshold
            if not np.any(keep):
                continue

            dist = dist.reshape(-1, 4)[keep]
            scores = scores[keep].astype(np.float32)
            class_ids = class_ids[keep].astype(np.int32)

            ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
            grid = np.stack([xs + 0.5, ys + 0.5], axis=-1).reshape(-1, 2)[keep]

            stride_x = input_size / float(max(w, 1))
            stride_y = input_size / float(max(h, 1))

            x1 = (grid[:, 0] - dist[:, 0]) * stride_x
            y1 = (grid[:, 1] - dist[:, 1]) * stride_y
            x2 = (grid[:, 0] + dist[:, 2]) * stride_x
            y2 = (grid[:, 1] + dist[:, 3]) * stride_y
            boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_class_ids.append(class_ids)

        if not all_boxes:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        return (
            np.concatenate(all_boxes, axis=0),
            np.concatenate(all_scores, axis=0),
            np.concatenate(all_class_ids, axis=0),
        )

    @staticmethod
    def _normalize_rknn_output_matrix(output: np.ndarray) -> np.ndarray | None:
        if output.ndim >= 3 and output.shape[0] == 1:
            output = output[0]

        if output.ndim == 4 and output.shape[0] == 1:
            output = output[0]

        # [C, N], [N, C], or [C, H, W]/[H, W, C] -> [N, C]
        if output.ndim == 3:
            if output.shape[0] <= 128 and output.shape[1] > output.shape[0]:
                return output.reshape(output.shape[0], -1).T
            if output.shape[-1] <= 128:
                return output.reshape(-1, output.shape[-1])
            return None

        if output.ndim == 2:
            # Normalize to [N, C]
            if output.shape[0] <= 128 and output.shape[1] > output.shape[0]:
                output = output.T
            return output

        return None

    @classmethod
    def _resolve_class_name(cls, class_id: int, labels: dict[int, str]) -> str:
        label = labels.get(class_id)
        if label:
            return str(label)
        if not labels and 0 <= class_id < len(cls.COCO80_CLASSES):
            return cls.COCO80_CLASSES[class_id]
        return f"class_{class_id}"

    @staticmethod
    def _nms_indices_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep_indices: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep_indices.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return np.asarray(keep_indices, dtype=np.int32)

    def _can_use_sahi(self) -> bool:
        if self._sahi_available is not None:
            return self._sahi_available

        try:
            from sahi import AutoDetectionModel  # noqa: F401
            from sahi.predict import get_sliced_prediction  # noqa: F401

            self._sahi_available = True
        except Exception:
            self._sahi_available = False
        return self._sahi_available

    def _predict_with_sahi(self, frame: np.ndarray, runtime: RuntimeOptions) -> list[Detection]:
        detections: list[Detection] = []
        if not self._active_model_path:
            return detections

        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            if self._sahi_model is None:
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=self._active_model_path,
                    confidence_threshold=runtime.confidence,
                    device="cpu",
                )

            prediction = get_sliced_prediction(
                image=frame[:, :, ::-1],
                detection_model=self._sahi_model,
                slice_height=runtime.sahi_slice_height,
                slice_width=runtime.sahi_slice_width,
                overlap_height_ratio=runtime.sahi_overlap_height_ratio,
                overlap_width_ratio=runtime.sahi_overlap_width_ratio,
                perform_standard_pred=False,
            )

            for item in prediction.object_prediction_list:
                bbox = item.bbox
                x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                class_name = str(item.category.name)
                score = float(item.score.value)
                detections.append(
                    Detection(
                        label=class_name,
                        class_name=class_name,
                        confidence=score,
                        box=(x1, y1, x2, y2),
                        kind="object",
                    )
                )
        except Exception as exc:
            self._status_note = f"SAHI inference failed: {exc}"

        return detections

    @staticmethod
    def _draw_box(
        image: np.ndarray,
        box: tuple[int, int, int, int],
        label: str,
        confidence: float,
        color: tuple[int, int, int],
        opacity: float,
    ) -> None:
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)

        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, opacity, image, 1.0 - opacity, 0.0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        caption = f"{label} {confidence:.2f}"
        text_size, baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_width, text_height = text_size
        top = max(y1 - text_height - baseline - 8, 0)
        cv2.rectangle(image, (x1, top), (x1 + text_width + 10, top + text_height + baseline + 8), color, -1)
        cv2.putText(
            image,
            caption,
            (x1 + 5, top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _color_for_label(label: str) -> tuple[int, int, int]:
        palette = [
            (46, 204, 113),
            (52, 152, 219),
            (231, 76, 60),
            (155, 89, 182),
            (241, 196, 15),
            (230, 126, 34),
            (26, 188, 156),
        ]
        index = sum(ord(char) for char in label) % len(palette)
        return palette[index]


class CameraWorker:
    def __init__(self, settings: Settings, vision: VisionEngine) -> None:
        self.settings = settings
        self.vision = vision
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._frame_event = threading.Event()
        self._latest_jpeg: bytes | None = None
        self._latest_detections: list[Detection] = []
        self._browser_frame: np.ndarray | None = None
        self._browser_frame_ts: float = 0.0
        self._latest_status: dict[str, Any] = {
            "camera_open": False,
            "frame_count": 0,
            "last_error": None,
            "known_faces": len(self.vision.face_analyzer.known_embeddings),
            "face_recognition_ready": self.vision.face_analyzer.enabled,
            "face_recognition_error": self.vision.face_analyzer.error,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def wait_for_frame(self, timeout: float = 5.0) -> bool:
        return self._frame_event.wait(timeout)

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            if self._latest_jpeg is not None:
                return self._latest_jpeg
        return self._placeholder_frame("Waiting for camera stream...")

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._latest_status)
        status.update(self.vision.get_runtime_snapshot())
        status["camera_source"] = self.vision.runtime.camera_source
        return status

    def get_latest_detections(self) -> list[Detection]:
        with self._lock:
            return list(self._latest_detections)

    def ingest_browser_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._browser_frame = frame.copy()
            self._browser_frame_ts = time.time()

    def _get_browser_frame(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._browser_frame is None:
                return None, 0.0
            return self._browser_frame.copy(), float(self._browser_frame_ts)

    def _run(self) -> None:
        frame_count = 0
        current_source = ""
        last_browser_frame_ts = 0.0

        while not self._stopped.is_set():
            runtime = self.vision.get_runtime_snapshot()
            requested_source = str(runtime.get("camera_source") or self.settings.camera_source)

            if requested_source.startswith("browser://"):
                if self._capture is not None:
                    self._capture.release()
                    self._capture = None

                browser_frame, browser_ts = self._get_browser_frame()
                if browser_frame is None:
                    self._store_placeholder("Waiting for browser camera frame...")
                    time.sleep(0.2)
                    continue
                if browser_ts <= last_browser_frame_ts:
                    time.sleep(0.01)
                    continue

                last_browser_frame_ts = browser_ts
                annotated, detections = self.vision.annotate(browser_frame)
                jpeg = self._encode_jpeg(annotated)
                if jpeg is None:
                    continue

                frame_count += 1
                with self._lock:
                    self._latest_jpeg = jpeg
                    self._latest_detections = detections
                    self._latest_status.update(
                        {
                            "camera_open": True,
                            "frame_count": frame_count,
                            "last_error": None,
                            "width": int(browser_frame.shape[1]),
                            "height": int(browser_frame.shape[0]),
                        }
                    )
                self._frame_event.set()
                time.sleep(max(1.0 / max(self.settings.camera_fps, 1), 0.01))
                continue

            if requested_source != current_source or self._capture is None or not self._capture.isOpened():
                if self._capture is not None:
                    self._capture.release()
                self._capture = self._open_capture(requested_source)
                current_source = requested_source
                frame_count = 0
                last_browser_frame_ts = 0.0

                if self._capture is None:
                    self._store_placeholder(f"Camera unavailable: {requested_source}")
                    time.sleep(1.0)
                    continue

            capture = self._capture
            assert capture is not None

            ok, frame = capture.read()
            if not ok or frame is None:
                self._store_placeholder("Frame capture failed")
                if self._capture is not None:
                    self._capture.release()
                self._capture = None
                time.sleep(0.4)
                continue

            annotated, detections = self.vision.annotate(frame)
            jpeg = self._encode_jpeg(annotated)
            if jpeg is None:
                continue

            frame_count += 1
            with self._lock:
                self._latest_jpeg = jpeg
                self._latest_detections = detections
                self._latest_status.update(
                    {
                        "camera_open": True,
                        "frame_count": frame_count,
                        "last_error": None,
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
            self._frame_event.set()
            time.sleep(max(1.0 / max(self.settings.camera_fps, 1), 0.01))

    def _open_capture(self, source: str) -> cv2.VideoCapture | None:
        source = source.strip()
        backend_v4l2 = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY

        if source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap
            cap.release()
            return None

        def _try_open(target: str | int, backend: int) -> cv2.VideoCapture | None:
            cap = cv2.VideoCapture(target, backend)
            if not cap.isOpened():
                cap.release()
                return None

            if isinstance(target, int) or str(target).startswith("/dev/video"):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.camera_width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.camera_height))
                cap.set(cv2.CAP_PROP_FPS, float(self.settings.camera_fps))
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warm up camera and verify at least one readable frame
            time.sleep(0.6)
            for _ in range(10):
                ok, _ = cap.read()
                if ok:
                    return cap
                time.sleep(0.12)

            cap.release()
            return None

        open_targets: list[str | int] = []
        if source.startswith("uvc://"):
            index_text = source.split("://", 1)[1].strip()
            index = int(index_text) if index_text.isdigit() else 0
            open_targets.append(index)
            open_targets.append(f"/dev/video{index}")
        elif source.isdigit():
            index = int(source)
            open_targets.append(index)
            open_targets.append(f"/dev/video{index}")
        else:
            open_targets.append(source)

        # Try V4L2 first, then a generic backend fallback.
        for target in open_targets:
            cap = _try_open(target, backend_v4l2)
            if cap is not None:
                return cap
            if backend_v4l2 != cv2.CAP_ANY:
                cap = _try_open(target, cv2.CAP_ANY)
                if cap is not None:
                    return cap

        return None

    def _store_placeholder(self, message: str) -> None:
        placeholder = self._placeholder_frame(message)
        with self._lock:
            self._latest_jpeg = placeholder
            self._latest_status.update(
                {
                    "camera_open": False,
                    "last_error": message,
                }
            )
        self._frame_event.set()

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes | None:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return buffer.tobytes()

    @staticmethod
    def _placeholder_frame(message: str) -> bytes:
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (18, 18), (1262, 702), (40, 40, 40), 2)
        cv2.putText(canvas, "Banana Pi Vision Lab", (54, 118), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, message, (54, 208), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 210, 120), 3, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Set camera source to uvc://0, rtsp://..., or rtmp://... from Settings.",
            (54, 278),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (190, 190, 190),
            2,
            cv2.LINE_AA,
        )
        ok, buffer = cv2.imencode(".jpg", canvas)
        if not ok:
            raise CameraError("Could not create placeholder frame")
        return buffer.tobytes()
