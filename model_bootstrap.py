from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from convert_model_to_json import _load_names_from_model


MODEL_SUFFIXES = {".pt", ".onnx", ".rknn"}
SIDECAR_SUFFIXES = {
    ".classes.json",
    ".json",
    ".labels.txt",
}


def _iter_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def _is_supported(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() in MODEL_SUFFIXES:
        return True
    return any(name.endswith(sfx) for sfx in SIDECAR_SUFFIXES)


def _copy_if_newer(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        src_stat = src.stat()
        dst_stat = dst.stat()
        if src_stat.st_mtime <= dst_stat.st_mtime and src_stat.st_size == dst_stat.st_size:
            return False
    shutil.copy2(src, dst)
    return True


def _sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".classes.json")


def _ensure_sidecar(model_path: Path) -> bool:
    sidecar = _sidecar_path(model_path)
    if sidecar.exists():
        return False

    names = _load_names_from_model(model_path)
    payload = {"names": names}
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def _choose_default_model(models_dir: Path, preferred: str | None) -> Path | None:
    if preferred:
        candidate = Path(preferred)
        if not candidate.is_absolute():
            candidate = models_dir / preferred
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in MODEL_SUFFIXES:
            return candidate

    models = sorted([p for p in _iter_files(models_dir) if p.suffix.lower() in MODEL_SUFFIXES])
    if not models:
        return None

    # Prefer RKNN on Banana Pi / Rockchip deployments, otherwise first available.
    rknn_models = [m for m in models if m.suffix.lower() == ".rknn"]
    return rknn_models[0] if rknn_models else models[0]


def main() -> int:
    models_dir = Path(os.getenv("MODELS_DIR", "/models")).resolve()
    import_dir = Path(os.getenv("MODEL_IMPORT_DIR", "/model-import")).resolve()
    preferred_model = os.getenv("YOLO_MODEL", "").strip() or None

    models_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    if import_dir.exists() and import_dir.is_dir():
        for src in _iter_files(import_dir):
            if not _is_supported(src):
                continue
            rel = src.relative_to(import_dir)
            dst = models_dir / rel
            if _copy_if_newer(src, dst):
                copied += 1

    generated_sidecars = 0
    sidecar_errors: list[str] = []
    for model_path in [p for p in _iter_files(models_dir) if p.suffix.lower() in MODEL_SUFFIXES]:
        try:
            if _ensure_sidecar(model_path):
                generated_sidecars += 1
        except Exception as exc:
            sidecar_errors.append(f"{model_path.name}: {exc}")

    selected = _choose_default_model(models_dir, preferred_model)
    if selected is not None:
        (models_dir / ".default_model").write_text(str(selected), encoding="utf-8")

    print(
        "[model_bootstrap] "
        f"models_dir={models_dir} import_dir={import_dir} copied={copied} generated_sidecars={generated_sidecars} "
        f"default_model={selected if selected else 'none'}"
    )
    if sidecar_errors:
        for err in sidecar_errors:
            print(f"[model_bootstrap] sidecar_error: {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
