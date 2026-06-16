from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize_names(names: Any) -> dict[str, str]:
    if isinstance(names, dict):
        ordered_keys = sorted(names.keys(), key=lambda key: int(key) if str(key).isdigit() else str(key))
        return {str(key): str(names[key]) for key in ordered_keys}

    if isinstance(names, list):
        return {str(index): str(name) for index, name in enumerate(names)}

    raise TypeError("Model names must be a list or dict")


def _load_json_like(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return {str(index): str(name) for index, name in enumerate(payload)}

    if isinstance(payload, dict):
        for key in ("names", "classes", "labels"):
            value = payload.get(key)
            if isinstance(value, (list, dict)):
                return _normalize_names(value)

    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_names_from_model(model_path: Path) -> dict[str, str]:
    if model_path.suffix.lower() in {".json", ".txt"}:
        if model_path.suffix.lower() == ".txt":
            labels = [line.strip() for line in model_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if not labels:
                raise ValueError(f"No labels found in {model_path}")
            return {str(index): label for index, label in enumerate(labels)}
        return _load_json_like(model_path)

    sidecar_candidates = [
        model_path.with_suffix(model_path.suffix + ".classes.json"),
        model_path.with_suffix(".classes.json"),
        model_path.with_suffix(".json"),
        model_path.with_suffix(".labels.txt"),
    ]
    for sidecar in sidecar_candidates:
        if not sidecar.exists():
            continue
        try:
            if sidecar.suffix == ".txt":
                labels = [line.strip() for line in sidecar.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
                if labels:
                    return {str(index): label for index, label in enumerate(labels)}
            else:
                return _load_json_like(sidecar)
        except Exception:
            continue

    try:
        from ultralytics import YOLO

        loaded = YOLO(str(model_path))
        names = getattr(loaded, "names", None)
        if names:
            return _normalize_names(names)
    except Exception:
        pass

    metadata_candidates = [
        model_path.with_name("metadata.yaml"),
        model_path.with_suffix(".yaml"),
        model_path.with_suffix(".yml"),
    ]
    for metadata_path in metadata_candidates:
        if not metadata_path.exists():
            continue
        try:
            import yaml

            payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue

        if isinstance(payload, dict):
            for key in ("names", "classes", "labels"):
                value = payload.get(key)
                if isinstance(value, (list, dict)):
                    return _normalize_names(value)

    raise ValueError(
        f"Could not extract class names from {model_path}. Provide a .pt/.onnx model or a matching sidecar JSON/TXT file."
    )


def _default_output_path(model_path: Path) -> Path:
    if model_path.suffix.lower() == ".json":
        return model_path
    return model_path.with_suffix(model_path.suffix + ".classes.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a YOLO model or sidecar labels file to JSON.")
    parser.add_argument("model", help="Path to the model or labels file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file path. Defaults to the model sidecar path, such as model.pt.classes.json",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for the output file",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_path = Path(args.model).expanduser().resolve()

    if not model_path.exists():
        raise SystemExit(f"Input file not found: {model_path}")
    if not model_path.is_file():
        raise SystemExit(f"Input path is not a file: {model_path}")

    names = _load_names_from_model(model_path)
    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(model_path)

    payload = {"names": names}
    output_path.write_text(json.dumps(payload, indent=args.indent, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())