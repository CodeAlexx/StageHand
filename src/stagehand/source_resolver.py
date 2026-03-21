"""Helpers for resolving Stagehand source references and Serenity manifests."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGEHAND_SOURCE_SUFFIXES = (".safetensors", ".fpk", ".slab")
_SHARDED_INDEX_NAMES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)
_SOURCE_MANIFEST_FORMAT = "serenity_source_manifest"


@dataclass(frozen=True)
class ResolvedStagehandSource:
    checkpoint_path: str | None
    file_backed_weights: bool
    manifest_path: str | None = None
    source_kind: str | None = None
    quant_mode: str | None = None


def normalize_stagehand_checkpoint_path(source_path: str | Path) -> tuple[str, bool]:
    """Normalize Stagehand checkpoint paths and detect file-backed compatibility."""

    path = Path(source_path).expanduser()
    lowered = str(path).lower()
    if path.is_file() and path.name == "model_index.json":
        path = path.parent / "transformer"
    elif path.is_dir() and (path / "model_index.json").exists():
        transformer_dir = path / "transformer"
        if transformer_dir.exists():
            path = transformer_dir
    elif any(lowered.endswith(index_name) for index_name in _SHARDED_INDEX_NAMES):
        path = path.parent

    is_sharded_dir = path.is_dir() and any((path / name).exists() for name in _SHARDED_INDEX_NAMES)
    file_backed = is_sharded_dir or str(path).lower().endswith(STAGEHAND_SOURCE_SUFFIXES)
    return str(path), file_backed


def resolve_stagehand_source_reference(source_reference: str | None) -> ResolvedStagehandSource | None:
    """Resolve a raw source path or Serenity `.source.json` manifest."""

    if not source_reference:
        return None

    source_path = Path(str(source_reference)).expanduser()
    manifest = _maybe_read_source_manifest(source_path)
    if manifest is None:
        checkpoint_path, file_backed = normalize_stagehand_checkpoint_path(source_path)
        return ResolvedStagehandSource(
            checkpoint_path=checkpoint_path,
            file_backed_weights=file_backed,
        )

    source = manifest.get("source", {})
    source_kind = _string_value(source.get("kind"))
    quant_mode = _string_value(manifest.get("quantization", {}).get("mode"))
    checkpoint_path: str | None = None
    file_backed = False

    for candidate in _candidate_manifest_paths(manifest):
        normalized_path, normalized_file_backed = normalize_stagehand_checkpoint_path(candidate)
        checkpoint_path = normalized_path
        file_backed = normalized_file_backed
        if normalized_file_backed:
            break

    return ResolvedStagehandSource(
        checkpoint_path=checkpoint_path,
        file_backed_weights=file_backed,
        manifest_path=str(source_path),
        source_kind=source_kind,
        quant_mode=quant_mode,
    )


def _maybe_read_source_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.suffix.lower() != ".json":
        return None

    manifest: dict[str, Any] | None = None
    try:
        from serenity_safetensors import read_manifest  # type: ignore

        loaded = read_manifest(str(path), True)
        if _looks_like_source_manifest(loaded):
            manifest = loaded
    except Exception:
        manifest = None

    if manifest is not None:
        return manifest

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not _looks_like_source_manifest(loaded):
        return None
    return _resolve_manifest_paths(loaded, path.parent)


def _looks_like_source_manifest(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("format") == _SOURCE_MANIFEST_FORMAT
        and isinstance(value.get("source"), dict)
        and isinstance(value.get("model"), dict)
    )


def _resolve_manifest_paths(manifest: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(manifest)
    source = resolved.get("source")
    if isinstance(source, dict):
        _resolve_path_field(source, "path", base_dir)
        _resolve_path_field(source, "original", base_dir)

    artifacts = resolved.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("path", "index", "block_map", "weights", "data_files", "files"):
            _resolve_path_field(artifacts, key, base_dir)
    return resolved


def _resolve_path_field(obj: dict[str, Any], key: str, base_dir: Path) -> None:
    value = obj.get(key)
    if isinstance(value, str) and value and not Path(value).is_absolute():
        obj[key] = str((base_dir / value).resolve())
    elif isinstance(value, list):
        obj[key] = [
            str((base_dir / item).resolve()) if isinstance(item, str) and item and not Path(item).is_absolute() else item
            for item in value
        ]


def _candidate_manifest_paths(manifest: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    source = manifest.get("source", {})
    artifacts = manifest.get("artifacts", {})

    for value in (
        source.get("path") if isinstance(source, dict) else None,
        artifacts.get("path") if isinstance(artifacts, dict) else None,
        artifacts.get("index") if isinstance(artifacts, dict) else None,
        artifacts.get("weights") if isinstance(artifacts, dict) else None,
    ):
        if isinstance(value, str) and value:
            candidates.append(value)

    if isinstance(artifacts, dict):
        for key in ("data_files", "files"):
            values = artifacts.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, str) and item:
                        candidates.append(item)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
