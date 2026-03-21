"""Unified model directory system for Serenity ecosystem.

Single source of truth at ~/.serenity/models/ with type-based subdirs.
Both Serenity (trainer) and SerenityFlow (workflow UI) use this resolver.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

__all__ = [
    "ModelType",
    "IndexEntry",
    "ModelIndex",
    "ModelResolver",
    "get_resolver",
    "resolve",
]

# Known model file extensions for scanning.
_MODEL_EXTENSIONS = frozenset({
    ".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt",
})


class ModelType(str, Enum):
    """Subdirectory categories within the model store."""

    CHECKPOINT = "checkpoints"
    LORA = "loras"
    VAE = "vaes"
    TEXT_ENCODER = "text_encoders"
    CONTROLNET = "controlnets"
    IPADAPTER = "ipadapters"
    UPSCALER = "upscalers"


@dataclass
class IndexEntry:
    """One model directory registered in the index."""

    name: str
    model_type: str
    format: str
    files: list[str]
    total_bytes: int
    shard_count: int
    path: str
    # Optional metadata.
    architecture: str | None = None
    added_at: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop None optional fields for cleaner JSON.
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> IndexEntry:
        return cls(
            name=d["name"],
            model_type=d["model_type"],
            format=d["format"],
            files=d["files"],
            total_bytes=d["total_bytes"],
            shard_count=d["shard_count"],
            path=d["path"],
            architecture=d.get("architecture"),
            added_at=d.get("added_at"),
        )


class ModelIndex:
    """In-memory representation of index.json."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], IndexEntry] = {}
        self.updated_at: str | None = None

    # ── persistence ────────────────────────────────────────────────────

    @classmethod
    def load(cls, index_path: Path) -> ModelIndex:
        """Read index.json from disk."""
        idx = cls()
        if not index_path.exists():
            return idx
        with open(index_path, "r") as f:
            data = json.load(f)
        idx.updated_at = data.get("updated_at")
        for m in data.get("models", []):
            entry = IndexEntry.from_dict(m)
            idx._entries[(entry.name, entry.model_type)] = entry
        return idx

    def save(self, index_path: Path) -> None:
        """Write index.json to disk."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        data = {
            "version": 1,
            "updated_at": self.updated_at,
            "models": [e.to_dict() for e in self._entries.values()],
        }
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── CRUD ───────────────────────────────────────────────────────────

    def add(self, entry: IndexEntry) -> None:
        """Add or update an entry."""
        self._entries[(entry.name, entry.model_type)] = entry

    def remove(self, name: str, model_type: str) -> None:
        """Remove an entry. No-op if not present."""
        self._entries.pop((name, model_type), None)

    def find(self, name: str, model_type: str | None = None) -> IndexEntry | None:
        """Lookup by name and optional type. Returns first match if type is None."""
        if model_type is not None:
            return self._entries.get((name, model_type))
        for (n, _), entry in self._entries.items():
            if n == name:
                return entry
        return None

    def list_type(self, model_type: str) -> list[IndexEntry]:
        """All entries of a given type."""
        return [e for e in self._entries.values() if e.model_type == model_type]

    def list_all(self) -> list[IndexEntry]:
        """All entries."""
        return list(self._entries.values())


def _detect_format(files: list[str]) -> str:
    """Detect model format from file extensions."""
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".safetensors":
            return "safetensors"
        if ext == ".gguf":
            return "gguf"
        if ext == ".bin":
            return "bin"
        if ext in (".pt", ".pth", ".ckpt"):
            return "pt"
    return "unknown"


def _scan_model_dir(model_dir: Path) -> tuple[list[str], int]:
    """Return (relative file list, total bytes) for model files in a directory."""
    files: list[str] = []
    total = 0
    for entry in sorted(model_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _MODEL_EXTENSIONS:
            files.append(entry.name)
            total += entry.stat().st_size
    return files, total


class ModelResolver:
    """Resolve model names to filesystem paths.

    Primary resolver for the Serenity ecosystem. Uses ~/.serenity/models/
    as the canonical model directory with index.json for fast lookups.
    Falls back to directory walk if index is stale.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / ".serenity" / "models"
        self.base_dir = Path(base_dir)
        self._index: ModelIndex | None = None

    # ── index property ─────────────────────────────────────────────────

    @property
    def index(self) -> ModelIndex:
        """Lazy-load index from disk."""
        if self._index is None:
            self._index = ModelIndex.load(self.base_dir / "index.json")
        return self._index

    # ── resolve ────────────────────────────────────────────────────────

    def resolve(self, name: str, model_type: str | ModelType | None = None) -> Path:
        """Resolve a model name to its filesystem path (directory).

        1. Check index first (fast).
        2. If not in index, walk the directory (slow, updates index).
        3. Raise FileNotFoundError if not found.
        """
        mt = model_type.value if isinstance(model_type, ModelType) else model_type

        # Fast path: index lookup.
        entry = self.index.find(name, mt)
        if entry is not None:
            p = Path(entry.path)
            if p.exists():
                return p

        # Slow path: walk filesystem.
        entry = self._walk_and_find(name, mt)
        if entry is not None:
            return Path(entry.path)

        raise FileNotFoundError(
            f"Model '{name}' not found"
            + (f" (type={mt})" if mt else "")
            + f" in {self.base_dir}"
        )

    def resolve_file(self, name: str, model_type: str | ModelType | None = None) -> Path:
        """Resolve to a specific model file.

        For single-file models, returns the file path directly.
        For sharded models, returns the directory.
        """
        mt = model_type.value if isinstance(model_type, ModelType) else model_type
        model_dir = self.resolve(name, mt)

        entry = self.index.find(name, mt)
        if entry is not None and entry.shard_count == 1 and entry.files:
            return model_dir / entry.files[0]
        return model_dir

    # ── directory management ───────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create base_dir and all type subdirs if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for mt in ModelType:
            (self.base_dir / mt.value).mkdir(exist_ok=True)

    # ── index management ───────────────────────────────────────────────

    def refresh_index(self) -> ModelIndex:
        """Scan the directory tree and rebuild index.json from the filesystem."""
        idx = ModelIndex()
        for mt in ModelType:
            type_dir = self.base_dir / mt.value
            if not type_dir.is_dir():
                continue
            for child in sorted(type_dir.iterdir()):
                if child.is_dir() or (child.is_symlink() and child.resolve().is_dir()):
                    # Directory (or symlink to directory) — scan for model files inside
                    files, total = _scan_model_dir(child)
                    if not files:
                        continue
                    fmt = _detect_format(files)
                    entry = IndexEntry(
                        name=child.name,
                        model_type=mt.value,
                        format=fmt,
                        files=files,
                        total_bytes=total,
                        shard_count=len(files),
                        path=str(child),
                        added_at=datetime.now(timezone.utc).isoformat(),
                    )
                    idx.add(entry)
                elif child.is_file() and child.suffix.lower() in _MODEL_EXTENSIONS:
                    # Individual model file (or symlink to file)
                    try:
                        size = child.resolve().stat().st_size
                    except OSError:
                        continue
                    entry = IndexEntry(
                        name=child.stem,
                        model_type=mt.value,
                        format=child.suffix.lstrip("."),
                        files=[child.name],
                        total_bytes=size,
                        shard_count=1,
                        path=str(child.parent),
                        added_at=datetime.now(timezone.utc).isoformat(),
                    )
                    idx.add(entry)

        idx.save(self.base_dir / "index.json")
        self._index = idx
        return idx

    def list_models(self, model_type: str | ModelType | None = None) -> list[IndexEntry]:
        """List all models, optionally filtered by type."""
        if model_type is None:
            return self.index.list_all()
        mt = model_type.value if isinstance(model_type, ModelType) else model_type
        return self.index.list_type(mt)

    # ── internal helpers ───────────────────────────────────────────────

    def _walk_and_find(self, name: str, model_type: str | None) -> IndexEntry | None:
        """Walk filesystem looking for a model, update index if found."""
        types_to_check = [ModelType(model_type)] if model_type else list(ModelType)

        for mt in types_to_check:
            candidate = self.base_dir / mt.value / name
            if candidate.is_dir() or (candidate.is_symlink() and candidate.resolve().is_dir()):
                files, total = _scan_model_dir(candidate)
                if not files:
                    continue
                fmt = _detect_format(files)
                entry = IndexEntry(
                    name=name,
                    model_type=mt.value,
                    format=fmt,
                    files=files,
                    total_bytes=total,
                    shard_count=len(files),
                    path=str(candidate),
                    added_at=datetime.now(timezone.utc).isoformat(),
                )
            elif candidate.is_file() or (candidate.is_symlink() and candidate.resolve().is_file()):
                try:
                    size = candidate.resolve().stat().st_size
                except OSError:
                    continue
                entry = IndexEntry(
                    name=name,
                    model_type=mt.value,
                    format=candidate.suffix.lstrip("."),
                    files=[candidate.name],
                    total_bytes=size,
                    shard_count=1,
                    path=str(candidate.parent),
                    added_at=datetime.now(timezone.utc).isoformat(),
                )
            else:
                # Also try with common extensions appended
                found = False
                for ext in (".safetensors", ".gguf", ".bin", ".pth"):
                    candidate_ext = self.base_dir / mt.value / (name + ext)
                    if candidate_ext.exists():
                        try:
                            size = candidate_ext.resolve().stat().st_size
                        except OSError:
                            continue
                        entry = IndexEntry(
                            name=name,
                            model_type=mt.value,
                            format=ext.lstrip("."),
                            files=[candidate_ext.name],
                            total_bytes=size,
                            shard_count=1,
                            path=str(candidate_ext.parent),
                            added_at=datetime.now(timezone.utc).isoformat(),
                        )
                        found = True
                        break
                if not found:
                    continue
            # Update index with discovered entry.
            self.index.add(entry)
            self.index.save(self.base_dir / "index.json")
            return entry

        return None


# ── module-level convenience ───────────────────────────────────────────────

_resolver: ModelResolver | None = None


def get_resolver() -> ModelResolver:
    """Get or create the global resolver singleton."""
    global _resolver
    if _resolver is None:
        _resolver = ModelResolver()
    return _resolver


def resolve(name: str, model_type: str | None = None) -> Path:
    """Shortcut: resolve a model name to path."""
    return get_resolver().resolve(name, model_type)
