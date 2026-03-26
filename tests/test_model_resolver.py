"""Tests for the unified model directory resolver."""
from __future__ import annotations

import json

import pytest

from stagehand.model_resolver import (
    IndexEntry,
    ModelIndex,
    ModelResolver,
    ModelType,
    get_resolver,
    resolve,
    _resolver,
)
import stagehand.model_resolver as resolver_mod


# ── helpers ────────────────────────────────────────────────────────────────


def _make_model_dir(base: object, model_type: str, name: str, files: dict[str, bytes] | None = None) -> None:
    """Create a fake model directory with files.

    *files* maps filename -> content bytes.  Defaults to a single safetensors file.
    """
    model_dir = base / model_type / name  # type: ignore[operator]
    model_dir.mkdir(parents=True, exist_ok=True)
    if files is None:
        files = {"diffusion_pytorch_model.safetensors": b"\x00" * 1024}
    for fname, content in files.items():
        (model_dir / fname).write_bytes(content)


def _make_entry(name: str = "test-model", model_type: str = "checkpoints", path: str = "/fake") -> IndexEntry:
    return IndexEntry(
        name=name,
        model_type=model_type,
        format="safetensors",
        files=["diffusion_pytorch_model.safetensors"],
        total_bytes=1024,
        shard_count=1,
        path=path,
    )


# ── tests ──────────────────────────────────────────────────────────────────


def test_model_type_enum():
    """All types have correct string values matching directory names."""
    assert ModelType.CHECKPOINT.value == "checkpoints"
    assert ModelType.LORA.value == "loras"
    assert ModelType.VAE.value == "vaes"
    assert ModelType.TEXT_ENCODER.value == "text_encoders"
    assert ModelType.CONTROLNET.value == "controlnets"
    assert ModelType.IPADAPTER.value == "ipadapters"
    assert ModelType.UPSCALER.value == "upscalers"
    assert len(ModelType) == 7


def test_ensure_dirs(tmp_path):
    """ensure_dirs creates base_dir and all type subdirs."""
    base = tmp_path / "models"
    resolver = ModelResolver(base_dir=base)
    resolver.ensure_dirs()

    assert base.is_dir()
    for mt in ModelType:
        assert (base / mt.value).is_dir()


def test_resolve_from_index(tmp_path):
    """Finds a model via index lookup."""
    base = tmp_path / "models"
    _make_model_dir(base, "checkpoints", "flux-1-dev")

    resolver = ModelResolver(base_dir=base)
    entry = IndexEntry(
        name="flux-1-dev",
        model_type="checkpoints",
        format="safetensors",
        files=["diffusion_pytorch_model.safetensors"],
        total_bytes=1024,
        shard_count=1,
        path=str(base / "checkpoints" / "flux-1-dev"),
    )
    resolver.index.add(entry)

    result = resolver.resolve("flux-1-dev", ModelType.CHECKPOINT)
    assert result == base / "checkpoints" / "flux-1-dev"


def test_resolve_fallback_to_walk(tmp_path):
    """Finds a model via filesystem walk when not in index, and updates index."""
    base = tmp_path / "models"
    _make_model_dir(base, "loras", "my-style-lora")

    resolver = ModelResolver(base_dir=base)
    # Index is empty — should walk.
    assert resolver.index.find("my-style-lora") is None

    result = resolver.resolve("my-style-lora", ModelType.LORA)
    assert result == base / "loras" / "my-style-lora"

    # Index should now contain the entry.
    entry = resolver.index.find("my-style-lora", "loras")
    assert entry is not None
    assert entry.format == "safetensors"
    assert entry.shard_count == 1


def test_resolve_not_found(tmp_path):
    """Raises FileNotFoundError for missing models."""
    base = tmp_path / "models"
    base.mkdir(parents=True)
    resolver = ModelResolver(base_dir=base)
    resolver.ensure_dirs()

    with pytest.raises(FileNotFoundError, match="nonexistent"):
        resolver.resolve("nonexistent")


def test_resolve_file_single(tmp_path):
    """resolve_file returns the single safetensors file path."""
    base = tmp_path / "models"
    _make_model_dir(base, "vaes", "sdxl-vae")

    resolver = ModelResolver(base_dir=base)
    entry = IndexEntry(
        name="sdxl-vae",
        model_type="vaes",
        format="safetensors",
        files=["diffusion_pytorch_model.safetensors"],
        total_bytes=1024,
        shard_count=1,
        path=str(base / "vaes" / "sdxl-vae"),
    )
    resolver.index.add(entry)

    result = resolver.resolve_file("sdxl-vae", ModelType.VAE)
    assert result == base / "vaes" / "sdxl-vae" / "diffusion_pytorch_model.safetensors"


def test_resolve_file_sharded(tmp_path):
    """resolve_file returns the directory for sharded models."""
    base = tmp_path / "models"
    shards = {
        f"diffusion_pytorch_model-0000{i}-of-00003.safetensors": b"\x00" * 512
        for i in range(1, 4)
    }
    _make_model_dir(base, "checkpoints", "big-model", files=shards)

    resolver = ModelResolver(base_dir=base)
    entry = IndexEntry(
        name="big-model",
        model_type="checkpoints",
        format="safetensors",
        files=list(shards.keys()),
        total_bytes=1536,
        shard_count=3,
        path=str(base / "checkpoints" / "big-model"),
    )
    resolver.index.add(entry)

    result = resolver.resolve_file("big-model", ModelType.CHECKPOINT)
    # Sharded → returns directory, not a specific file.
    assert result == base / "checkpoints" / "big-model"


def test_index_save_load_roundtrip(tmp_path):
    """Save and reload index.json preserves all data."""
    index_path = tmp_path / "index.json"
    idx = ModelIndex()
    entry = _make_entry(name="roundtrip-model", path="/some/path")
    idx.add(entry)
    idx.save(index_path)

    loaded = ModelIndex.load(index_path)
    found = loaded.find("roundtrip-model", "checkpoints")
    assert found is not None
    assert found.name == "roundtrip-model"
    assert found.format == "safetensors"
    assert found.total_bytes == 1024
    assert found.shard_count == 1

    # Verify JSON structure.
    with open(index_path) as f:
        raw = json.load(f)
    assert raw["version"] == 1
    assert "updated_at" in raw
    assert len(raw["models"]) == 1


def test_index_add_remove():
    """Add entry, verify present, remove, verify gone."""
    idx = ModelIndex()

    entry = _make_entry(name="temp-model", model_type="loras")
    idx.add(entry)
    assert idx.find("temp-model", "loras") is not None

    idx.remove("temp-model", "loras")
    assert idx.find("temp-model", "loras") is None


def test_list_models_by_type(tmp_path):
    """Filter by model_type returns only matching entries."""
    base = tmp_path / "models"
    resolver = ModelResolver(base_dir=base)

    resolver.index.add(_make_entry(name="m1", model_type="checkpoints", path="/a"))
    resolver.index.add(_make_entry(name="m2", model_type="checkpoints", path="/b"))
    resolver.index.add(_make_entry(name="m3", model_type="loras", path="/c"))

    ckpts = resolver.list_models(ModelType.CHECKPOINT)
    assert len(ckpts) == 2
    assert {e.name for e in ckpts} == {"m1", "m2"}

    loras = resolver.list_models(ModelType.LORA)
    assert len(loras) == 1
    assert loras[0].name == "m3"

    all_models = resolver.list_models()
    assert len(all_models) == 3


def test_refresh_index(tmp_path):
    """Scan directory tree and rebuild index from filesystem."""
    base = tmp_path / "models"

    # Create several model dirs across types.
    _make_model_dir(base, "checkpoints", "model-a")
    _make_model_dir(base, "loras", "style-lora", files={
        "adapter.safetensors": b"\x00" * 2048,
    })
    _make_model_dir(base, "vaes", "my-vae", files={
        "model.gguf": b"\x00" * 512,
    })
    # Also create the remaining type dirs (empty).
    for mt in ModelType:
        (base / mt.value).mkdir(parents=True, exist_ok=True)

    resolver = ModelResolver(base_dir=base)
    idx = resolver.refresh_index()

    assert len(idx.list_all()) == 3

    ckpt = idx.find("model-a", "checkpoints")
    assert ckpt is not None
    assert ckpt.format == "safetensors"
    assert ckpt.shard_count == 1
    assert ckpt.total_bytes == 1024

    lora = idx.find("style-lora", "loras")
    assert lora is not None
    assert lora.format == "safetensors"
    assert lora.total_bytes == 2048

    vae = idx.find("my-vae", "vaes")
    assert vae is not None
    assert vae.format == "gguf"
    assert vae.total_bytes == 512

    # Verify index.json was written.
    assert (base / "index.json").exists()


def test_singleton(tmp_path, monkeypatch):
    """get_resolver() returns same instance on repeated calls."""
    # Reset global state.
    monkeypatch.setattr(resolver_mod, "_resolver", None)
    monkeypatch.setenv("HOME", str(tmp_path))
    # Patch Path.home to return tmp_path so it doesn't touch real ~/.serenity
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))

    r1 = get_resolver()
    r2 = get_resolver()
    assert r1 is r2

    # Cleanup.
    monkeypatch.setattr(resolver_mod, "_resolver", None)
