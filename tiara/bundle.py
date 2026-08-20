"""Model discovery and reproducibility checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_bundle() -> Path:
    override = os.environ.get("TIARA2_MODEL_BUNDLE")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "models" / "default" / "model_manifest.json"


def manifest_path(bundle=None) -> Path:
    path = Path(bundle).expanduser() if bundle else default_bundle()
    if path.is_dir():
        path = path / "model_manifest.json"
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tiara2 model manifest not found: {path}")
    return path


def verify_bundle(bundle=None):
    manifest_file = manifest_path(bundle)
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("format") != "tiara2-biosignal-residual-v1":
        raise ValueError("unsupported Tiara2 model format")

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else (manifest_file.parent / path).resolve()

    artifacts = [
        ("base", resolve(manifest["base"]["checkpoint"]), manifest["base"]["sha256"]),
        ("expert", resolve(manifest["expert"]["model"]), manifest["expert"]["sha256"]),
    ]
    tfidf_dir = resolve(manifest["tfidf"])
    for name, expected in manifest["tfidf_sha256"].items():
        artifacts.append((f"tfidf/{name}", tfidf_dir / name, expected))
    checked = []
    for name, path, expected in artifacts:
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}; run `git lfs pull` in the Tiara2 repository")
        with path.open("rb") as handle:
            if handle.read(42).startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError(f"{name} is a Git LFS pointer, not model data; run `git lfs pull`")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
        checked.append({"name": name, "bytes": path.stat().st_size, "sha256": actual})
    return {"ok": True, "name": manifest.get("name", "Tiara2"), "manifest": str(manifest_file), "artifacts": checked}
