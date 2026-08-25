"""Model discovery and reproducibility checks."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path


MODEL_RELEASE = "1.0.1"
MODEL_ARTIFACTS = {
    "multiobjective_model.pt": {
        "sha256": "1b227cb5acd192fda07febcdd4a6eafa8cd6894eb695a0ca5ca34de45e98a887",
        "bytes": 906449706,
    },
    "biosignal_expert.joblib": {
        "sha256": "5d8c163f12d00b89004610df780fe14e8ce40f0bddc608197a6c3cffb140c372",
        "bytes": 296887858,
    },
    "tfidf/model.npy": {
        "sha256": "2beeb73b339dab892cf68e43afb55194a7a7a4ede5fca6594139af396efa4715",
        "bytes": 65664,
    },
}
MODEL_URL_ROOT = f"https://github.com/shwenyu/Tiara2/raw/v{MODEL_RELEASE}/tiara/models/default"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def packaged_bundle() -> Path:
    return Path(__file__).resolve().parent / "models" / "default" / "model_manifest.json"


def model_cache_dir() -> Path:
    override = os.environ.get("TIARA2_MODEL_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        root = Path(os.environ["XDG_CACHE_HOME"])
    else:
        root = Path.home() / ".cache"
    return (root / "tiara2" / "models" / f"v{MODEL_RELEASE}").resolve()


def default_bundle() -> Path:
    override = os.environ.get("TIARA2_MODEL_BUNDLE")
    if override:
        return Path(override).expanduser().resolve()
    cached = model_cache_dir() / "model_manifest.json"
    return cached if cached.is_file() else packaged_bundle()


def _download(url: str, destination: Path, expected: str, expected_bytes=None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Tiara2-model-downloader"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            while block := response.read(8 << 20):
                handle.write(block)
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"downloaded SHA-256 mismatch for {destination.name}: expected {expected}, got {actual}")
        if expected_bytes is not None and partial.stat().st_size != expected_bytes:
            raise ValueError(
                f"downloaded size mismatch for {destination.name}: "
                f"expected {expected_bytes}, got {partial.stat().st_size}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": expected}


def download_bundle(destination=None, *, force=False, url_root=MODEL_URL_ROOT):
    """Download the frozen model bundle to a user cache with SHA-256 verification."""
    target = Path(destination).expanduser().resolve() if destination else model_cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for relative, metadata in MODEL_ARTIFACTS.items():
        path = target / relative
        if not force and path.is_file() and sha256(path) == metadata["sha256"]:
            downloaded.append({"path": str(path), "bytes": path.stat().st_size, "sha256": metadata["sha256"], "cached": True})
            continue
        url = f"{url_root.rstrip('/')}/{relative}"
        downloaded.append(_download(url, path, metadata["sha256"], metadata["bytes"]))

    source_dir = packaged_bundle().parent
    shutil.copyfile(source_dir / "model_manifest.json", target / "model_manifest.json")
    (target / "tfidf").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / "tfidf" / "params.txt", target / "tfidf" / "params.txt")
    report = verify_bundle(target / "model_manifest.json")
    report.update({"model_release": MODEL_RELEASE, "downloaded": downloaded})
    return report


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
            raise FileNotFoundError(
                f"missing {name}: {path}; run `tiara2-download-models` or provide --bundle"
            )
        with path.open("rb") as handle:
            if handle.read(42).startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError(
                    f"{name} is a Git LFS pointer, not model data; run `tiara2-download-models`"
                )
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
        checked.append({"name": name, "bytes": path.stat().st_size, "sha256": actual})
    return {"ok": True, "name": manifest.get("name", "Tiara2"), "manifest": str(manifest_file), "artifacts": checked}
