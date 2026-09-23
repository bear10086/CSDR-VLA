"""Content identities and isolated, atomically published evaluation merges."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def fingerprint_files(paths):
    digest = hashlib.sha256()
    for value in paths:
        path = Path(value)
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
            if file_hash is None:
                file_hash = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    file_hash.update(block)
        digest.update(file_hash.digest())
    return digest.hexdigest()


def checkpoint_identity(directory):
    directory = Path(directory).resolve()
    files = sorted(p for p in directory.iterdir() if p.is_file() and (
        p.suffix in (".json", ".safetensors", ".bin", ".py", ".model", ".txt")
    ) and p.name != "merge_complete.json")
    if not files:
        raise ValueError(f"Checkpoint contains no model files: {directory}")
    return {"path": str(directory), "content_sha256": fingerprint_files(files)}


def cached_merge(cache_root, identity, merge):
    """Call merge(output_dir) once per identity; never reuse a basename-only cache."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = cache_root / ("checkpoint-" + key)
    lock = cache_root / (key + ".lock")
    deadline = time.monotonic() + 3600
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Merge lock remains held: {lock}")
            time.sleep(1)
    try:
        marker = target / "merge_complete.json"
        if target.exists():
            if not marker.is_file() or json.loads(marker.read_text()) != identity:
                raise RuntimeError(f"Invalid merged cache; refusing reuse: {target}")
            if not (target / "config.json").is_file() or not any(target.glob("*.safetensors")) and not any(target.glob("pytorch_model*.bin")):
                raise RuntimeError(f"Incomplete merged model: {target}")
            return target
        temporary = Path(tempfile.mkdtemp(prefix=key + ".partial-", dir=cache_root))
        merge(temporary)
        if not (temporary / "config.json").is_file() or not any(temporary.glob("*.safetensors")) and not any(temporary.glob("pytorch_model*.bin")):
            raise RuntimeError(f"Merge produced no complete model: {temporary}")
        (temporary / "merge_complete.json").write_text(json.dumps(identity, indent=2))
        temporary.rename(target)
        return target
    finally:
        lock.unlink()


def cursor_after_updates(epoch_micro_batches, accumulation, updates):
    """Count consumed batches, including short accumulation groups at epoch ends."""
    if accumulation < 1 or updates < 0:
        raise ValueError("Invalid accumulation or update count")
    cursor = 0
    for batches in epoch_micro_batches:
        epoch_updates = (batches + accumulation - 1) // accumulation
        if updates >= epoch_updates:
            cursor += batches
            updates -= epoch_updates
        else:
            return cursor + updates * accumulation
    if updates:
        raise ValueError("Saved updates exceed the training plan")
    return cursor
