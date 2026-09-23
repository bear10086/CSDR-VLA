"""Byte-offset random access for uncompressed TFDS TFRecord shards."""

from __future__ import annotations

import json
import os
import struct
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


def scan_tfrecord_offsets(path: Path) -> list[tuple[int, int]]:
    records = []
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        while handle.tell() < file_size:
            header_offset = handle.tell()
            header = handle.read(12)
            if not header:
                break
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header at {path}:{header_offset}.")
            length = struct.unpack("<Q", header[:8])[0]
            data_offset = header_offset + 12
            next_offset = data_offset + length + 4
            if next_offset > file_size:
                raise ValueError(f"Invalid TFRecord length at {path}:{header_offset}.")
            records.append((data_offset, length))
            handle.seek(next_offset)
    return records


def build_tfrecord_index(dataset_dir: Path) -> dict:
    files = sorted(dataset_dir.glob("*-train.tfrecord-*"))
    if not files:
        raise FileNotFoundError(f"No train TFRecord shards in {dataset_dir}.")
    return {
        "version": 1,
        "dataset_dir": str(dataset_dir),
        "files": {
            path.name: {
                "path": str(path),
                "records": scan_tfrecord_offsets(path),
            }
            for path in files
        },
    }


def split_tfds_id(tfds_id: str) -> tuple[str, int]:
    try:
        filename, record_index = tfds_id.rsplit("__", 1)
        return filename, int(record_index)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Unsupported TFDS id: {tfds_id!r}.") from error


class RandomAccessTFRecordStore:
    def __init__(self, index_path: Path, max_open_files: int = 32) -> None:
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.max_open_files = max(int(max_open_files), 1)
        self._handles: OrderedDict[str, Any] = OrderedDict()
        self._handle_lock = threading.Lock()

    def close(self) -> None:
        with self._handle_lock:
            while self._handles:
                _, handle = self._handles.popitem()
                handle.close()

    def _handle(self, path: str):
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = open(path, "rb")
        self._handles[path] = handle
        while len(self._handles) > self.max_open_files:
            _, stale = self._handles.popitem(last=False)
            stale.close()
        return handle

    def serialized_episode(self, tfds_id: str) -> bytes:
        filename, record_index = split_tfds_id(tfds_id)
        file_entry = self.index["files"].get(filename)
        if file_entry is None:
            raise KeyError(f"TFRecord shard {filename!r} is absent from the index.")
        try:
            data_offset, length = file_entry["records"][record_index]
        except IndexError as error:
            raise IndexError(f"Record {record_index} is absent from {filename}.") from error
        # pread avoids mutating the shared file offset. Keep handle lookup and
        # the short read atomic so LRU eviction cannot close an in-use handle.
        with self._handle_lock:
            handle = self._handle(file_entry["path"])
            payload = os.pread(handle.fileno(), length, data_offset)
        if len(payload) != length:
            raise IOError(f"Short TFRecord read for {tfds_id}.")
        return payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
