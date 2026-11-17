"""Download and verify the six paper benchmark HDF5 files.

The public release URL is read from data/benchmark_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import certifi


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/benchmark_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"unsafe benchmark path: {relative}")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe benchmark path: {relative}")
    return path


def valid(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry["size_bytes"] and sha256(path) == entry["sha256"]


def download(url: str, destination: Path, expected_size: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "EntNet-benchmark-downloader/1"})
    try:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=60, context=tls_context) as response, destination.open("wb") as output:
            transferred = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                transferred += len(block)
                print(f"\rDownloading: {transferred / 1024**2:.1f}/{expected_size / 1024**2:.1f} MiB",
                      end="", file=sys.stderr, flush=True)
    except (OSError, urllib.error.URLError):
        destination.unlink(missing_ok=True)
        raise
    print(file=sys.stderr)


def extract_verified(archive: Path, target: Path, files: list[dict]) -> None:
    expected = {entry["path"]: entry for entry in files}
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        if set(members) != set(expected) | {"benchmark_files.json"}:
            raise ValueError("archive contents differ from the published manifest")
        for entry in files:
            name = entry["path"]
            member = members[name]
            if not member.isfile() or member.size != entry["size_bytes"]:
                raise ValueError(f"invalid archive member: {name}")
            destination = checked_path(target, name)
            if valid(destination, entry):
                print(f"Verified, skipping: {name}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            temporary.unlink(missing_ok=True)
            digest = hashlib.sha256()
            try:
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {name}")
                with source, temporary.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                        output.write(block)
                if temporary.stat().st_size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"SHA-256/size verification failed for {name}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Installed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/benchmarks")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--archive-file", type=Path, help="Use a local release package instead of downloading")
    parser.add_argument("--url", help="Override the release URL for an institutional mirror")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or len(manifest.get("files", [])) != 6:
        raise ValueError("invalid benchmark manifest")
    files = manifest["files"]
    target = args.data_root.resolve()
    pending = [entry for entry in files if not valid(checked_path(target, entry["path"]), entry)]
    if not pending:
        print("All six benchmarks are present and SHA-256 verified; no download needed.")
        return
    needed = sum(entry["size_bytes"] for entry in pending)
    if args.archive_file is None:
        needed += manifest["archive"]["size_bytes"]
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    print(f"Need up to {needed / 1024**2:.1f} MiB; free space {free / 1024**2:.1f} MiB.")
    if free < needed:
        raise OSError("insufficient free disk space for benchmark download and extraction")
    temporary = None
    if args.archive_file is None:
        url = args.url or manifest["archive"].get("url")
        if not url:
            raise RuntimeError("benchmark release URL is not published yet; use --archive-file for a local package")
        temporary = target / (manifest["archive"]["asset_name"] + ".partial")
        temporary.unlink(missing_ok=True)
        download(url, temporary, manifest["archive"]["size_bytes"])
        archive = temporary
    else:
        archive = args.archive_file.resolve()
    try:
        if not archive.is_file() or archive.stat().st_size != manifest["archive"]["size_bytes"]:
            raise ValueError("benchmark archive is absent or has the wrong size")
        if sha256(archive) != manifest["archive"]["sha256"]:
            raise ValueError("benchmark archive SHA-256 mismatch; remove it and retry")
        extract_verified(archive, target, files)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if any(not valid(checked_path(target, entry["path"]), entry) for entry in files):
        raise RuntimeError("one or more extracted benchmarks failed final verification")
    print(f"All six benchmarks installed and verified in {target}")


if __name__ == "__main__":
    main()
