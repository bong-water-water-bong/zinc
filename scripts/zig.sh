#!/bin/sh
# Run the pinned compiler without replacing an installed system compiler.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
VERSION="$(cat "$ROOT/.zig-version")"
if command -v zig >/dev/null 2>&1 && [ "$(zig version)" = "$VERSION" ]; then
    exec zig "$@"
fi
CACHE="${ZINC_TOOLCHAIN_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/zinc-toolchains}"
TOOLCHAIN="$CACHE/zig-$VERSION"
if [ ! -x "$TOOLCHAIN/zig" ]; then
    python3 - "$VERSION" "$TOOLCHAIN" <<'PY'
import hashlib
import json
import pathlib
import platform
import sys
import tarfile
import tempfile
import urllib.request

version, destination = sys.argv[1], pathlib.Path(sys.argv[2])
arch = {"arm64": "aarch64", "AMD64": "x86_64"}.get(platform.machine(), platform.machine())
system = {"Darwin": "macos", "Linux": "linux"}.get(platform.system())
if system is None:
    raise SystemExit("Install the pinned Zig release manually on this platform.")
with urllib.request.urlopen("https://ziglang.org/download/index.json") as response:
    info = json.load(response)[version][f"{arch}-{system}"]
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
    temporary = pathlib.Path(temporary)
    archive = temporary / "zig.tar.xz"
    print(f"Downloading Zig {version} for {arch}-{system}...", file=sys.stderr)
    urllib.request.urlretrieve(info["tarball"], archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != info["shasum"]:
        raise SystemExit("Zig archive SHA256 does not match the official release index.")
    with tarfile.open(archive) as contents:
        # Python before 3.12 does not support tarfile's data extraction filter.
        if hasattr(tarfile, "data_filter"):
            contents.extractall(temporary, filter="data")
        else:
            for member in contents.getmembers():
                target = (temporary / member.name).resolve()
                if temporary.resolve() not in target.parents or member.issym() or member.islnk():
                    raise SystemExit("Unsafe toolchain archive member.")
            contents.extractall(temporary)
    extracted = next(path for path in temporary.iterdir() if path.is_dir())
    try:
        extracted.rename(destination)
    except OSError:
        if not (destination / "zig").is_file():
            raise
PY
fi
exec "$TOOLCHAIN/zig" "$@"
