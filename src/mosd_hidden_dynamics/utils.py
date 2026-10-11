import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(repo):
    repo = Path(repo)
    files = {}
    for pattern in ("src/**/*.py", "scripts/*.py", "configs/*.json", "requirements.txt", "pyproject.toml"):
        for path in sorted(repo.glob(pattern)):
            files[path.relative_to(repo).as_posix()] = sha256_file(path)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    git_commit, git_dirty = None, None
    if (repo / ".git").exists():
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        if commit.returncode == 0:
            git_commit = commit.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        git_dirty = bool(status.stdout.strip())
    return {"sha256": fingerprint, "files": files, "git_commit": git_commit, "git_dirty": git_dirty}


def environment_record():
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
    }
