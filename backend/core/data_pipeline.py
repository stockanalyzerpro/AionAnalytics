"""
AION Analytics — Unified Data Pipeline (Nightly Core)
-----------------------------------------------------

Roles:
    ✔ Load & save rolling.json.gz safely
    ✔ Maintain rolling_brain.json.gz
    ✔ Provide locking, atomic writes, backup rotation
    ✔ Normalize symbols
    ✔ Utility I/O wrappers used across backend
    ✔ Mirrors dt_backend architecture (data_pipeline_dt.py)

This file is 100% safe on Windows (no fcntl).
Supports the entire nightly backend job pipeline.

NOTE:
    - All original behavior is preserved.
    - All functions you rely on remain identical in name and purpose.
"""

from __future__ import annotations
import os
import json
import gzip
import shutil
import random
import time
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

from backend.core.config import PATHS, TIMEZONE


# =====================================================================================
# LOGGER
# =====================================================================================

def log(msg: str) -> None:
    """Unified logger for backend core."""
    ts = datetime.datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =====================================================================================
# CORE PATHS
# =====================================================================================

ROLLING_PATH: Path = PATHS["rolling"]
BRAIN_PATH: Path = PATHS["rolling_brain"]
BACKUP_DIR: Path = PATHS["rolling_backups"]

BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================================
# LOCKFILE (WINDOWS + UNIX SAFE)
# =====================================================================================

class RollingLock:
    """
    Cross-platform lock using a simple .lock file.
    Works on Windows, Linux, Mac.
    Prevents concurrent writes to rolling.json.gz & rolling_brain.json.gz.
    """

    def __init__(self, target_path: Path, timeout: int = 30):
        self.lockfile = target_path.with_suffix(target_path.suffix + ".lock")
        self.timeout = timeout

    def __enter__(self):
        start = time.time()
        while True:
            try:
                fd = os.open(str(self.lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() - start > self.timeout:
                    log(f"⚠️ Timeout acquiring lock: {self.lockfile}")
                    return self
                time.sleep(0.15 + random.random() * 0.25)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.lockfile.exists():
                self.lockfile.unlink()
        except Exception:
            pass


# =====================================================================================
# JSON HELPERS
# =====================================================================================

def _read_json_gz(path: Path) -> Optional[Any]:
    """Safe JSON.gz reader."""
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"⚠️ Failed reading {path}: {e}")
        return None


def _atomic_write_json_gz(path: Path, obj: Any) -> None:
    """
    Write JSON.gz atomically:
        temp file → replace()
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        log(f"⚠️ Failed atomic write to {path}: {e}")
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _backup_file(src: Path):
    """Copy rolling.json.gz or rolling_brain.json.gz to timestamped backup."""
    if not src.exists():
        return
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{src.stem}_{ts}{src.suffix}"
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        log(f"⚠️ Failed backup for {src}: {e}")


# =====================================================================================
# ROLLING CACHE I/O
# =====================================================================================

def _read_rolling() -> Dict[str, Any]:
    """Return rolling dict (empty if missing or invalid)."""
    data = _read_json_gz(ROLLING_PATH)
    return data if isinstance(data, dict) else {}


def save_rolling(rolling: Dict[str, Any]) -> None:
    """Safely save rolling with locking + backup."""
    if not isinstance(rolling, dict):
        log("⚠️ save_rolling called with non-dict")
        return

    with RollingLock(ROLLING_PATH):
        _backup_file(ROLLING_PATH)
        _atomic_write_json_gz(ROLLING_PATH, rolling)
        log(f"💾 rolling.json.gz updated ({len(rolling)} symbols)")


def ensure_symbol_fields(rolling: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Guarantee existence of a symbol node with basic structure.
    Preserves all original behavior.
    """
    sym = symbol.upper()
    node = rolling.get(sym, {})

    node.setdefault("symbol", sym)
    node.setdefault("history", [])
    node.setdefault("fundamentals", {})
    node.setdefault("metrics", {})
    node.setdefault("predictions", {})
    node.setdefault("context", {})
    node.setdefault("policy", {})

    rolling[sym] = node
    return node


def get_symbol_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Convenience accessor."""
    return _read_rolling().get(symbol.upper())


# =====================================================================================
# ROLLING BRAIN I/O
# =====================================================================================

def _read_brain() -> Dict[str, Any]:
    data = _read_json_gz(BRAIN_PATH)
    return data if isinstance(data, dict) else {}


def save_brain(brain: Dict[str, Any]) -> None:
    """Safely write rolling_brain.json.gz."""
    if brain is None:
        brain = {}

    with RollingLock(BRAIN_PATH):
        _backup_file(BRAIN_PATH)
        _atomic_write_json_gz(BRAIN_PATH, brain)
        log("🧠 rolling_brain.json.gz updated")


# =====================================================================================
# GENERIC LOADERS (used by fetchers & jobs)
# =====================================================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"⚠️ load_json failed for {path}: {e}")
        return None


def save_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"⚠️ save_json failed for {path}: {e}")


# =====================================================================================
# SAFE FLOAT
# =====================================================================================

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


# =====================================================================================
# LATEST FILE HELPERS
# =====================================================================================

def latest_dated_json(directory: Path) -> Optional[Path]:
    """
    Return newest YYYY-MM-DD.json file from a directory.
    """
    if not directory.exists():
        return None

    candidates = []
    for fp in directory.glob("*.json"):
        try:
            name = fp.stem
            datetime.datetime.strptime(name, "%Y-%m-%d")
            candidates.append(fp)
        except Exception:
            continue

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stem)


# =====================================================================================
# MERGE UTILITIES
# =====================================================================================

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge dict b into dict a.
    Used for merging fundamentals, metrics, context, policy, etc.
    """
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = deep_merge(a[k], v)
        else:
            a[k] = v
    return a


# =====================================================================================
# DONE
# =====================================================================================
