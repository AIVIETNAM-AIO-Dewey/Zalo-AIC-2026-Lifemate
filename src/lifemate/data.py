"""Shared data access; app state is accessed exclusively by zaic_tools."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import unicodedata


def read_rows(root, split):
    path = Path(root) / split / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate sample IDs: {path}")
    return rows


def select_rows(rows, ids_path=None, limit=None):
    if ids_path:
        ids = set(json.loads(Path(ids_path).read_text(encoding="utf-8")))
        missing = ids - {r["id"] for r in rows}
        if missing:
            raise ValueError(f"IDs not found: {sorted(missing)[:5]}")
        rows = [r for r in rows if r["id"] in ids]
    return rows[:limit] if limit is not None else rows


def audio_turns(row):
    """History first, then latest user request; one entry per spoken turn."""
    return [h for h in row.get("history", []) if h.get("role") == "user"] + [row]


def audio_path(root, split, relative):
    base = (Path(root) / split).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base / "audio"):
        raise ValueError(f"Audio must live under {base / 'audio'}: {relative}")
    return path


def load_runtime(root, split):
    """Load the package shipped with THIS split. Never mix runtime versions."""
    package = (Path(root) / split / "zaic_tools").resolve()
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing organisers' runtime: {package}")
    existing = sys.modules.get("zaic_tools")
    if existing and Path(existing.__file__).resolve().parent != package:
        raise RuntimeError("Start a fresh process to use another split's zaic_tools")
    sys.path.insert(0, str(package.parent))
    return importlib.import_module("zaic_tools")


def load_catalog(root, split):
    path = Path(root) / split / f"{split}_tools_catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))


def catalog_map(catalog):
    return {t["name"]: t for a in catalog["apps"] for t in a["tools"]}


def category(row):
    calls = row["label"]["tool_calls"]
    if not calls:
        return "empty"
    return "missing_info" if any(c["name"] == "sys.ask_missing_info" for c in calls) else "execute"


def make_split(rows, seed=42, dev_fraction=0.2):
    """Group by identical normalised latest request; all history stays with its row.

    This is NOT a speaker-disjoint or template-disjoint split: neither speaker
    identities nor reliable template annotations are provided in the JSONL.
    """
    groups = {}
    for row in rows:
        text = " ".join(unicodedata.normalize("NFC", row["audio_text"]).lower().split())
        key = hashlib.sha256(text.encode()).hexdigest()
        groups.setdefault(key, []).append(row["id"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    dev_keys = set(keys[:max(1, round(len(keys) * dev_fraction))])
    dev_ids = sorted(i for k, ids in groups.items() if k in dev_keys for i in ids)
    train_ids = sorted(i for k, ids in groups.items() if k not in dev_keys for i in ids)
    return train_ids, dev_ids


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
