from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def compute_task_hash(
    task_name: str,
    command: list[str],
    args: list[str],
    packages: list[str],
    env: dict[str, str],
    parent_hashes: dict[str, str] | None = None,
) -> str:
    """Compute a deterministic SHA256 cache key for a task and its upstream lineage."""
    hasher = hashlib.sha256()
    payload = {
        "name": task_name,
        "command": command,
        "args": args,
        "packages": sorted(packages),
        "env": sorted(env.items()),
        "parent_hashes": sorted((parent_hashes or {}).items()),
    }
    hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


class TaskCache:
    """Manages task artifact and parameter memoization on the shared PVC."""

    def __init__(self, workspace_path: str | Path | None = None):
        if workspace_path:
            self.ws = Path(workspace_path)
        else:
            vol = os.environ.get("VOLUME")
            self.ws = Path(vol) if vol else Path("local_pipeline_volume").resolve()
        self.cache_dir = self.ws / ".tkf" / "cache"

    def get_cache_entry(self, cache_key: str) -> dict[str, Any] | None:
        """Return cached task metadata and outputs if cache hit exists."""
        meta_file = self.cache_dir / cache_key / "meta.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text())
                if data.get("status") == "SUCCESS":
                    return data
            except Exception:
                pass
        return None

    def record_success(
        self,
        cache_key: str,
        task_name: str,
        outputs: dict[str, str],
    ) -> None:
        """Record successful task completion in persistent cache."""
        target_dir = self.cache_dir / cache_key
        target_dir.mkdir(parents=True, exist_ok=True)
        meta_file = target_dir / "meta.json"
        data = {
            "task_name": task_name,
            "cache_key": cache_key,
            "status": "SUCCESS",
            "outputs": outputs,
        }
        meta_file.write_text(json.dumps(data, indent=2))
