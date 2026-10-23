#!/usr/bin/env python3
"""Download Hugging Face repos (models or datasets) into /opt/dlami/nvme/models/<name>.

Usage: fetch_models.py name=org/repo[:dataset][@include_glob,...] ...
Every repo must be on ops/allowlist.txt (the same list ops/guard.py enforces).
Gated repos (401/403) are appended to reports/hf_gated.txt so Amin gets ONE Notion task listing them all.
Status per repo goes to /opt/dlami/nvme/models/_status.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline.keys import get  # noqa: E402

OUT = Path("/opt/dlami/nvme/models")
DATA_OUT = Path("/opt/dlami/nvme/data/raw")


def allowed(repo: str) -> bool:
    for line in (REPO / "ops" / "allowlist.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line.startswith("hf "):
            continue
        pat = line[3:].strip().lower()
        if repo.lower() == pat or (pat.endswith("/*") and repo.lower().startswith(pat[:-1])):
            return True
    return False


def main(specs: list[str]) -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    token = get("HF_TOKEN")
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        name, rest = spec.split("=", 1)
        include = None
        if "@" in rest:
            rest, inc = rest.split("@", 1)
            include = inc.split(",")
        repo_type = "model"
        if rest.endswith(":dataset"):
            rest, repo_type = rest[: -len(":dataset")], "dataset"
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "name": name, "repo": rest, "type": repo_type}
        if not allowed(rest):
            rec["status"] = "not-allowlisted"
            print(json.dumps(rec), flush=True)
            continue
        t0 = time.time()
        try:
            dest = (DATA_OUT if repo_type == "dataset" else OUT) / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            path = snapshot_download(rest, repo_type=repo_type, local_dir=str(dest), token=token,
                                     allow_patterns=include, max_workers=16)
            size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
            rec.update(status="ok", path=path, gb=round(size / 2**30, 2), sec=round(time.time() - t0))
        except GatedRepoError:
            rec["status"] = "gated"
            with open(REPO / "reports" / "hf_gated.txt", "a") as f:
                f.write(rest + "\n")
        except RepositoryNotFoundError:
            rec["status"] = "not-found-or-gated"
            with open(REPO / "reports" / "hf_gated.txt", "a") as f:
                f.write(rest + "  # 404/401: missing or gated\n")
        except Exception as e:  # keep going with the rest
            msg = str(e).replace(token, "***")
            rec.update(status="error", error=msg[:300])
        with open(OUT / "_status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
