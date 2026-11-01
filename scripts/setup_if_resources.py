#!/usr/bin/env python3
"""Provision hash-pinned official NLTK English data once, before offline jobs."""

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rlvr.ifeval import RESOURCE_ROOT, VENDOR, check_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify files without network access"
    )
    args = parser.parse_args()
    resources = json.loads((VENDOR / "UPSTREAM.json").read_text())["resources"]
    for resource in resources:
        if not args.check:
            with urllib.request.urlopen(resource["url"], timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != resource["sha256"]:
                raise ValueError(f"NLTK archive checksum mismatch: {resource['url']}")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for name in resource["files"]:
                    target = RESOURCE_ROOT / name
                    content = archive.read(name.removeprefix("tokenizers/"))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
    check_resources()
    print(f"NLTK English resources verified: {RESOURCE_ROOT}")


if __name__ == "__main__":
    main()
