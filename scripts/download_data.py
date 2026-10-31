"""Download the original fold1 files without modifying their contents."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "4ad5ba1eb23ade177bf2ccc73ac884ce580b8a69"
BASE_URL = (
    "https://raw.githubusercontent.com/alexbrandsen/Archaeo-NER-data-English/"
    f"{REVISION}/5-folds-test-train-val-split/fold1"
)
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "fold1"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": "https://github.com/alexbrandsen/Archaeo-NER-data-English",
                "revision": REVISION, "files": {}}
    for name in ("train.txt", "val.txt", "test.txt"):
        url = f"{BASE_URL}/{name}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if not content.strip() or content.lstrip().startswith(b"<!DOCTYPE"):
            raise ValueError(f"Unexpected download contents: {url}")
        target = DESTINATION / name
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
        manifest["files"][name] = {
            "url": url, "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        print(f"Downloaded {name}: {len(content):,} bytes")
    (ROOT / "data" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
