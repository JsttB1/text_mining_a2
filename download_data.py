"""Download the Archaeo-NER fold1 dataset files from GitHub."""

from pathlib import Path
from urllib.request import urlopen

REVISION = "4ad5ba1eb23ade177bf2ccc73ac884ce580b8a69"
BASE_URL = (
    "https://raw.githubusercontent.com/alexbrandsen/Archaeo-NER-data-English/"
    f"{REVISION}/5-folds-test-train-val-split/fold1"
)
DESTINATION = Path(__file__).resolve().parent / "data" / "raw" / "fold1"


def download():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    files = ["train.txt", "val.txt", "test.txt"]

    for filename in files:
        target = DESTINATION / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"Already exists: {target} ({target.stat().st_size:,} bytes)")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {filename} from {url}...")
        with urlopen(url, timeout=60) as response:
            content = response.read()

        target.write_bytes(content)
        print(f"Saved {filename} ({len(content):,} bytes)")


if __name__ == "__main__":
    download()
