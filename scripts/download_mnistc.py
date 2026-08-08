"""Fetch MNIST-C (Mu & Gilmer, 2019) from Zenodo record 3239543.

    python scripts/download_mnistc.py            # download, verify, extract
    python scripts/download_mnistc.py --check    # report what is already present

Downloads mnist_c.zip (~342 MB) to ./data and extracts to ./data/MNIST-C.
Both are gitignored. Source: https://zenodo.org/record/3239543
Corruption generation code: https://github.com/google-research/mnist-c
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import MNIST_C_CORRUPTIONS

URL = "https://zenodo.org/record/3239543/files/mnist_c.zip"
ARCHIVE = ROOT / "data" / "mnist_c.zip"
DEST = ROOT / "data" / "MNIST-C"


def report(dest=DEST):
    base = dest / "mnist_c" if (dest / "mnist_c").exists() else dest
    if not base.exists():
        print(f"not present: {dest}")
        return False
    missing = [c for c in MNIST_C_CORRUPTIONS if not (base / c / "test_images.npy").exists()]
    found = len(MNIST_C_CORRUPTIONS) - len(missing)
    print(f"{found}/{len(MNIST_C_CORRUPTIONS)} corruptions present at {base}")
    if missing:
        print("missing:", ", ".join(missing))
    return not missing


def _progress(blocks, block_size, total):
    if total <= 0:
        return
    pct = min(100.0, blocks * block_size * 100.0 / total)
    mb = blocks * block_size / 1e6
    print(f"\r  {pct:5.1f}%  {mb:7.1f} / {total/1e6:.1f} MB", end="", flush=True)


def download():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        print(f"archive already present: {ARCHIVE} ({ARCHIVE.stat().st_size/1e6:.1f} MB)")
        return
    print(f"downloading {URL}")
    tmp = ARCHIVE.with_suffix(".zip.part")
    urllib.request.urlretrieve(URL, tmp, reporthook=_progress)
    print()
    tmp.replace(ARCHIVE)
    print(f"saved {ARCHIVE} ({ARCHIVE.stat().st_size/1e6:.1f} MB)")
    print("sha256:", hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())


def extract():
    print(f"extracting to {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(DEST)
    # Flatten the mnist_c/ wrapper directory if the archive has one.
    inner = DEST / "mnist_c"
    if inner.exists():
        for child in inner.iterdir():
            target = DEST / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
        inner.rmdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report presence and exit")
    ap.add_argument("--keep-archive", action="store_true", help="do not delete mnist_c.zip after extraction")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if report() else 1)
    if report():
        print("nothing to do")
        return

    download()
    extract()
    if not args.keep_archive and ARCHIVE.exists():
        ARCHIVE.unlink()
        print("removed archive")
    if not report():
        sys.exit("extraction incomplete")
    print("\nMNIST-C ready")


if __name__ == "__main__":
    main()
