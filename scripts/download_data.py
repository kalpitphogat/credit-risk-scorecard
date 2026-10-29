"""Download Lending Club 2007-2011 loan data (LoanStats3a) and verify its checksum.

Lending Club no longer hosts this file, so we use a public GitHub mirror of the
original download and check it against a fixed SHA-256.

Usage:
    python scripts/download_data.py
"""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://raw.githubusercontent.com/jsabrams/default_risk_prediction/"
       "master/LoanStats3a.csv.zip")
SHA256 = "6e53451aca45fdc3265d01ea99e1543e3730c80a53b68cf098ef3db06ee3bc32"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "LoanStats3a.csv.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Checksum mismatch: got {digest}, expected {SHA256}")
    print("Checksum OK")

    with zipfile.ZipFile(zip_path) as z:
        z.extract("LoanStats3a.csv", RAW_DIR)
    print(f"Extracted to {RAW_DIR / 'LoanStats3a.csv'}")


if __name__ == "__main__":
    main()
