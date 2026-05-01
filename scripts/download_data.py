"""Download the Credit Card Fraud dataset from Kaggle and verify its integrity."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

DATASET = "mlg-ulb/creditcardfraud"
EXPECTED_SHAPE = (284807, 31)
RAW_DIR = Path("data/raw")


def _check_kaggle_credentials() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "ERROR: ~/.kaggle/kaggle.json not found.\n"
            "Download your API key at https://www.kaggle.com/settings/account\n"
            "then place it at ~/.kaggle/kaggle.json (chmod 600).",
            file=sys.stderr,
        )
        sys.exit(1)


def _download(raw_dir: Path) -> None:
    import kaggle  # noqa: PLC0415

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{DATASET}' to {raw_dir} …")
    kaggle.api.dataset_download_files(DATASET, path=str(raw_dir), unzip=False)

    zip_path = raw_dir / "creditcardfraud.zip"
    print("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()


def _verify(csv_path: Path) -> None:
    print("Verifying dataset integrity …")
    df = pd.read_csv(csv_path)
    if df.shape != EXPECTED_SHAPE:
        print(
            f"ERROR: unexpected shape {df.shape}, expected {EXPECTED_SHAPE}",
            file=sys.stderr,
        )
        sys.exit(1)
    n_fraud = int(df["Class"].sum())
    pct = n_fraud / len(df) * 100
    print(f"OK — {df.shape[0]:,} rows × {df.shape[1]} cols | {n_fraud} frauds ({pct:.3f}%)")


def main() -> None:
    csv_path = RAW_DIR / "creditcard.csv"
    if csv_path.exists():
        print(f"Dataset already present at {csv_path}. Skipping download.")
        _verify(csv_path)
        return

    _check_kaggle_credentials()
    _download(RAW_DIR)
    _verify(csv_path)
    print("Done.")


if __name__ == "__main__":
    main()
