#!/usr/bin/env bash
#
# Download the NYU Depth V2 data this project trains and evaluates on:
#   - sync/                 ~50k training RGB-depth pairs (BTS preprocessed, 16-bit mm depth)
#   - official_splits/test/ 654 Eigen test images (extracted from the labeled .mat)
#
# Usage:
#   bash scripts/download_nyu.sh            # downloads into data/nyu_depth_v2
#   DATA_DIR=/path/to/nyu bash scripts/download_nyu.sh
#
# Requirements (pip): gdown, h5py, scipy, numpy, opencv-python, Pillow
#   pip install gdown h5py scipy numpy opencv-python Pillow
#
# Already-downloaded steps are skipped, so the script is safe to re-run.

set -euo pipefail

# Repo root (this script lives in scripts/), captured before we cd into the data dir.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Where everything lands (matches configs/_base/nyu.yaml -> data/nyu_depth_v2).
DATA_DIR="${DATA_DIR:-data/nyu_depth_v2}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BTS_RAW="https://raw.githubusercontent.com/cleinc/bts/master/utils"
SYNC_GDRIVE_ID="1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP"
# Public mirror used by NeWCRFs when the BTS Google Drive link hits quota limits.
SYNC_MIRROR_URL="https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/newcrfs/datasets/nyu/sync.zip"
LABELED_MAT_URL="http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

download_file() {
    local url="$1"
    local out="$2"
    if command -v wget >/dev/null 2>&1; then
        wget -c "$url" -O "$out"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL -C - "$url" -o "$out"
    else
        echo ">> Need wget or curl to download $url" >&2
        exit 1
    fi
}

download_sync_zip() {
    rm -f sync.zip
    echo ">> Trying Google Drive (BTS sync.zip)..."
    if gdown "$SYNC_GDRIVE_ID" -O sync.zip; then
        return 0
    fi

    rm -f sync.zip
    echo ">> Google Drive failed (often a quota limit). Trying public mirror..."
    download_file "$SYNC_MIRROR_URL" sync.zip
}

echo ">> Target directory: $(pwd)"

# 1. Training set: sync.zip (~4–14 GB depending on source) -> sync/
if [ -d "sync" ]; then
    echo ">> [1/3] sync/ already exists, skipping training download."
else
    echo ">> [1/3] Downloading training set (sync.zip)..."
    download_sync_zip
    echo ">> Unzipping sync.zip..."
    unzip -q sync.zip
    rm -f sync.zip
fi

# 2. Labeled .mat (~2.8 GB): source for the official test split.
if [ -f "nyu_depth_v2_labeled.mat" ] || [ -d "official_splits/test" ]; then
    echo ">> [2/3] labeled .mat already present (or test already extracted), skipping."
else
    echo ">> [2/3] Downloading labeled .mat (~2.8 GB)..."
    download_file "$LABELED_MAT_URL" nyu_depth_v2_labeled.mat
fi

# 3. Extract official_splits/test/ (654 images) from the .mat using the BTS helper.
if [ -d "official_splits/test" ]; then
    echo ">> [3/3] official_splits/test/ already exists, skipping extraction."
else
    echo ">> [3/3] Fetching the split-index file (the buggy BTS extractor is replaced by ours)..."
    download_file "$BTS_RAW/splits.mat" splits.mat

    echo ">> Extracting test set from the .mat..."
    # Our scripts/extract_nyu_test.py fixes BTS's numpy-2.x crash and writes test only.
    python "$REPO_ROOT/scripts/extract_nyu_test.py" \
        nyu_depth_v2_labeled.mat splits.mat official_splits

    # The .mat and split file are only needed for this step; remove to save space.
    rm -f nyu_depth_v2_labeled.mat splits.mat
fi

echo ">> Done. Layout under $DATA_DIR:"
echo "     sync/                  (~50k train pairs)"
echo "     official_splits/test/  (654 test images)"
