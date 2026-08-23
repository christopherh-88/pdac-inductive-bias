#!/usr/bin/env bash
# Download the PANORAMA public training and development dataset (4 Zenodo batches)
# plus the labels repository. Usage: bash scripts/download/download_panorama.sh /path/to/data
#
# Batch record IDs are taken from the official datasets page:
# https://panorama.grand-challenge.org/datasets-imaging-labels/
# License: CC BY-NC 4.0. ~180 GB zipped; ensure ~400 GB free.
set -euo pipefail

DATA_ROOT="${1:?Usage: download_panorama.sh /path/to/data}"
mkdir -p "$DATA_ROOT/panorama/images" "$DATA_ROOT/panorama/zips"

declare -A BATCHES=(
  [batch_1]=13715870
  [batch_2]=13742336
  [batch_3]=11034011
  [batch_4]=10999754
)

for batch in batch_1 batch_2 batch_3 batch_4; do
  rec="${BATCHES[$batch]}"
  dir="$DATA_ROOT/panorama/zips/$batch"
  if [ -f "$dir/.done" ]; then
    echo ">> $batch already downloaded, skipping"
    continue
  fi
  echo ">> Downloading $batch (Zenodo record $rec) ..."
  mkdir -p "$dir"
  # zenodo_get resolves the record, downloads all files, and verifies md5 checksums
  zenodo_get -o "$dir" "$rec"
  touch "$dir/.done"
done

echo ">> Extracting ..."
for batch in batch_1 batch_2 batch_3 batch_4; do
  for z in "$DATA_ROOT/panorama/zips/$batch"/*.zip; do
    [ -e "$z" ] || continue
    unzip -n -q "$z" -d "$DATA_ROOT/panorama/images"
  done
done

echo ">> Cloning labels (manual_labels/, automatic_labels/, clinical_information.xlsx) ..."
if [ -d "$DATA_ROOT/panorama/panorama_labels/.git" ]; then
  git -C "$DATA_ROOT/panorama/panorama_labels" pull --ff-only
else
  # Labels repo uses git-lfs for the segmentation files
  command -v git-lfs >/dev/null || echo "WARNING: git-lfs not found; label files may be pointers only"
  git clone https://github.com/DIAGNijmegen/panorama_labels "$DATA_ROOT/panorama/panorama_labels"
fi
# Record the labels commit — the repo is live and may be updated upstream; the study uses one frozen commit.
git -C "$DATA_ROOT/panorama/panorama_labels" rev-parse HEAD > "$DATA_ROOT/panorama/labels_commit.txt"
echo ">> Labels commit (freeze this in config/frozen_thresholds.yaml): $(cat "$DATA_ROOT/panorama/labels_commit.txt")"

echo ">> Done. Images: $DATA_ROOT/panorama/images  Labels: $DATA_ROOT/panorama/panorama_labels"
