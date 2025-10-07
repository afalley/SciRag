#!/usr/bin/env bash
# gather_pdfs_flat.sh — recursively copy only PDFs to a flat destination folder

set -euo pipefail




SRC="/Users/andreasfalley/Downloads/pdfdata"
DEST="/Users/andreasfalley/PycharmProjects/SciRag/data/pdfs"

# Resolve absolute, physical paths
[[ -d "$SRC" ]] || { echo "Not a directory: $SRC" >&2; exit 3; }
SRC_ABS="$(cd "$SRC" && pwd -P)"
mkdir -p "$DEST"
DEST_ABS="$(cd "$DEST" && pwd -P)"

# Safety: refuse if DEST is inside SRC (would recurse on itself)
case "$DEST_ABS/" in
  "$SRC_ABS/"*)
    echo "Refusing: DEST ($DEST_ABS) is inside SRC ($SRC_ABS)" >&2
    exit 4
    ;;
esac

count=0
# Find PDFs (case-insensitive), copy to DEST, avoid name collisions
find "$SRC_ABS" -type f -iname '*.pdf' -print0 |
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  name="${base%.*}"
  ext="${base##*.}"
  out="$DEST_ABS/$base"
  i=1
  while [[ -e "$out" ]]; do
    out="$DEST_ABS/${name}_$i.$ext"
    ((i++))
  done
  cp -p "$f" "$out"
  ((count++))
done

echo "Copied $count PDF file(s) to: $DEST_ABS"
