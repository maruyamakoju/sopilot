#!/usr/bin/env bash
# Download Japanese dashcam accident videos from YouTube
# Usage: bash scripts/download_jp_dashcam.sh

set -euo pipefail

OUTPUT_DIR="data/jp_dashcam"
ARCHIVE_FILE="data/jp_dashcam/downloaded.txt"

mkdir -p "$OUTPUT_DIR"

echo "Downloading 20 Japanese dashcam accident videos..."
echo "Search query: ドライブレコーダー 事故 まとめ 2025"

yt-dlp \
  --download-archive "$ARCHIVE_FILE" \
  --ignore-errors \
  --no-overwrites \
  -f "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best" \
  --merge-output-format mp4 \
  -o "$OUTPUT_DIR/%(upload_date)s_%(id)s.%(ext)s" \
  --max-downloads 20 \
  "ytsearch20:ドライブレコーダー 事故 まとめ 2025"

echo ""
echo "Download complete!"
echo "Videos saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.mp4 | wc -l
echo "videos downloaded."
