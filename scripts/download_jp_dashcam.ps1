# Download Japanese dashcam accident videos from YouTube (PowerShell)
# Usage: .\scripts\download_jp_dashcam.ps1

$ErrorActionPreference = "Stop"

$OUTPUT_DIR = "data\jp_dashcam"
$ARCHIVE_FILE = "data\jp_dashcam\downloaded.txt"

New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

Write-Host "Downloading 20 Japanese dashcam accident videos..."
Write-Host "Search query: ドライブレコーダー 事故 まとめ 2025"

yt-dlp `
  --download-archive $ARCHIVE_FILE `
  --ignore-errors `
  --no-overwrites `
  -f "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best" `
  --merge-output-format mp4 `
  -o "$OUTPUT_DIR/%(upload_date)s_%(id)s.%(ext)s" `
  --max-downloads 20 `
  "ytsearch20:ドライブレコーダー 事故 まとめ 2025"

Write-Host ""
Write-Host "Download complete!"
Write-Host "Videos saved to: $OUTPUT_DIR"
$count = (Get-ChildItem -Path "$OUTPUT_DIR\*.mp4").Count
Write-Host "$count videos downloaded."
