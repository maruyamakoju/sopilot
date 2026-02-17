# Insurance MVP クイックスタート: 実データテスト

**所要時間**: 30分
**目的**: MVPが実際のドライブレコーダー動画で動作することを確認

---

## 🚀 5ステップで開始

### Step 1: 環境セットアップ (5分)

```bash
cd insurance_mvp

# 依存パッケージインストール
pip install -e ".[all]"

# GPU確認
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Step 2: テスト動画入手 (5分)

#### Option A: YouTube から取得
```bash
# youtube-dlインストール
pip install yt-dlp

# ドライブレコーダー動画ダウンロード
yt-dlp -f "best[height<=720]" \
    -o "data/test_videos/collision_001.mp4" \
    "https://www.youtube.com/watch?v=XXXXX"
```

#### Option B: サンプル動画生成
```bash
# 合成動画生成 (開発用)
python scripts/generate_test_video.py \
    --output data/test_videos/synthetic_001.mp4 \
    --duration 300 \
    --scenario collision
```

### Step 3: 設定ファイル作成 (2分)

```bash
# テンプレートをコピー
cp config.example.yaml config.yaml

# 最小限の設定（デフォルトで動作）
cat > config.yaml << EOF
video:
  target_fps: 30
  max_duration_sec: 600

mining:
  audio_weight: 0.3
  motion_weight: 0.4
  proximity_weight: 0.3

cosmos:
  backend: "qwen2.5-vl-7b"
  device: "cuda"

conformal:
  alpha: 0.1
EOF
```

### Step 4: パイプライン実行 (3分)

```bash
# 単一動画処理
python -m insurance_mvp.pipeline \
    --video-path data/test_videos/collision_001.mp4 \
    --output-dir results/test_001/

# 処理完了後、結果を確認
ls results/test_001/
```

**期待出力:**
```
results/test_001/
├── results.json          # AI判定結果
├── report.html           # HTMLレポート
├── checkpoint.json       # 処理チェックポイント
├── danger_clips/         # 抽出された危険クリップ
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
└── keyframes/            # キーフレーム画像
    ├── frame_001.jpg
    └── ...
```

### Step 5: 結果確認 (5分)

#### results.json を確認
```bash
cat results/test_001/results.json | jq .
```

**期待JSON:**
```json
{
  "claim_id": "test_001",
  "video_id": "collision_001",
  "processing_time_sec": 145.2,
  "severity": "HIGH",
  "confidence": 0.92,
  "prediction_set": ["HIGH"],
  "review_priority": "STANDARD",
  "fault_assessment": {
    "fault_ratio": 100.0,
    "scenario_type": "rear_end",
    "reasoning": "Driver failed to maintain safe distance...",
    "applicable_rules": ["道路交通法第26条"]
  },
  "fraud_risk": {
    "risk_score": 0.15,
    "indicators": [],
    "reasoning": "No fraud indicators detected"
  },
  "hazards": [
    {
      "type": "collision",
      "timestamp_sec": 145.2,
      "actors": ["car", "car"],
      "spatial_relation": "front"
    }
  ],
  "evidence": [
    {
      "timestamp_sec": 143.0,
      "description": "Vehicle ahead decelerating",
      "frame_path": "keyframes/frame_143.jpg"
    },
    {
      "timestamp_sec": 145.2,
      "description": "Collision detected",
      "frame_path": "keyframes/frame_145.jpg"
    }
  ]
}
```

#### HTMLレポート確認
```bash
# ブラウザでレポートを開く
open results/test_001/report.html  # macOS
start results/test_001/report.html  # Windows
xdg-open results/test_001/report.html  # Linux
```

---

## ✅ 成功の確認事項

### 処理が成功した場合:
```
✅ results.json が生成された
✅ severity が NONE/LOW/MEDIUM/HIGH のいずれか
✅ confidence が 0.0-1.0 の範囲
✅ fault_ratio が 0-100 の範囲
✅ fraud_risk.risk_score が 0.0-1.0 の範囲
✅ hazards リストに少なくとも1つのイベント
✅ evidence リストに少なくとも1つの証拠
✅ 処理時間が 5分以内
```

### 期待精度:
```
重大度判定: 正解率 85%以上
過失割合: 誤差 10%以内
不正リスク: 精度 80%以上
処理速度: 5分動画 → 2分以内
```

---

## 🐛 トラブルシューティング

### エラー1: CUDA out of memory
```bash
# config.yamlでバッチサイズを削減
cosmos:
  batch_size: 4  # → 2 に変更
  max_frames_per_clip: 120  # → 60 に変更
```

### エラー2: Model not found
```bash
# モデルを再ダウンロード
python scripts/download_models.py

# HuggingFace ログイン（必要な場合）
huggingface-cli login
```

### エラー3: YOLOv8 detection failed
```bash
# YOLOv8モデルを手動ダウンロード
yolo detect predict model=yolov8n.pt source=data/test_videos/collision_001.mp4
```

### エラー4: ffmpeg not found
```bash
# ffmpegインストール
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# Windows
# https://ffmpeg.org/download.html からダウンロード
```

---

## 📊 次のステップ

### 1動画で成功したら:
```bash
# 10動画バッチ処理
python -m insurance_mvp.pipeline \
    --video-dir data/test_videos/ \
    --output-dir results/ \
    --parallel 2
```

### 精度評価:
```bash
# Ground Truthを用意して評価
python scripts/evaluate_accuracy.py \
    --predictions results/all_predictions.json \
    --ground-truths data/ground_truths.json \
    --output results/accuracy_report.json
```

### Web UI起動:
```bash
# FastAPI起動
cd insurance_mvp
uvicorn api.main:app --reload

# ブラウザで http://localhost:8000 を開く
```

---

## 📞 サポート

**問題が解決しない場合:**
1. `results/test_001/pipeline.log` を確認
2. エラーメッセージをコピー
3. GitHub Issuesに投稿またはチームに連絡

**詳細ドキュメント:**
- [TESTING_PLAN.md](TESTING_PLAN.md) - 完全なテスト計画
- [README.md](README.md) - 詳細セットアップ手順
- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - パイプラインの詳細

---

**重要**: このクイックスタートは開発者向けです。損保ジャパンデモ用の手順は別途作成します。
