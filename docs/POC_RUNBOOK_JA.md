# Insurance MVP PoC Runbook

## 概要

Insurance MVP PoC デモの実行手順書です。3つのプラン（A/B/C）を用意しており、環境に応じて切り替えられます。

| プラン | 必要環境 | 推論 | 所要時間 |
|--------|----------|------|----------|
| **Plan A** (Real) | NVIDIA GPU (14GB+ VRAM) | Qwen2.5-VL-7B-Instruct | ~3分/動画 |
| **Plan B** (Replay) | CPU のみ | 事前計算済みJSON再生 | 即時 |
| **Plan C** (Mock) | CPU のみ | ルールベース即時応答 | <1秒 |

---

## 事前準備

### 共通

```bash
# リポジトリクローン
git clone https://github.com/maruyamakoju/sopilot.git
cd sopilot

# Docker 確認
docker --version        # Docker 24+ 推奨
docker compose version  # Docker Compose v2+
```

### Plan A のみ（GPU推論）

```bash
# GPU 確認
nvidia-smi  # VRAM 14GB+ 必要

# GPU チェックスクリプト
python scripts/insurance_gpu_check.py
```

---

## Plan A: Real GPU 推論

**対象**: NVIDIA GPU (RTX 3090/4090/5090 等、14GB+ VRAM) がある環境

### 起動

```bash
# Linux / Mac
./scripts/poc_up.sh --gpu

# Windows (PowerShell)
.\scripts\poc_up.ps1 -Gpu
```

### デモ手順

1. ブラウザで http://localhost:8501 を開く
2. サイドバーで **Language → 日本語** を選択
3. サイドバーの **Backend** で **Real (Qwen2.5-VL-7B)** を選択
   - GPU情報が表示されることを確認
4. **入力モード → デモ動画** を選択
5. 3つのデモ動画を順に実行:

| 動画 | 期待結果 | 所要時間 |
|------|----------|----------|
| `collision.mp4` | 重大度: **HIGH** (🔴), 過失: 70-100% | ~2.5分 |
| `normal.mp4` | 重大度: **NONE** (🟢), 過失: 0% | ~2.5分 |
| `near_miss.mp4` | 重大度: **LOW~MEDIUM** (🔵🟡), 過失: 50% | ~2.5分 |

6. 結果画面で以下を確認:
   - 重大度バッジと信頼度
   - 過失割合と推論理由
   - 不正リスクスコア
   - Conformal Prediction（予測集合）
   - 因果推論（AIの判断根拠）

### 停止

```bash
./scripts/poc_up.sh --down
# or
.\scripts\poc_up.ps1 -Down
```

---

## Plan B: Replay モード（推奨フォールバック）

**対象**: GPUが無い環境、またはPlan Aが動作しない場合

RTX 5090 で事前に実行した実測結果（JSON）を再生します。推論は行いません。

### 起動

```bash
# Linux / Mac
./scripts/poc_up.sh

# Windows (PowerShell)
.\scripts\poc_up.ps1
```

### デモ手順

1. ブラウザで http://localhost:8501 を開く
2. サイドバーで **Language → 日本語** を選択
3. サイドバーの **Backend** で **Replay (実測JSON)** を選択
4. **ファイル選択** で `real_benchmark_rtx5090.json` を選択
   - このファイルは `insurance_mvp/demo_assets/` に同梱済み
5. 動画を選択して結果を表示:

| 動画 | 実測結果 (RTX 5090) | 備考 |
|------|---------------------|------|
| collision | HIGH (信頼度 0.95) | 正解 ✅ |
| normal | NONE (信頼度 0.95) | 正解 ✅ |
| near_miss | LOW (信頼度 0.90) | ボーダーライン 🟡 (期待: MEDIUM) |

6. **比較タブ** で精度を確認:
   - 重大度一致率: 66.7% (2/3 exact, near_miss は1段階差)
   - near_miss は「🟡 ボーダーライン」表示（90% CI でカバー）
   - 推論時間: 平均 151秒/動画

### 補足: メタデータ表示

Replay モードでは以下のメタデータを表示:
- モデル名: Qwen2.5-VL-7B-Instruct
- GPU: RTX 5090 (31.8 GB VRAM)
- 総推論時間: 453.78秒
- ベンチマーク日時

### 停止

```bash
./scripts/poc_up.sh --down
```

---

## Plan C: Mock モード

**対象**: 最小構成の動作確認、またはUIデモのみ

ルールベースのモック推論を使用。VLM は使いません。

### 起動

```bash
# Docker なしでも起動可能
pip install streamlit numpy opencv-python-headless pydantic scikit-learn jinja2 pillow
streamlit run insurance_mvp/dashboard.py

# Docker 経由
./scripts/poc_up.sh
```

### デモ手順

1. ブラウザで http://localhost:8501 を開く
2. サイドバーの **Backend** で **Mock** を選択
3. デモ動画を選択して実行
   - 即座に結果が表示される（<1秒）
   - 結果はルールベースの固定値

### 注意

- Mock の結果は実際の VLM 推論ではありません
- パイプライン構造・UI の確認用途に限定

---

## トラブルシューティング

### Docker ビルドが失敗する

```bash
# キャッシュクリアして再ビルド
docker compose -f docker-compose.poc.yml build --no-cache
```

### ダッシュボードが起動しない

```bash
# ログ確認
docker compose -f docker-compose.poc.yml logs dashboard

# 直接起動で確認
pip install streamlit
streamlit run insurance_mvp/dashboard.py
```

### Plan A: GPU が認識されない

```bash
# NVIDIA Container Toolkit 確認
nvidia-smi                    # ドライバー確認
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi  # Docker GPU 確認

# GPU チェック
python scripts/insurance_gpu_check.py
```

### Plan A: モデルのダウンロードに時間がかかる

初回起動時、Qwen2.5-VL-7B (~15GB) のダウンロードに10-30分かかります。
2回目以降はキャッシュから即座にロード (~14秒) します。

### Plan B: Replay JSON が見つからない

`insurance_mvp/demo_assets/real_benchmark_rtx5090.json` がリポジトリに同梱されています。
ファイルが無い場合は `git pull` で最新を取得してください。

---

## 精度サマリー（PoC 説明用）

### VLM 重大度分類 (Qwen2.5-VL-7B, RTX 5090)

| 指標 | 値 |
|------|-----|
| 重大度一致率 | 66.7% (2/3) |
| 平均距離 | 0.33 (4段階) |
| 推論時間 | 151秒/動画 (48フレーム) |

### 過失割合エンジン（ルールベース）

| 指標 | 値 |
|------|-----|
| テストケース | 25件 |
| 完全一致率 | 100% |
| MAE | 0.0% |

### 不正検知エンジン

| 指標 | 値 |
|------|-----|
| テストケース | 18件 |
| 正解率 | 100% |
| スコア分離度 | 0.59 |

### Conformal Prediction

| 指標 | 値 |
|------|-----|
| 目標カバレッジ | 90% |
| 実測カバレッジ | 100% |
| 平均予測集合サイズ | 1.0 |

---

## ファイル構成

```
sopilot/
├── docker-compose.poc.yml      # PoC用 Docker Compose
├── docker/Dockerfile.poc       # PoC用 Dockerfile
├── .env.poc.example            # 環境変数テンプレート
├── scripts/
│   ├── poc_up.sh               # Linux/Mac 起動スクリプト
│   └── poc_up.ps1              # Windows 起動スクリプト
├── insurance_mvp/
│   ├── dashboard.py            # Streamlit ダッシュボード
│   ├── demo_assets/
│   │   └── real_benchmark_rtx5090.json  # RTX 5090 実測結果
│   └── ...
├── data/dashcam_demo/          # デモ動画 (collision/near_miss/normal)
└── reports/                    # 出力先
```
