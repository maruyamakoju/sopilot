# Insurance MVP - 保険映像レビュー自動化システム

**目的**: 損保ジャパン PoC向けMVP
**開発期間**: 2026年2月16日 〜 2026年5月15日（12週間）

---

## 🎯 プロダクト概要

ドライブレコーダー映像から事故シーンを自動検出し、過失割合と不正リスクを評価するAIシステム。

### 主な機能
```
1. マルチモーダル危険信号マイニング
   ├─ 音声分析（急ブレーキ音、クラクション）
   ├─ オプティカルフロー（急な動き）
   └─ 物体検出（近接車両、歩行者）

2. Video-LLM 推論
   ├─ 事故シーンの自動検出
   ├─ 重大度判定（NONE/LOW/MEDIUM/HIGH）
   └─ 因果推論（なぜ事故が起きたか）

3. 不確実性定量化（Conformal Prediction）
   ├─ 予測区間の出力（例: {MEDIUM, HIGH}）
   ├─ レビュー優先度の自動計算
   └─ 90%信頼度保証

4. 過失割合判定
   ├─ 交通ルールに基づく判定
   ├─ シナリオ別ロジック（追突、出会い頭等）
   └─ 根拠の自動生成

5. 不正請求検知
   ├─ 異常パターン検出
   ├─ 過去事故との類似度分析
   └─ 統計的外れ値検出
```

---

## 📊 導入効果

### 損保ジャパン（想定）
```
処理能力: 8万件/年 → 60万件/年（7.5倍）
人件費削減: 年間3億円
不正削減: 年間4.5億円
合計効果: 年間7.5億円

投資回収期間: 1.1ヶ月
3年間ROI: 1394%
```

---

## 🏗️ アーキテクチャ

```
insurance_mvp/
├─ mining/           # B1: マルチモーダル信号マイニング
│   ├─ audio.py      # 音声分析
│   ├─ motion.py     # オプティカルフロー
│   ├─ proximity.py  # YOLOv8 物体検出
│   └─ fuse.py       # 信号融合
│
├─ cosmos/           # B2: Video-LLM 推論
│   ├─ client.py     # Cosmos/Qwen2.5-VL
│   └─ prompt.py     # 保険特化プロンプト
│
├─ conformal/        # 不確実性定量化（SOPilotから移植）
│   └─ split_conformal.py
│
├─ insurance/        # 保険ドメインロジック
│   ├─ schema.py     # Pydanticモデル
│   ├─ fault_assessment.py  # 過失割合判定
│   └─ fraud_detection.py   # 不正検知
│
├─ review/           # 人間レビューワークフロー
│   ├─ queue.py      # レビューキュー管理
│   └─ workflow.py   # ステータス管理
│
├─ api/              # FastAPI
│   └─ main.py       # REST API
│
├─ templates/        # Web UI
│   ├─ review.html   # レビュー画面
│   └─ queue.html    # キュー一覧
│
└─ pipeline.py       # メインパイプライン
```

---

## 🚀 セットアップ

### 前提条件
```
- Python 3.10+
- NVIDIA GPU（RTX 5090 or A6000推奨）
- CUDA 12.1+
- VRAM 24GB以上
```

### インストール
```bash
# 1. 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 依存パッケージインストール
pip install -e ".[all]"

# 3. モデルダウンロード（初回のみ）
python scripts/download_models.py
```

### 設定
```bash
# config.yaml を編集
cp config.example.yaml config.yaml
nano config.yaml
```

---

## 💻 使い方

### 1. 単一動画の処理
```bash
python -m insurance_mvp.pipeline \
    --video-path data/dashcam001.mp4 \
    --output-dir results/
```

### 2. バッチ処理
```bash
python -m insurance_mvp.pipeline \
    --video-dir data/dashcam/ \
    --output-dir results/ \
    --parallel 4
```

### 3. Web UI起動
```bash
python -m insurance_mvp.api.main

# ブラウザで http://localhost:8000 を開く
```

---

## 🧪 テスト

```bash
# 全テスト実行
pytest tests/ -v

# カバレッジ計測
pytest tests/ --cov=insurance_mvp --cov-report=html
```

---

## 📊 評価

### 精度評価
```bash
python scripts/evaluate_accuracy.py \
    --benchmark data/benchmark_100.jsonl \
    --output results/accuracy_report.json
```

### Conformal Coverage検証
```bash
python scripts/evaluate_conformal.py \
    --benchmark data/benchmark_100.jsonl \
    --alpha 0.1  # 90%信頼度
```

---

## 🛣️ 開発ロードマップ

### Week 1-4: MVP Core
```
✅ マイニングパイプライン
✅ Cosmos推論
✅ 保険特化カスタマイズ（過失割合、不正検知）
```

### Week 5-8: Conformal Prediction + UI
```
⏳ 不確実性定量化統合
⏳ 人間レビュー画面
⏳ レビューワークフロー
```

### Week 9-12: PoC準備
```
⏳ デモ動画作成
⏳ 精度評価レポート
⏳ PoC提案書
```

---

## 📞 サポート

**開発チーム**: SOPilot株式会社
**Email**: [メールアドレス]
**Slack**: [Slackチャンネル]

---

## 📄 ライセンス

Proprietary - 社内利用のみ

---

## 📎 関連資料

- [営業ピッチ資料](../INSURANCE_PITCH.md)
- [開発ロードマップ](../INSURANCE_MVP_ROADMAP.md)
- [ROI計算シート](../INSURANCE_ROI_CALCULATOR.md)
- [技術ホワイトペーパー](docs/WHITEPAPER.md)
