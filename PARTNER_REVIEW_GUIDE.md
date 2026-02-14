# SOPilot — Partner Review Guide

## 🎯 見る順番（3分で全体把握）

### Step 1: 技術の主張を確認（30秒）
**README冒頭の3つの図**を開く:
- https://github.com/maruyamakoju/sopilot#readme

**確認ポイント**:
1. **Training works**: 1.7 → 81.5 (+79.9 points, 100% success)
2. **Soft-DTW superior**: 43000× discrimination vs Cosine 5.9×
3. **E2E pipeline**: 10-panel complete architecture

### Step 2: 成果物をダウンロード（1分）
**GitHub Release v1.0.0**:
- https://github.com/maruyamakoju/sopilot/releases/tag/v1.0.0
- Windows: `demo_outputs_v1.0.zip` (2.1 MB)
- Linux/Mac: `demo_outputs_v1.0.tar.gz` (2.1 MB)

**中身**:
- 12 PNG figures (200 DPI, publication-quality)
- 2 JSON summaries (training_summary.json, ablation_summary.json)

### Step 3: ローカル再現（オプション、2分）
```bash
git clone https://github.com/maruyamakoju/sopilot.git
cd sopilot
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev,vigil]"
python scripts/run_demo_suite.py --quick  # 2 minutes
```

**Output**: `demo_outputs/` with 12 figures

---

## 📋 Review Questions（3つに絞る）

### Q1: 技術の主張はクリアか？
**観点**: READMEの3つの図（30秒で見る）で、以下が理解できるか？

- [ ] Neural training が機能している証拠（1.7 → 81.5）
- [ ] Soft-DTWが既存手法より優れている根拠（43000× discrimination）
- [ ] システム全体のアーキテクチャ（E2E 10-panel）

**フィードバック**:
- クリアに理解できた部分:
- 不明確だった部分:
- 追加で見たい証拠:

---

### Q2: 次に検証すべき"実データ"は？
**観点**: 現在は合成データで証明済み。実用化に向けて、どの実データで検証すべきか？

**候補**:
- [ ] 工場メンテナンス手順（保全作業）
- [ ] 医療プロトコル（手術/処置手順）
- [ ] 訓練/教育ビデオ（資格試験など）
- [ ] その他（具体的に: ___________）

**フィードバック**:
- 最優先の実データ領域:
- データ入手の難易度（簡単/普通/困難）:
- 期待される価値（なぜそのデータが重要か）:

---

### Q3: 価値が最大になるプロダクト形態は？
**観点**: 技術は証明済み。次にどの形で価値を出すか？

**候補**:
- [ ] **REST API** — 他システムと連携（SOP評価API、VIGIL検索API）
- [ ] **Web UI** — ブラウザで触れるデモ（Streamlit等）
- [ ] **レポート生成** — PDFで評価レポート自動生成
- [ ] **SaaS** — クラウドでホスト、顧客が直接利用
- [ ] **SDK/Library** — Pythonパッケージで配布
- [ ] その他（具体的に: ___________）

**フィードバック**:
- 最優先のプロダクト形態:
- 理由（どのユーザーにどう届けたいか）:
- 次のマイルストーン（何ヶ月で何を目指すか）:

---

## ✅ Review完了後のアクション

**フィードバックを受けて次に実装するもの**:
1. Q2の答え → 実データ検証パイプライン構築
2. Q3の答え → プロダクト形態に応じた開発（UI/API/SaaS）
3. その他要求 → 優先順位付けして実装

---

## 📚 Deep Dive（詳細を知りたい場合）

### 技術詳細
- **ACCOMPLISHMENTS.md**: 開発サマリー全体
- **README.md**: システム概要、研究背景、ベンチマーク結果
- **DEMO_PACK_READY.md**: Demo Pack準備状況

### コード詳細
- **Neural Pipeline**: `src/sopilot/nn/` (Soft-DTW, DILATE, Conformal)
- **VIGIL-RAG**: `src/sopilot/rag_service.py`, `src/sopilot/event_detection_service.py`
- **Tests**: 871 tests in `tests/` (100% passing)

### Demo Scripts
- `scripts/demo_neural_pipeline.py` — 6 figures
- `scripts/demo_ablation_study.py` — 5 figures + JSON
- `scripts/demo_e2e_pipeline.py` — 10-panel figure
- `scripts/demo_training_convergence.py` — 8-panel + JSON
- `scripts/run_demo_suite.py` — One-command runner

---

**Prepared by**: Claude Opus 4.6
**Date**: 2026-02-15
**Status**: Ready for partner review
**Repository**: https://github.com/maruyamakoju/sopilot
