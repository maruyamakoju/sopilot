# 保険映像レビューMVP：3ヶ月開発ロードマップ

**目標**: 2026年5月15日までに損保ジャパンPoC獲得
**開発期間**: 2026年2月16日 〜 2026年5月15日（12週間）

---

## 📋 マイルストーン

```
Week 1-4  (2/16-3/15):  MVP Core（基本処理パイプライン）
Week 5-8  (3/16-4/12):  Conformal Prediction + 人間レビュー画面
Week 9-12 (4/13-5/10):  PoC準備（デモ動画、提案書、監査ログ）
Week 13   (5/11-5/15):  損保ジャパンへ提案
```

---

## 🎯 Week 1-4: MVP Core

### 目標
```
✅ AutoRisk-RMを保険向けにカスタマイズ
✅ 基本処理パイプライン完成
✅ 1動画を入力 → 危険度判定 → JSON出力
```

### Week 1: 環境構築 + AutoRisk-RM移植
```
Day 1-2: プロジェクト構造作成
├─ insurance_mvp/
│   ├─ mining/          # B1: マイニング（AutoRisk-RMから）
│   ├─ cosmos/          # B2: VLM推論
│   ├─ conformal/       # NEW: SOPilotから移植
│   ├─ review/          # NEW: 人間レビュー
│   ├─ pipeline.py      # メインパイプライン
│   └─ config.py        # 設定管理

Day 3-5: AutoRisk-RMのコア機能を移植
├─ mining/audio.py      # 音声信号分析
├─ mining/motion.py     # オプティカルフロー
├─ mining/proximity.py  # YOLOv8物体検出
├─ mining/fuse.py       # 信号融合
└─ テスト: 1動画で危険ピーク検出が動作

Day 6-7: Cosmos推論の簡略化
├─ cosmos/client.py     # VLM推論（nvidia Cosmos or Qwen2.5-VL）
├─ cosmos/prompt.py     # 保険特化プロンプト
└─ テスト: 1クリップで severity判定が動作
```

### Week 2: 保険特化カスタマイズ
```
Day 1-3: 過失割合判定ロジック
├─ insurance/fault_assessment.py
│   ├─ 過失割合の基本パターン（追突、出会い頭、左折巻き込み等）
│   ├─ 信号情報の考慮
│   └─ 道路状況の考慮（優先道路、一時停止等）

Day 4-5: 不正請求検知
├─ insurance/fraud_detection.py
│   ├─ 異常パターン検出（衝突音なし、損傷不一致等）
│   ├─ 過去事故との類似度
│   └─ 統計的外れ値検出

Day 6-7: スキーマ定義
├─ insurance/schema.py
│   ├─ ClaimAssessment（保険請求評価）
│   │   ├─ severity: NONE/LOW/MEDIUM/HIGH
│   │   ├─ fault_ratio: 0-100%（当事者の過失割合）
│   │   ├─ fraud_risk: 0.0-1.0
│   │   ├─ evidence: list[Evidence]
│   │   └─ recommended_action: APPROVE/REVIEW/REJECT
│   └─ テスト: JSON出力の妥当性確認
```

### Week 3: パイプライン統合
```
Day 1-3: メインパイプライン
├─ pipeline.py
│   ├─ Step 1: 動画取り込み（MP4 → フレーム抽出）
│   ├─ Step 2: B1マイニング（危険ピーク検出）
│   ├─ Step 3: クリップ抽出（±5秒）
│   ├─ Step 4: Cosmos推論（各クリップの評価）
│   ├─ Step 5: 過失割合判定
│   ├─ Step 6: 不正リスク評価
│   └─ Step 7: JSON出力

Day 4-5: バッチ処理対応
├─ 複数動画の並列処理
├─ GPU/CPUリソース管理
└─ エラーハンドリング（動画読み込み失敗等）

Day 6-7: 単体テスト
├─ pytest で各モジュールをテスト
├─ カバレッジ80%以上
└─ CI/CD設定（GitHub Actions）
```

### Week 4: E2Eテスト + ドキュメント
```
Day 1-3: E2Eテスト
├─ 実動画10本で完全な処理を実行
├─ 処理時間計測（目標: 1動画5分以内）
└─ 精度の初期評価（ベースライン）

Day 4-5: 技術ドキュメント
├─ README.md（セットアップ手順）
├─ ARCHITECTURE.md（システム設計）
└─ API.md（JSON入出力仕様）

Day 6-7: Week 1-4の振り返り
├─ 動作デモ（社内レビュー）
├─ 課題リストアップ
└─ Week 5-8の計画調整
```

---

## 🎯 Week 5-8: Conformal Prediction + 人間レビュー

### 目標
```
✅ 不確実性定量化（Conformal Prediction）統合
✅ 人間レビュー画面（簡易版）
✅ レビューワークフロー実装
```

### Week 5: Conformal Prediction移植
```
Day 1-2: SOPilotのコードを移植
├─ conformal/split_conformal.py
│   ├─ SplitConformal クラス
│   ├─ fit(X_calib, y_calib)
│   └─ predict_set(X_test, alpha=0.1)

Day 3-4: 保険データへの適用
├─ 特徴量設計（マイニングスコア + Cosmos出力）
├─ キャリブレーションデータ準備（GT付き100動画）
└─ Conformalモデルのfitting

Day 5-7: パイプライン統合
├─ pipeline.py に Step 8を追加: Conformal Prediction
├─ 出力に prediction_set を追加
│   例: {"severity": "HIGH", "prediction_set": ["MEDIUM", "HIGH"]}
└─ テスト: 予測区間の妥当性確認
```

### Week 6: レビュー優先度付け
```
Day 1-3: 優先度ロジック
├─ review/priority.py
│   ├─ compute_review_priority(assessment, prediction_set)
│   │   ├─ URGENT: HIGH × prediction_set広い
│   │   ├─ STANDARD: MEDIUM
│   │   └─ LOW_PRIORITY: LOW/NONE
│   └─ レビューキュー生成（優先度順ソート）

Day 4-5: ワークフロー管理
├─ review/workflow.py
│   ├─ ReviewQueue（レビュー待ちリスト）
│   ├─ assign_to_reviewer(案件 → 担当者）
│   └─ update_review_status(案件ID, 人間判断)

Day 6-7: データベース設計
├─ SQLite（簡易版）
│   ├─ claims（請求データ）
│   ├─ assessments（AI判定結果）
│   ├─ reviews（人間レビュー結果）
│   └─ audit_log（監査ログ）
```

### Week 7: 人間レビュー画面（Web UI）
```
Day 1-3: FastAPI バックエンド
├─ api/main.py
│   ├─ POST /claims/upload（動画アップロード）
│   ├─ GET /claims/{id}/assessment（AI判定結果取得）
│   ├─ GET /reviews/queue（レビュー待ちリスト）
│   ├─ POST /reviews/{id}/decision（人間判断を記録）
│   └─ GET /reviews/{id}/history（監査ログ）

Day 4-7: フロントエンド（HTML + JavaScript）
├─ templates/
│   ├─ review.html（レビュー画面）
│   │   ├─ 動画プレーヤー（危険クリップ再生）
│   │   ├─ AI判定表示（severity, fault_ratio, fraud_risk）
│   │   ├─ 予測区間表示（Conformal）
│   │   ├─ 人間判断入力（APPROVE/REVIEW/REJECT）
│   │   └─ コメント入力
│   └─ queue.html（レビューキュー一覧）
```

### Week 8: 監査ログ + テスト
```
Day 1-3: 監査ログ機能
├─ audit/logger.py
│   ├─ 全判定履歴の記録（誰が、いつ、何を判断したか）
│   ├─ AI判定 vs 人間判定の比較
│   └─ CSV/JSON エクスポート（金融庁提出用）

Day 4-5: E2Eテスト（UI含む）
├─ playwright でブラウザテスト
├─ 動画アップロード → レビュー → 判断記録の一連のフロー
└─ 処理時間計測

Day 6-7: Week 5-8の振り返り
├─ 動作デモ（社内レビュー）
├─ UIフィードバック反映
└─ Week 9-12の計画調整
```

---

## 🎯 Week 9-12: PoC準備

### 目標
```
✅ デモ動画作成（自動処理の全フロー）
✅ PoC提案書（損保向け）
✅ 精度評価レポート
```

### Week 9: デモ動画作成
```
Day 1-3: デモシナリオ設計
├─ シナリオ1: 追突事故（過失割合判定）
├─ シナリオ2: 不正請求疑い（異常検知）
└─ シナリオ3: 人間レビュー（Conformal活用）

Day 4-7: 画面録画 + 編集
├─ 動画アップロード → AI判定 → レビュー画面の一連の流れ
├─ ナレーション追加（効果説明）
└─ 5分のデモ動画完成
```

### Week 10: 精度評価
```
Day 1-3: GT付きデータセット作成
├─ 過去事故動画100件を手動ラベリング
│   ├─ severity（NONE/LOW/MEDIUM/HIGH）
│   ├─ fault_ratio（0-100%）
│   └─ fraud（True/False）

Day 4-7: 精度評価実施
├─ AI判定 vs GT の比較
├─ 評価指標:
│   ├─ Accuracy（severity分類）
│   ├─ MAE（fault_ratio誤差）
│   ├─ Precision/Recall（fraud検知）
│   └─ Conformal coverage（予測区間の妥当性）
└─ レポート作成（Excel + グラフ）
```

### Week 11: PoC提案書作成
```
Day 1-3: 技術ホワイトペーパー
├─ AutoRisk-RM技術詳細
├─ Conformal Predictionの説明
└─ 保険特化カスタマイズの説明

Day 4-7: PoC提案書
├─ 損保ジャパン向けにカスタマイズ
├─ 精度評価結果を掲載
├─ ROI試算（年間3億円削減）
└─ 導入スケジュール（3-6ヶ月）
```

### Week 12: 最終準備 + リハーサル
```
Day 1-3: プレゼン資料作成
├─ PowerPoint（20-30ページ）
│   ├─ 損保ジャパンの課題
│   ├─ SOPilotの解決策
│   ├─ デモ動画
│   ├─ 精度評価結果
│   ├─ ROI試算
│   └─ 次のステップ

Day 4-5: プレゼンリハーサル
├─ 社内レビュー
├─ Q&A想定集作成
└─ デモ動作確認

Day 6-7: 損保ジャパンへ提案
├─ 面談設定（30分）
├─ プレゼン実施
└─ PoC契約締結
```

---

## 🛠️ 技術スタック

### バックエンド
```
言語: Python 3.10+
フレームワーク: FastAPI 0.100+
DB: SQLite（MVP）→ PostgreSQL（本番）
GPU: NVIDIA RTX 5090 or A6000
```

### AI/ML
```
物体検出: YOLOv8n
VLM: nvidia Cosmos Reason 2 or Qwen2.5-VL-7B
不確実性定量化: SOPilot Conformal Prediction
```

### フロントエンド
```
HTML + TailwindCSS
JavaScript（Vanilla）
動画プレーヤー: Video.js
```

### インフラ
```
開発環境: ローカル（GPU必須）
本番環境: AWS/GCP（後で決定）
CI/CD: GitHub Actions
```

---

## 👥 チーム構成（想定）

```
役割1: テックリード（1名）
- アーキテクチャ設計
- コアロジック実装
- コードレビュー

役割2: MLエンジニア（1名）
- Conformal Prediction統合
- 精度評価
- モデルチューニング

役割3: フルスタック（1名）
- Web UI実装
- API開発
- デプロイ

役割4: ビジネス（1名）
- 営業資料作成
- 損保ジャパンとの調整
- PoC契約交渉
```

---

## 📊 成功指標（KPI）

### 技術指標
```
Week 4終了時:
✅ 1動画処理時間: < 5分
✅ パイプライン動作率: 95%以上

Week 8終了時:
✅ Conformal coverage: 90%以上
✅ UI応答時間: < 2秒

Week 12終了時:
✅ 精度（severity分類）: > 85%
✅ 過失割合誤差（MAE）: < 15%
✅ 不正検知（Recall）: > 80%
```

### ビジネス指標
```
Week 12終了時:
✅ 損保ジャパンへ提案完了
✅ PoC契約獲得（目標）
```

---

## 🚨 リスク管理

### リスク1: Cosmos推論が遅い
```
影響: 1クリップ4.4分 × 20クリップ = 88分
対策:
- バッチサイズ最適化
- GPUメモリ管理改善
- 最悪: Qwen2.5-VL に変更（SOPilot既存実装）
```

### リスク2: 精度が目標未達
```
影響: PoC契約獲得できない
対策:
- Week 10で早期評価
- 精度向上施策:
  ├─ プロンプトチューニング
  ├─ アンサンブル（複数モデル）
  └─ 閾値調整
```

### リスク3: 開発遅延
```
影響: Week 12に間に合わない
対策:
- Week毎のマイルストーン厳守
- 遅延発生時は機能削減
  ├─ 優先度1: コアパイプライン
  ├─ 優先度2: Conformal Prediction
  └─ 優先度3: UI（最悪Jupyter Notebookで代替）
```

---

## 📅 次のアクション（今週）

### Day 1-2（2/16-17）: プロジェクト立ち上げ
```
✅ Gitリポジトリ作成
✅ 開発環境構築
✅ AutoRisk-RMコードのフォーク
```

### Day 3-5（2/18-20）: MVP Core開発開始
```
✅ mining/ モジュール移植
✅ 1動画で危険ピーク検出が動作
```

### Day 6-7（2/21-22）: Cosmos推論実装
```
✅ cosmos/ モジュール実装
✅ 1クリップで severity判定が動作
```

---

**ステータス**: 🚀 **開発開始準備完了**

**次の1手**: プロジェクト構造を作成し、AutoRisk-RMのコア機能を移植
