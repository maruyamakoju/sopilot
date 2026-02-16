# 商用パイプライン準備完了

**日付**: 2026-02-16
**ステータス**: ✅ **パートナーデータ送付準備完了（運用確定）**

---

## 完了した「次の3タスク」

### ✅ 1️⃣ 最新コミットをpush（リモート完全同期）

**実施内容**:
- 005b4e1まで既にpush済み確認
- eb98848をコミット＋push完了（6ファイル追加）

**Git状況**:
```
eb98848 (HEAD -> master, origin/master) feat: Commercial readiness - Partner data pipeline
005b4e1 docs: Add P0 completion summary (all 6 actions complete)
f39de3f docs: Update COMMERCIAL_READINESS to reflect P0 fix completion
```

**結果**: ✅ リモート完全同期、バックアップ確保

---

### ✅ 2️⃣ パートナーへのデータ提供依頼「送付＋回収オペレーション」確定

**成果物**: `PARTNER_DATA_SENDING_PROCEDURE.md`

**内容**:
1. **送付メッセージ（コピペ用）**
   - 件名・本文完成
   - 添付: PARTNER_DATA_REQUEST.md
   - 納期明示: データ提供から1週間でレポート返却

2. **受け取り方法の指定**（相手が迷わない）
   - Google Drive / Box（推奨）
   - S3バケット（大容量向け）
   - 暗号化zip（セキュリティ重視）

3. **ファイル命名規約**
   ```
   {SOP名}_{役割}_{日付}.mp4
   例: oilchange_gold_202602.mp4
       oilchange_trainee1_202602.mp4
   ```

4. **受領後チェックリスト**
   - 受領後30分: 検品（validate_partner_videos.py）
   - 受領後24時間: indexing + chunk一覧生成
   - 受領後1週間: 全評価完了 + レポート送付

**次のアクション**: 📧 **このメッセージをパートナーに送付**

---

### ✅ 3️⃣ 受領データの「検品→評価→レポート」運用整備

#### 3-1. ✅ 受領データ検品スクリプト作成

**ファイル**: `scripts/validate_partner_videos.py`

**機能**:
- 解像度 / fps / 再生時間
- 音声トラックの有無（製造業SOP=音声optional）
- 破損・読み込み不可の検出
- シーン分割数概算（PySceneDetect統合可能）
- 命名規約チェック（{sop}_{role}_{date}.mp4）

**使用例**:
```bash
python scripts/validate_partner_videos.py \
    --dir demo_videos/partner \
    --out validation_report.json
```

**出力**:
```json
{
  "summary": {
    "total_files": 6,
    "readable": 6,
    "naming_convention_ok": 6,
    "errors": 0
  },
  "videos": [
    {
      "filename": "oilchange_gold_202602.mp4",
      "readable": true,
      "width": 1920,
      "height": 1080,
      "fps": 30.0,
      "duration_sec": 245.3,
      "has_audio": null,
      "estimated_scenes": 25,
      "naming_convention_ok": true,
      "sop_name": "oilchange",
      "role": "gold",
      "date": "202602"
    }
  ]
}
```

#### 3-2. ✅ GT作成を最小工数にする設計（Chunk単位）

**ファイル**: `GT_CREATION_WORKFLOW.md`

**設計方針**:
- **A案（Chunk単位GT）採用**: 秒入力164回 → chunk選択のみ
- PySceneDetectのshot境界 = 視覚的に意味のある区切り
- 商談で十分強い（顧客は「逸脱検出」を求める、秒単位精度は不要）

**ワークフロー（3ステップ）**:

1. **動画indexing（自動）**
   ```bash
   python scripts/index_partner_video.py \
       --video demo_videos/partner/oilchange_gold_202602.mp4 \
       --video-id oilchange-gold \
       --hierarchical \
       --embedding-model ViT-H-14
   ```

2. **Chunk一覧生成（半自動）**
   ```bash
   python scripts/list_video_chunks.py \
       --video-id oilchange-gold \
       --level micro \
       --out chunks/oilchange_gold_chunks.json
   ```

3. **GT作成（手作業、ツール支援）**
   - Keyframe画像を見ながら `relevant_clip_ids` を記入
   - または: `create_gt_interactive.py`（作成予定）で動画再生しながら記録

**Manufacturing-v1 クエリ設計**:
- Visual queries: 45個（各SOPの重要ステップ）
- Trainee deviation queries: 37個（欠落・順序・安全違反）
- 合計: 82個

#### 3-3. ✅ ベンチマークバリデーション（P1安全装置）

**ファイル**: `scripts/validate_benchmark.py`

**チェック項目**:
- GT未指定（relevant_clip_ids + relevant_time_ranges両方空）
- GT時間範囲が広すぎ（> 60秒で警告）
- 重複query_id
- video_id存在確認（video_paths.local.json）

**使用例**:
```bash
python scripts/validate_benchmark.py \
    --benchmark benchmarks/manufacturing_v1.jsonl \
    --video-map benchmarks/video_paths.local.json
```

**出力例**:
```
ℹ️  Total queries: 82
ℹ️  Errors: 0
ℹ️  Warnings: 3
⚠️  m05: GT time range too wide (72.3s > 60.0s). This may cause R@1=1.0 saturation.
✅ Validation PASSED
```

#### 3-4. ✅ 補助スクリプト作成

**`scripts/index_partner_video.py`**:
- vigil_helpers.index_video_all_levels() のラッパー
- 製造業SOP用のデフォルト（hierarchical + ViT-H-14）
- 使用例:
  ```bash
  python scripts/index_partner_video.py \
      --video oilchange_gold.mp4 \
      --video-id oilchange-gold \
      --hierarchical \
      --embedding-model ViT-H-14
  ```

**`scripts/list_video_chunks.py`**:
- Qdrant/FAISSからchunk一覧を抽出
- GT作成用のJSON生成
- 使用例:
  ```bash
  python scripts/list_video_chunks.py \
      --video-id oilchange-gold \
      --level micro \
      --out chunks/oilchange_gold_chunks.json
  ```

---

## 実データ受領後の運用フロー（確定版）

### Day 1: 受領・検品（30分）

```bash
# 1. 検品実行
python scripts/validate_partner_videos.py \
    --dir demo_videos/partner \
    --out validation_report.json

# 2. 結果確認
cat validation_report.json | jq '.summary'
```

**チェック項目**:
- ✅ 命名規約OK（{sop}_{role}_{date}.mp4）
- ✅ 動画再生可能
- ✅ 解像度・fps・再生時間記録
- ⚠️ 音声なしでもOK（製造業SOP=visual中心）

### Day 2-3: Indexing（2〜4時間）

```bash
# Gold動画indexing（6本 × 30分 = 3時間）
for video in oilchange tirechange brakepad; do
    python scripts/index_partner_video.py \
        --video demo_videos/partner/${video}_gold_202602.mp4 \
        --video-id ${video}-gold \
        --hierarchical \
        --embedding-model ViT-H-14
done

# Trainee動画indexing（9本 × 20分 = 3時間）
for video in oilchange_trainee1 oilchange_trainee2 ...; do
    python scripts/index_partner_video.py \
        --video demo_videos/partner/${video}.mp4 \
        --video-id ${video} \
        --hierarchical \
        --embedding-model ViT-H-14
done
```

### Day 4-5: GT作成（1日）

```bash
# 1. Chunk一覧生成（全動画分）
for video_id in oilchange-gold tirechange-gold brakepad-gold; do
    python scripts/list_video_chunks.py \
        --video-id ${video_id} \
        --level micro \
        --out chunks/${video_id}_chunks.json
done

# 2. GT作成（手作業）
# - chunks/*.json を見ながらkeyframe確認
# - benchmarks/manufacturing_v1.jsonl に relevant_clip_ids 記入

# 3. バリデーション
python scripts/validate_benchmark.py \
    --benchmark benchmarks/manufacturing_v1.jsonl \
    --video-map benchmarks/video_paths.local.json
```

### Day 6: 評価実行（1日）

```bash
# Manufacturing-v1ベンチマーク評価
python scripts/evaluate_vigil_real.py \
    --benchmark benchmarks/manufacturing_v1.jsonl \
    --video-map benchmarks/video_paths.local.json \
    --hierarchical \
    --embedding-model ViT-H-14 \
    --output results/manufacturing_v1_results.json

# 結果確認
cat results/manufacturing_v1_results.json | jq '.aggregate'
```

**期待結果**:
- R@1 = 0.7〜0.85（real_v2の1.0飽和を脱却）
- MRR = 0.8〜0.9
- R@5 = 0.9〜1.0

### Day 7: レポート生成（半日）

```bash
# 全Trainee動画で逸脱検出レポート生成
for trainee in oilchange_trainee1 oilchange_trainee2 ...; do
    python scripts/sopilot_evaluate_pilot.py \
        --gold demo_videos/partner/${sop}_gold_202602.mp4 \
        --trainee demo_videos/partner/${trainee}.mp4 \
        --sop ${sop} \
        --out reports/${trainee}_report.json

    # PDF生成
    python scripts/sopilot_evaluate_pilot.py \
        --gold demo_videos/partner/${sop}_gold_202602.mp4 \
        --trainee demo_videos/partner/${trainee}.mp4 \
        --sop ${sop} \
        --out reports/${trainee}_report.pdf
done
```

**納品物**:
- JSON/PDF レポート（各Trainee動画分）
- 逸脱一覧（欠落・順序・安全違反）
- タイムスタンプ付き
- 是正アクション提案

---

## 成果物サマリー

### ドキュメント（2本）

1. **PARTNER_DATA_SENDING_PROCEDURE.md**
   - 送付メッセージ（コピペ用）
   - 受け取り方法・命名規約・チェックリスト

2. **GT_CREATION_WORKFLOW.md**
   - Chunk単位GT設計（A案）
   - 3ステップワークフロー
   - Manufacturing-v1クエリ設計（82個）

### スクリプト（4本）

1. **validate_partner_videos.py**: 受領データ検品
2. **index_partner_video.py**: 動画indexingラッパー
3. **list_video_chunks.py**: Chunk一覧生成
4. **validate_benchmark.py**: ベンチマークバリデータ

---

## 次のアクション（優先順）

### 📧 即座（今日中）

**PARTNER_DATA_REQUEST.md を送付**:
1. PARTNER_DATA_SENDING_PROCEDURE.md の「送付メッセージ」をコピー
2. PARTNER_DATA_REQUEST.md を添付
3. パートナーにメール/Slack送信
4. 受け取り方法を確認（Google Drive推奨）

### ⏸️ データ受領後（1週間）

1. **受領後30分**: validate_partner_videos.py で検品
2. **受領後24時間**: 全動画indexing完了
3. **受領後2〜3日**: GT作成（82クエリ）
4. **受領後4〜5日**: Manufacturing-v1評価実行
5. **受領後6〜7日**: レポート生成・送付

---

## R@1=1.0の扱い（重要）

### 発見

P0修正後、real_v2で R@1=1.0, MRR=1.0 が出た。
→ **ベンチマーク簡単すぎ/GT広すぎ**

### 対策（既に実装済み）

1. **validate_benchmark.py**: GT範囲 > 60秒で警告
2. **min_overlap_sec調整可能**: 0.0（緩い） → 0.5〜1.0（厳しい）
3. **Manufacturing-v1で実データ**: target R@1=0.7〜0.85

### 商談での説明

- ❌ 「R@1=1.0は性能が完璧」（real_v2が簡単すぎるだけ）
- ✅ 「Manufacturing-v1で実データ難易度を測定中」（R@1=0.7〜0.85目標）
- ✅ 「顧客は研究指標ではなく、逸脱検出レポートを求める」

---

## Git状況

**最新コミット**: eb98848
```
eb98848 feat: Commercial readiness - Partner data pipeline (2️⃣ + 3️⃣)
005b4e1 docs: Add P0 completion summary (all 6 actions complete)
f39de3f docs: Update COMMERCIAL_READINESS to reflect P0 fix completion
ae269a6 fix: P0 evaluation metrics - eliminate circular dependency
```

**追加ファイル（6本）**:
- PARTNER_DATA_SENDING_PROCEDURE.md
- GT_CREATION_WORKFLOW.md
- scripts/validate_partner_videos.py
- scripts/index_partner_video.py
- scripts/list_video_chunks.py
- scripts/validate_benchmark.py

---

## まとめ

### ✅ 完了（技術 + 運用）

1. **P0修正完了**（評価指標循環参照排除、所要20分）
2. **パートナー送付準備完了**（メッセージ・受領方法・命名規約）
3. **受領→評価→レポート運用確定**（Day 1〜7チェックリスト）
4. **GT作成最小工数化**（Chunk単位、秒入力164回 → chunk選択のみ）
5. **検品・バリデーションスクリプト完備**（自動化）

### 📧 次の一手（今日中）

**PARTNER_DATA_REQUEST.md送付**:
- 送付メッセージ: PARTNER_DATA_SENDING_PROCEDURE.md参照
- 受け取り方法: Google Drive / Box（推奨）
- 命名規約: {sop}_{role}_{date}.mp4
- 納期約束: データ提供から1週間でレポート返却

### ⏸️ データ受領待ち

受領後は Day 1〜7 フローに従って運用開始。
Manufacturing-v1 評価で R@1=0.7〜0.85 を確認 → 商談で使える実データ実証完了。

---

**ステータス**: ✅ **商用パイプライン準備完了、パートナー送付のみ**

**方針**: 研究 → **金を取る実装**に完全シフト完了
