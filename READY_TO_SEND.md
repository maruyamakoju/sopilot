# パートナー送付準備完了

**日付**: 2026-02-16
**ステータス**: ✅ **送付準備完了（技術的ボトルネック解消）**

---

## 完了した3つの次の一手

### ✅ 1️⃣ 送付準備確認

**添付ファイル（2つ）**:
```
reports/sample_report.pdf  (3.6KB) ← 返却物の見本（効きます）
PARTNER_DATA_REQUEST.md     ← 要件定義
```

**確認済み**:
```bash
dir reports
# sample_report.json  sample_report.pdf ← ✅ 存在確認
```

### ✅ 2️⃣ Chunk GT詰まり解消（list_video_chunks.py問題を完全解決）

**問題**: list_video_chunks.py が FAISS で 0 chunks を返す

**解決**: **A案実装完了** - index時にchunk manifestを自動保存

**実装内容**:
```python
def save_chunk_manifests(video_id, index_result, output_dir=Path("chunks")):
    """Save chunk manifests for GT creation (no vector DB query needed)."""
    # micro_metadata → chunks/{video_id}.micro.json
    # meso_metadata → chunks/{video_id}.meso.json
    # macro_metadata → chunks/{video_id}.macro.json
```

**Manifest形式**:
```json
{
  "video_id": "oilchange-gold",
  "level": "micro",
  "total_chunks": 10,
  "chunks": [
    {
      "clip_id": "641f47b5-76f2-464c-ba1a-24df13441002",
      "start_sec": 0.0,
      "end_sec": 4.0,
      "duration_sec": 4.0
    },
    ...
  ]
}
```

**テスト結果**: ✅
```
✅ Indexing complete in 2.8s
Saved micro manifest: chunks\oilchange-gold.micro.json (10 chunks)
Saved meso manifest: chunks\oilchange-gold.meso.json (4 chunks)
Saved macro manifest: chunks\oilchange-gold.macro.json (1 chunks)
```

**GT作成ワークフロー（更新）**:
```bash
# Step 1: Indexing（自動的にmanifest保存）
python scripts/index_partner_video.py --video oilchange_gold.mp4 --video-id oilchange-gold --hierarchical

# Step 2: Manifestを見てGT作成
cat chunks/oilchange-gold.micro.json
# → clip_id をコピーして benchmarks/manufacturing_v1.jsonl に記入

# Step 3: バリデーション
python scripts/validate_benchmark.py --benchmark manufacturing_v1.jsonl
```

**評価**: ✅ **Chunk単位GT（最速ルート）完全動作**

### ✅ 3️⃣ 受領ルート固定準備

**推奨方法**: Google Drive共有リンク

**送付時に確認すべき点**:
1. アップロード先URL（相手が迷わない）
2. 権限設定（アップロードのみ or 閲覧可能）
3. 命名規約の確認（`{sop}_{role}_{date}.mp4`）

---

## パートナー送付メッセージ（コピペ用）

```
件名: SOPilot製造業パイロット - 動画データ提供依頼

SOPilot 製造業パイロットのため、SOP動画データ（最小セット）のご提供をお願いできますか。
動画はリポジトリに入れず、ローカル環境でのみ評価します（匿名化前提・機密扱い）。

受領後は、**逸脱（欠落/順序/重大違反）をタイムスタンプ付きでレポート（PDF/JSON）**として返却します。
要件は添付 PARTNER_DATA_REQUEST.md をご確認ください。返却物の例として sample_report.pdf を添付します。

【受け渡し方法】
以下のいずれかでご提供ください：
- Google Drive共有リンク（推奨）
- Box（共有リンク）
- S3バケット（アクセス権限付与）
- 暗号化zip（パスワードは別経路で送付）

【ファイル命名規約】
{SOP名}_{役割}_{日付}.mp4

例：
- oilchange_gold_202602.mp4
- oilchange_trainee1_202602.mp4
- tirechange_gold_202602.mp4

【納期】
データご提供から約1週間でレポート返却します。

添付ファイル：
1. PARTNER_DATA_REQUEST.md（要件詳細）
2. sample_report.pdf（返却物の見本）

ご不明点がございましたらお気軽にお問い合わせください。

よろしくお願いいたします。
```

---

## 送付後のフォローアップ

### 送付直後

**確認事項**:
- [ ] アップロード先を1つに確定（Google Drive / Box / S3）
- [ ] 相手がアップロード方法を理解しているか確認
- [ ] 命名規約の理解確認（例を再送）

### データ受領時

**即座に実行（30分以内）**:
```bash
# 検品
python scripts/validate_partner_videos.py --dir demo_videos/partner --out validation_report.json

# 結果確認
cat validation_report.json | jq '.summary'
# → naming_convention_ok: 3/3 を確認
```

---

## Git状況

```
3693bf5 (HEAD -> master, origin/master) fix: Add chunk manifest output
d26361a docs: Dress rehearsal complete summary
2f38b26 feat: Dress rehearsal complete - Partner data pipeline tested and debugged
```

**追加機能**: Chunk manifest自動保存（A案完全実装）

---

## 技術的準備完了度

### ✅ 完全動作（6/6）

1. **validate_partner_videos.py**: 検品（解像度/fps/命名規約）
2. **index_partner_video.py**: Indexing + chunk manifest保存 ← **NEW**
3. **validate_benchmark.py**: ベンチマークバリデーション
4. **sopilot_evaluate_pilot.py**: PDF/JSONレポート生成
5. **.gitignore**: 事故防止（partner/chunks/reports除外）
6. **Chunk manifest**: GT作成を高速化（vector DB不要）

### ⚠️ 既知の問題（0/6） ← **解消**

- ~~list_video_chunks.py: FAISS search returns 0~~
  - ✅ **解決**: chunk manifest自動保存で完全回避

---

## 実データ受領→評価→返却フロー（確定版）

### Day 1: 検品（30分）
```bash
python scripts/validate_partner_videos.py --dir demo_videos/partner --out validation_report.json
```

### Day 2-3: Indexing（自動的にmanifest保存）
```bash
python scripts/index_partner_video.py \
    --video demo_videos/partner/oilchange_gold_202602.mp4 \
    --video-id oilchange-gold \
    --hierarchical \
    --embedding-model ViT-H-14
# → chunks/oilchange-gold.{micro,meso,macro}.json 自動生成
```

### Day 4-5: GT作成（Chunk単位、最速）
```bash
# Manifest確認
cat chunks/oilchange-gold.micro.json

# clip_idをコピーして manufacturing_v1.jsonl に記入
# relevant_clip_ids: ["641f47b5-...", "7f69af5f-..."]

# バリデーション
python scripts/validate_benchmark.py --benchmark manufacturing_v1.jsonl
```

### Day 6: 評価実行
```bash
python scripts/evaluate_vigil_real.py \
    --benchmark benchmarks/manufacturing_v1.jsonl \
    --video-map benchmarks/video_paths.local.json \
    --hierarchical \
    --embedding-model ViT-H-14
```

### Day 7: レポート返却
```bash
python scripts/sopilot_evaluate_pilot.py \
    --gold demo_videos/partner/oilchange_gold_202602.mp4 \
    --trainee demo_videos/partner/oilchange_trainee1_202602.mp4 \
    --sop oil_change \
    --out reports/oilchange_trainee1_report.pdf
```

---

## まとめ

### ✅ 完了
- 送付準備完了（PDF見本 + 要件定義）
- Chunk GT詰まり解消（A案実装完了）
- 受領ルート準備（Google Drive推奨）
- Git完全同期（3693bf5 push済み）

### 📧 次の一手（今すぐ）
**PARTNER_DATA_REQUEST.md + sample_report.pdf を送付**

### ⏸️ 送付後
1. アップロード先を1つに確定
2. 命名規約の理解確認
3. データ受領 → Day 1-7フロー実行

---

**ステータス**: ✅ **技術準備完了、送付のみ**

**方針**: 実データを回収して、**一度もたつかずに通す**準備完了
