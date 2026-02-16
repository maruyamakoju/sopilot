# Ground Truth (GT) 作成ワークフロー

**日付**: 2026-02-16
**目的**: Manufacturing-v1ベンチマーク用のGT作成を最小工数で実現

---

## 設計方針：Chunk単位GT（A案）

### なぜChunk単位か

**手作業で秒を打つと詰む**:
- 例: 5分動画で82クエリ → 82回 × start/end秒 = 164回の時刻入力
- エラー率高い、工数大きい、再現性低い

**Chunk単位なら**:
- 動画をindexing済みなら、micro chunk（clip_id）が既に存在
- 人間は「このchunkが正解」を選ぶだけ
- クエリ作成が高速、GTの再利用可能

### 商談でも十分強い理由

- Chunk境界 = PySceneDetectのshot境界 = 視覚的に意味のある区切り
- 「オイル交換の手順3（フィルター取り外し）」は1〜2 micro chunksに収まる
- 顧客は「秒単位の正確さ」より「逸脱を見落とさない」を求める

---

## ワークフロー（3ステップ）

### Step 1: 動画のIndexing（自動）

```bash
# Gold動画をindexing
python scripts/index_partner_video.py \
    --video demo_videos/partner/oilchange_gold_202602.mp4 \
    --video-id oilchange-gold \
    --hierarchical \
    --embedding-model ViT-H-14 \
    --reindex
```

**出力**: Qdrantに保存された micro/meso/macro chunks

**確認**:
```python
from sopilot.qdrant_service import QdrantService
qdrant = QdrantService(...)
clips = qdrant.search(video_id="oilchange-gold", level="micro", query_vector=..., k=100)
# clips[i]["clip_id"], clips[i]["start_sec"], clips[i]["end_sec"]
```

### Step 2: Chunk一覧の生成（半自動）

```bash
# Chunk一覧をJSON出力
python scripts/list_video_chunks.py \
    --video-id oilchange-gold \
    --level micro \
    --out chunks/oilchange_gold_chunks.json
```

**出力例** (`chunks/oilchange_gold_chunks.json`):
```json
{
  "video_id": "oilchange-gold",
  "level": "micro",
  "chunks": [
    {
      "clip_id": "oilchange-gold_micro_0",
      "start_sec": 0.0,
      "end_sec": 8.5,
      "duration_sec": 8.5,
      "keyframe_path": "artifacts/oilchange-gold/keyframes/micro_0.jpg"
    },
    {
      "clip_id": "oilchange-gold_micro_1",
      "start_sec": 8.5,
      "end_sec": 15.2,
      "duration_sec": 6.7,
      "keyframe_path": "artifacts/oilchange-gold/keyframes/micro_1.jpg"
    },
    ...
  ]
}
```

### Step 3: GT作成（手作業、ツール支援）

#### 方法A: Keyframeを見ながら手作業

1. `chunks/oilchange_gold_chunks.json` を開く
2. 各クエリに対して、keyframe画像を確認
3. 正解となる `clip_id` をリストアップ
4. `benchmarks/manufacturing_v1.jsonl` に記入

**例**:
```jsonl
{
  "query_id": "m01",
  "query_text": "Worker wearing safety glasses and gloves (PPE)",
  "query_type": "visual",
  "video_id": "oilchange-gold",
  "relevant_clip_ids": ["oilchange-gold_micro_0", "oilchange-gold_micro_1"],
  "relevant_time_ranges": []
}
```

#### 方法B: 動画再生しながらツールで記録（推奨）

**ツール**: `scripts/create_gt_interactive.py`（作成予定）

```bash
python scripts/create_gt_interactive.py \
    --video demo_videos/partner/oilchange_gold_202602.mp4 \
    --chunks chunks/oilchange_gold_chunks.json \
    --out benchmarks/manufacturing_v1_gt.jsonl
```

**操作**:
- 動画再生中に `SPACE` でchunk境界をマーク
- `1〜9` キーで「このchunkが正解」をタグ付け
- `S` でクエリとGTを保存

**出力**: `manufacturing_v1_gt.jsonl` に追記

---

## Manufacturing-v1 クエリ設計

### SOP構造

各SOPは以下の構造：
- **Gold動画**: 模範手順（全ステップ正しい）
- **Trainee動画**: 訓練者（逸脱あり）

### クエリ種類（82個の内訳）

#### 1. Visual Queries（45個）

各SOPの重要ステップをカバー:
- PPE着用（安全メガネ、手袋）
- 車両配置（ジャッキ/ランプ）
- 工具使用（レンチ、トルクレンチ）
- 手順の視覚的証拠（フィルター取り外し、オイル注入）

#### 2. Trainee Deviation Queries（37個）

訓練者動画での逸脱検出:
- 欠落（Missing）: ステップをスキップ
- 順序ミス（Wrong sequence）: 手順が逆
- 安全違反（Safety violation）: PPE未着用、トルク確認スキップ

### クエリ例（Oil Change SOP）

```jsonl
// Visual: Gold動画でのステップ確認
{"query_id": "m01", "query_text": "Worker wearing safety glasses and gloves", "query_type": "visual", "video_id": "oilchange-gold", "relevant_clip_ids": ["oilchange-gold_micro_0"]}
{"query_id": "m02", "query_text": "Vehicle positioned on jack or ramp", "query_type": "visual", "video_id": "oilchange-gold", "relevant_clip_ids": ["oilchange-gold_micro_1"]}
{"query_id": "m03", "query_text": "Using wrench to remove oil filter", "query_type": "visual", "video_id": "oilchange-gold", "relevant_clip_ids": ["oilchange-gold_micro_5"]}

// Deviation: Trainee動画での逸脱検出
{"query_id": "m11", "query_text": "Missing safety equipment (no gloves)", "query_type": "visual", "video_id": "oilchange-trainee1", "relevant_clip_ids": ["oilchange-trainee1_micro_0"]}
{"query_id": "m12", "query_text": "Wrong sequence: oil added before filter installed", "query_type": "visual", "video_id": "oilchange-trainee1", "relevant_clip_ids": ["oilchange-trainee1_micro_7"]}
{"query_id": "m13", "query_text": "Skipped torque verification step", "query_type": "visual", "video_id": "oilchange-trainee1", "relevant_clip_ids": []}
```

**Note**: `relevant_clip_ids: []` は「該当なし」（スキップされたステップ）

---

## ベンチマークバリデーション（P1安全装置）

### 自動チェック項目

```python
def validate_benchmark(benchmark_path: Path) -> list[str]:
    """Validate Manufacturing-v1 benchmark.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    with open(benchmark_path) as f:
        queries = [json.loads(line) for line in f]

    for q in queries:
        # 1. relevant_clip_ids または relevant_time_ranges が必須
        if not q.get("relevant_clip_ids") and not q.get("relevant_time_ranges"):
            errors.append(f"{q['query_id']}: No GT specified")

        # 2. relevant_time_ranges が動画全体は警告
        if q.get("relevant_time_ranges"):
            for r in q["relevant_time_ranges"]:
                duration = r["end_sec"] - r["start_sec"]
                if duration > 60:  # 60秒以上は広すぎ
                    errors.append(f"{q['query_id']}: GT range too wide ({duration:.1f}s)")

        # 3. video_id が存在するか（video_paths.local.json）
        # （実装時に追加）

    return errors
```

### min_overlap_sec 調整

評価時に厳しさを調整可能:

```python
# 緩い（デフォルト）
r1 = _recall_at_k(results, gt_clip_ids, gt_time_ranges, k=1, min_overlap_sec=0.0)

# 厳しい（1秒以上の重複が必要）
r1 = _recall_at_k(results, gt_clip_ids, gt_time_ranges, k=1, min_overlap_sec=1.0)
```

**推奨**: `min_overlap_sec=0.5` から始めて、難易度を調整

---

## 実データ受領後の手順（チェックリスト）

### Day 1: 受領・検品

- [ ] `validate_partner_videos.py` で検品
- [ ] 命名規約確認
- [ ] 動画の再生可能性確認

### Day 2-3: Indexing + Chunk一覧生成

- [ ] Gold動画 indexing（hierarchical, ViT-H-14）
- [ ] Trainee動画 indexing
- [ ] Chunk一覧JSON生成（全動画分）

### Day 4-5: GT作成

- [ ] Gold動画のステップクエリ作成（45個）
- [ ] Trainee動画の逸脱クエリ作成（37個）
- [ ] `validate_benchmark()` でバリデーション

### Day 6: 評価実行

- [ ] `evaluate_vigil_real.py --benchmark manufacturing_v1.jsonl --hierarchical`
- [ ] R@1, MRR, R@5 計算
- [ ] 結果レビュー（難易度調整）

### Day 7: レポート生成

- [ ] `sopilot_evaluate_pilot.py` で全Trainee動画評価
- [ ] JSON/PDF生成
- [ ] パートナーへ送付

---

## 次のアクション

1. **即座**: `list_video_chunks.py` スクリプト作成（Step 2自動化）
2. **P1**: `create_gt_interactive.py` 作成（Step 3ツール支援）
3. **実データ受領後**: 上記チェックリストに従って運用

---

**ステータス**: 📋 **ワークフロー確定、ツール作成待ち**
