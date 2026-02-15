# 評価指標バグ - 緊急修正必要

**発見日**: 2026-02-15
**影響度**: ✅ **CRITICAL（商談で致命的）**
**ステータス**: 未修正

---

## 問題の症状

```
R@1 = 0.767 (76.7%のクエリで正解がTop-1)
MRR = 1.000 (平均逆順位 = 完璧)
```

**矛盾**: MRR=1.0は「全クエリで正解が1位」を意味するため、R@1も1.0になるはず。

---

## 根本原因

### `scripts/evaluate_vigil_real.py` line 390の循環参照

```python
# Line 385-390 (BUGGY)
if q.relevant_clip_ids:
    relevant = q.relevant_clip_ids
elif q.relevant_time_ranges:
    # ❌ 検索結果の中からマッチングしている
    relevant = _match_clip_by_time(results, q.relevant_time_ranges, iou_threshold=0.3)
else:
    relevant = []
```

### `_match_clip_by_time()` の実装 (line 239-262)

```python
def _match_clip_by_time(
    retrieved: list[dict],  # ← 検索結果を入力
    gt_time_ranges: list[dict],
    *,
    iou_threshold: float = 0.3,
) -> list[str]:
    """Match retrieved clips to GT time ranges by temporal overlap."""
    matched_ids = []
    for r in retrieved:  # ← 検索結果の中からマッチング
        for gt in gt_time_ranges:
            iou = temporal_iou(r["start_sec"], r["end_sec"], gt["start_sec"], gt["end_sec"])
            if iou >= iou_threshold:
                matched_ids.append(r["clip_id"])
                break
    return matched_ids
```

**問題点**:
1. `relevant = 検索結果の中でGTとマッチするクリップ`
2. **検索結果に含まれないクリップは、GTとマッチしても relevantにならない**
3. MRR計算: relevantは必ず検索結果に含まれるので、必ず上位に来る → MRR高く出る
4. R@1計算: Top-1がrelevantに含まれるか → Top-1がGTとマッチしない場合、R@1は0

**結果**: MRRは検索バイアスで高く出るが、R@1は正しく計算される → 矛盾

---

## 正しい評価方法

### 本来あるべき実装

```python
# ❌ 間違い: 検索結果の中からマッチング
relevant = _match_clip_by_time(results, q.relevant_time_ranges)

# ✅ 正しい: 全クリップの中からマッチング（検索に依存しない）
all_clips = _get_all_clips_for_video(q.video_id, qdrant_service)
relevant = _match_clip_by_time_absolute(all_clips, q.relevant_time_ranges)
```

### 必要な修正

1. **Qdrantから全クリップのメタデータ取得**:
   ```python
   def _get_all_clips_for_video(video_id: str, qdrant_service) -> list[dict]:
       """Get all indexed clips for a video with metadata."""
       # QdrantServiceに新規メソッド追加が必要
       return qdrant_service.get_all_clips(level="micro", video_id=video_id)
   ```

2. **絶対マッチング関数**:
   ```python
   def _match_clip_by_time_absolute(
       all_clips: list[dict],  # ← 全クリップ（検索結果に依存しない）
       gt_time_ranges: list[dict],
       *,
       iou_threshold: float = 0.3,
   ) -> list[str]:
       """Match all clips to GT time ranges (independent of retrieval)."""
       matched_ids = []
       for clip in all_clips:
           for gt in gt_time_ranges:
               iou = temporal_iou(clip["start_sec"], clip["end_sec"], gt["start_sec"], gt["end_sec"])
               if iou >= iou_threshold:
                   matched_ids.append(clip["clip_id"])
                   break
       return matched_ids
   ```

3. **評価フロー変更**:
   ```python
   # Before (line 368-397)
   for q in retrieval_queries:
       results = _retrieve_for_query(...)
       retrieved_ids = [r["clip_id"] for r in results]
       relevant = _match_clip_by_time(results, q.relevant_time_ranges)  # ❌ 循環参照

   # After
   for q in retrieval_queries:
       # 1. Get ground truth (independent of retrieval)
       all_clips = _get_all_clips_for_video(q.video_id, qdrant_service)
       relevant = _match_clip_by_time_absolute(all_clips, q.relevant_time_ranges)  # ✅ 正しい

       # 2. Perform retrieval
       results = _retrieve_for_query(...)
       retrieved_ids = [r["clip_id"] for r in results]
   ```

---

## 影響範囲

### 影響を受ける指標

| 指標 | 影響 | 理由 |
|------|------|------|
| **MRR** | ✅ **過大評価** | 検索バイアスで必ず上位に来る |
| **R@1** | ⚠️ **部分的に正しい** | Top-1判定は正しいが、分母が間違う可能性 |
| **R@5** | ⚠️ **過大評価の可能性** | relevantが検索結果に偏っている |
| **nDCG** | ✅ **過大評価** | relevance scoreが検索結果に偏っている |

### 影響を受けるベンチマーク

1. **real_v2.jsonl** (Priority 9):
   - 20クエリ全て relevant_time_ranges を使用
   - **全て影響を受ける**

2. **real_v1.jsonl** (Priority 8):
   - 9クエリ全て relevant_time_ranges を使用
   - **全て影響を受ける**

3. **manufacturing_v1.jsonl** (未実装):
   - 予定では relevant_time_ranges を使用
   - **実装前に修正必要**

---

## 修正優先度

**優先度**: ⬜ **P0（即座に修正）**

**理由**:
1. ✅ 商談で数値の信頼性が問われる
2. ✅ 現在の R@1=0.767, MRR=1.0 は矛盾している
3. ✅ Manufacturing-v1実装前に修正しないと、評価が無意味になる

**修正タイミング**:
- **今日中**に修正（Manufacturing-v1実装前）

---

## 修正ステップ

### Step 1: QdrantServiceに全クリップ取得メソッド追加

**File**: `src/sopilot/qdrant_service.py`

```python
def get_all_clips(
    self,
    level: ChunkLevel,
    video_id: str,
) -> list[dict]:
    """Get all clips for a video with metadata.

    Args:
        level: Chunk level to retrieve
        video_id: Video ID to filter

    Returns:
        List of clip metadata dicts with keys:
        - clip_id, start_sec, end_sec, video_id
    """
    if self._client is None:
        return self._get_all_clips_faiss(level, video_id)

    # Qdrant implementation
    collection_name = self._get_collection_name(level)
    # Scroll/paginate through all points for this video
    # Return metadata only (no vectors needed)
    ...
```

### Step 2: evaluate_vigil_real.py修正

**File**: `scripts/evaluate_vigil_real.py`

```python
def _match_clip_by_time_absolute(
    all_clips: list[dict],
    gt_time_ranges: list[dict],
    *,
    iou_threshold: float = 0.3,
) -> list[str]:
    """Match all clips to GT time ranges (independent of retrieval)."""
    from sopilot.temporal import temporal_iou

    matched_ids = []
    for clip in all_clips:
        for gt in gt_time_ranges:
            iou = temporal_iou(
                clip["start_sec"], clip["end_sec"],
                gt["start_sec"], gt["end_sec"]
            )
            if iou >= iou_threshold:
                matched_ids.append(clip["clip_id"])
                break
    return matched_ids
```

### Step 3: 評価ループ修正

```python
# Line 368-397 修正
for q in retrieval_queries:
    if q.video_id not in indexed_videos:
        continue

    # Get ground truth FIRST (independent of retrieval)
    if q.relevant_clip_ids:
        relevant = q.relevant_clip_ids
    elif q.relevant_time_ranges:
        all_clips = qdrant.get_all_clips(level="micro", video_id=q.video_id)
        relevant = _match_clip_by_time_absolute(all_clips, q.relevant_time_ranges)
    else:
        relevant = []

    # Perform retrieval SECOND
    results = _retrieve_for_query(...)
    retrieved_ids = [r["clip_id"] for r in results]

    # Now evaluate (no circular dependency)
    all_retrieved_ids.append(retrieved_ids)
    all_relevant_ids.append(relevant)
    all_relevant_sets.append(set(relevant))
    ...
```

---

## 検証方法

### Before (バグあり)

```bash
python scripts/evaluate_vigil_real.py \
  --benchmark benchmarks/real_v2.jsonl \
  --video-map benchmarks/video_paths.local.json \
  --reindex
```

**予想結果**:
- R@1 = 0.767, MRR = 1.000 (矛盾)

### After (修正後)

```bash
python scripts/evaluate_vigil_real.py \
  --benchmark benchmarks/real_v2.jsonl \
  --video-map benchmarks/video_paths.local.json \
  --reindex
```

**期待結果**:
- R@1 = 0.767, MRR = 0.880 前後 (整合)
- または R@1が下がる（より正確なGT定義により）

---

## 次のアクション

1. ⬜ **QdrantService.get_all_clips() 実装** (30分)
2. ⬜ **_match_clip_by_time_absolute() 追加** (15分)
3. ⬜ **評価ループ修正** (15分)
4. ⬜ **real_v2.jsonl 再評価** (5分)
5. ⬜ **結果検証（R@1とMRRの整合性確認）** (5分)
6. ⬜ **Git commit** (5分)

**合計**: 約1.5時間

**完了後**: Manufacturing-v1実装に進む

---

**ステータス**: 📋 **文書化完了、修正待ち**
