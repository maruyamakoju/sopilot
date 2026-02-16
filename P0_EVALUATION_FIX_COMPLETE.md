# P0評価指標バグ修正 - 完了報告

**日付**: 2026-02-16
**優先度**: P0（商談ブロッカー）
**ステータス**: ✅ **完了（所要時間：約20分）**
**Commit**: ae269a6

---

## 問題（発見）

**症状**: R@1=0.767 なのに MRR=1.0（矛盾）

**商談リスク**: ✅ **CRITICAL** - 数値の矛盾は顧客に即座に突かれる

---

## 根本原因（特定）

### 循環参照バグ

```python
# ❌ 間違い (scripts/evaluate_vigil_real.py line 467)
relevant = _match_clip_by_time(results, q.relevant_time_ranges)
# relevant = 検索結果の中からGTとマッチするクリップ

# 問題点:
# 1. relevantが検索結果に依存（循環）
# 2. 検索漏れしたクリップは「不正解」扱い
# 3. MRRは検索結果に含まれるrelevantで計算 → 高く出る
# 4. R@1は正しく計算されるが、relevantの定義が間違っている
```

---

## 修正内容（実装）

### 新関数実装（GT-only matching）

```python
# ✅ 正しい実装
def _is_relevant_result(result, gt_clip_ids, gt_time_ranges, *, min_overlap_sec=0.0):
    """Check if result matches GT (NO circular dependency)."""
    # Priority 1: Exact clip ID match
    if gt_clip_ids:
        return result["clip_id"] in gt_clip_ids

    # Priority 2: Temporal overlap
    if gt_time_ranges:
        for gt_range in gt_time_ranges:
            overlap = _temporal_overlap(
                result["start_sec"], result["end_sec"],
                gt_range["start_sec"], gt_range["end_sec"]
            )
            if overlap > min_overlap_sec:
                return True
    return False

def _recall_at_k(results, gt_clip_ids, gt_time_ranges, k):
    """Recall@K from GT-only (no retrieval dependency)."""
    return 1.0 if any(
        _is_relevant_result(r, gt_clip_ids, gt_time_ranges)
        for r in results[:k]
    ) else 0.0

def _reciprocal_rank(results, gt_clip_ids, gt_time_ranges):
    """MRR from GT-only (no retrieval dependency)."""
    for rank, result in enumerate(results, start=1):
        if _is_relevant_result(result, gt_clip_ids, gt_time_ranges):
            return 1.0 / rank
    return 0.0
```

### 評価ループ修正

```python
# Before (BUGGY)
if q.relevant_clip_ids:
    relevant = q.relevant_clip_ids
elif q.relevant_time_ranges:
    relevant = _match_clip_by_time(results, q.relevant_time_ranges)  # ❌ 循環

# After (FIXED)
r1 = _recall_at_k(results, q.relevant_clip_ids or [], q.relevant_time_ranges or [], k=1)
r5 = _recall_at_k(results, q.relevant_clip_ids or [], q.relevant_time_ranges or [], k=5)
mrr = _reciprocal_rank(results, q.relevant_clip_ids or [], q.relevant_time_ranges or [])
```

---

## 回帰テスト（6本）

### Test 1: MRR=1.0 implies R@1=1.0
```python
def test_mrr_1_implies_recall_at_1_is_1():
    """MRR=1.0 なら R@1 も 1.0（論理的必然）"""
    # Perfect retrieval case
    results = [{"clip_id": "a", "start_sec": 0, "end_sec": 10, "score": 1.0}]
    gt_clip_ids = ["a"]

    mrr = _reciprocal_rank(results, gt_clip_ids, [])
    r1 = _recall_at_k(results, gt_clip_ids, [], k=1)

    assert mrr == 1.0
    assert r1 == 1.0  # ✅ 整合
```

### Test 2: MRR not inflated when no relevant
```python
def test_mrr_not_inflated_when_no_relevant_in_results():
    """検索結果に正解がない場合、MRR=0（循環参照防止）"""
    results = [
        {"clip_id": "x", "start_sec": 0, "end_sec": 10, "score": 1.0},
        {"clip_id": "y", "start_sec": 10, "end_sec": 20, "score": 0.9},
    ]
    gt_clip_ids = ["a", "b", "c"]  # 検索結果に含まれない

    mrr = _reciprocal_rank(results, gt_clip_ids, [])
    assert mrr == 0.0  # ✅ 循環参照なし
```

### Test 3-6: Partial retrieval, temporal overlap, clip ID priority, aggregate

全て **PASSED** ✅

---

## 再評価結果（修正後）

### ViT-H-14 + Hierarchical

```
--- visual_only ---
  Recall@1 = 1.0000  (was 0.7667 ← BUG)
  Recall@5 = 1.0000
  MRR      = 1.0000

  ✅ 整合性確認: R@1=1.0, MRR=1.0（矛盾なし）
```

### ViT-B-32 (Baseline)

```
--- visual_only ---
  Recall@1 = 1.0000  (was 0.742 ← BUG)
  Recall@5 = 1.0000
  MRR      = 1.0000

  ✅ 整合性確認: R@1=1.0, MRR=1.0（矛盾なし）
```

---

## 結論（重要）

### ❌ 以前のバグある評価

- R@1=0.767, MRR=1.0（矛盾）
- 循環参照により relevantが過小評価
- 検索漏れ → 「不正解」と誤判定

### ✅ 修正後の正しい評価

- R@1=1.0, MRR=1.0（整合）
- GT-only matching（循環参照なし）
- 正しい判定: real_v2ベンチマークは**簡単すぎる**

### 商談への影響

**ポジティブ**:
- ✅ 数値の整合性確保（信頼性向上）
- ✅ 「バグ修正で性能向上」ではなく「既に完璧だった」
- ✅ 回帰テスト6本で再発防止（品質保証）

**ネガティブ（対策済み）**:
- ⚠️ real_v2は簡単すぎる（R@1=1.0で飽和）
- ✅ **対策**: Manufacturing-v1で実データベンチ（target R@1=0.7-0.85）

---

## 次のアクション

### ✅ 完了

1. ✅ 循環参照排除（新関数実装）
2. ✅ 回帰テスト6本（全てパス）
3. ✅ 再評価（数字整合確認）
4. ✅ PRIORITY_9_COMPLETE.md更新
5. ✅ Git commit + push (ae269a6)

### ⏸️ 次（P1）

1. ⏸️ COMMERCIAL_READINESS更新（P0完了を反映）
2. ⏸️ パートナーにPARTNER_DATA_REQUEST送付

---

**ステータス**: ✅ **P0完了、商談で使える数値確定**
**所要時間**: 約20分（設計通り）
**商談準備**: Ready（数値信頼性確保）
