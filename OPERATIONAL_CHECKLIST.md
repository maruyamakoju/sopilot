# SOPilot商用パイロット：運用チェックリスト

**目的**: 実データ回収から返却までの7日間を迷わず回す

---

## 📋 Phase 0: 送信前の最終確認（今日）

### ✅ 準備完了の確認
- [x] `PARTNER_DATA_REQUEST.md` 存在確認
- [x] `reports/sample_report.pdf` 存在確認（3.6KB）
- [x] 3層セキュリティ（.gitignore + pre-commit hook + partner_private分離）
- [x] 6スクリプト動作確認（dress rehearsal完了）

### 🎯 今日やること（3ステップ）

#### 1. Google Driveフォルダ作成（5分）
```
1. Google Driveで新規フォルダ作成
2. 名前: "SOPilot_Partner_Upload"
3. 共有設定: 「リンクを知っている全員が編集可」
4. URLをコピー
```

#### 2. メール送信（10分）
```
1. PARTNER_EMAIL_TEMPLATE.md を開く
2. [Google Drive フォルダURL] を実際のURLに置換
3. 添付ファイル確認:
   - PARTNER_DATA_REQUEST.md
   - reports/sample_report.pdf
4. 送信
```

#### 3. 受領ルート記録（2分）
```bash
# Google DriveフォルダURLをローカルに記録
echo "https://drive.google.com/drive/folders/XXXXX" > partner_upload_url.txt

# .gitignoreに追加（機密URL防止）
echo "partner_upload_url.txt" >> .gitignore
```

---

## 📋 Phase 1: データ受領日（Day 1）

### 到着通知を受けたら（30分以内に完了）

#### Step 1: ダウンロード＋配置（5分）
```bash
# Google Driveから手動ダウンロード
# → demo_videos/partner/ に配置

# 確認
ls demo_videos/partner/
# 出力例:
#   oilchange_gold_20260101.mp4
#   oilchange_trainee_001.mp4
#   oilchange_trainee_002.mp4
```

#### Step 2: 一次検品（5分）
```bash
python scripts/validate_partner_videos.py \
    --input-dir demo_videos/partner \
    --output-format terminal

# 出力を確認:
#  - ファイル数
#  - 命名規約 OK/NG
#  - 解像度・fps
#  - 推定時間
```

#### Step 3: 受領確認メール送信（10分）
```
1. RECEPTION_RESPONSE_TEMPLATE.md を開く
2. 検品結果を転記（ファイル数、OK/NG）
3. 受領日時を記入
4. 返却予定日を計算（受領日+7営業日）
5. 送信
```

#### Step 4: 不足確認（10分）
- 手順リストがあるか？ → なければ即座に追加依頼
- 逸脱定義があるか？ → なければ即座に追加依頼
- **この時点で1回だけ聞く**（後で何度も聞かない）

---

## 📋 Phase 2: インデックス作成（Day 1-2）

### Gold動画のインデックス（動画1本あたり5分）

```bash
# 例: oilchange_gold_20260101.mp4
python scripts/index_partner_video.py \
    --video-path demo_videos/partner/oilchange_gold_20260101.mp4 \
    --video-id oilchange-gold \
    --hierarchical

# 確認: Chunk manifestが生成された
ls chunks/
# 出力例:
#   oilchange-gold.micro.json  (例: 10 chunks)
#   oilchange-gold.meso.json   (例: 4 chunks)
#   oilchange-gold.macro.json  (例: 1 chunk)
```

### Trainee動画のインデックス（動画1本あたり5分）

```bash
# 例: oilchange_trainee_001.mp4
python scripts/index_partner_video.py \
    --video-path demo_videos/partner/oilchange_trainee_001.mp4 \
    --video-id oilchange-trainee-001 \
    --hierarchical

# 他のTrainee動画も同様
```

---

## 📋 Phase 3: GT作成（Day 2-4）

### Chunk manifestからGT作成（1 SOPあたり30分）

```bash
# Step 1: Chunk manifestを確認
cat chunks/oilchange-gold.micro.json

# 出力例:
# {
#   "video_id": "oilchange-gold",
#   "level": "micro",
#   "total_chunks": 10,
#   "chunks": [
#     {"clip_id": "oilchange-gold-micro-0", "start_sec": 0.0, "end_sec": 9.6, ...},
#     {"clip_id": "oilchange-gold-micro-1", "start_sec": 9.6, "end_sec": 19.2, ...},
#     ...
#   ]
# }

# Step 2: テンプレートをコピー
cp benchmarks/manufacturing_v1.jsonl.template \
   benchmarks/partner_private/oilchange_v1_partner.jsonl

# Step 3: GTを記入（手作業）
# - 手順リストと照合
# - 各ステップに対応する clip_id を記入
# - 逸脱定義に対応する clip_id を記入

# 例:
# {"query_id": "oil_q01", "query_text": "オイルドレンボルトを緩める",
#  "relevant_clip_ids": ["oilchange-gold-micro-2"]}
```

### GT検証（5分）

```bash
python scripts/validate_benchmark.py \
    --benchmark benchmarks/partner_private/oilchange_v1_partner.jsonl \
    --video-map benchmarks/video_paths.local.json

# 出力を確認:
#  - 空GTがないか
#  - Time rangeが広すぎないか（>60秒）
#  - Duplicate query_idがないか
```

---

## 📋 Phase 4: 評価実行（Day 5-6）

### Trainee動画の評価（動画1本あたり10分）

```bash
# 例: oilchange_trainee_001
python scripts/evaluate_vigil_real.py \
    --benchmark benchmarks/partner_private/oilchange_v1_partner.jsonl \
    --video-map benchmarks/video_paths.local.json \
    --hierarchical \
    --embedding-model ViT-B-32

# 出力:
#  - R@1, R@5, MRR（性能指標）
#  - 各クエリの検索結果（clip_id + score）
```

### SOPilot評価（SOP 1セットあたり5分）

```bash
python scripts/sopilot_evaluate_pilot.py \
    --gold-path demo_videos/partner/oilchange_gold_20260101.mp4 \
    --trainee-path demo_videos/partner/oilchange_trainee_001.mp4 \
    --output-dir reports \
    --report-name oilchange_trainee_001

# 生成物:
#  - reports/oilchange_trainee_001_report.pdf
#  - reports/oilchange_trainee_001_report.json
```

---

## 📋 Phase 5: レポート返却（Day 7）

### 返却前の最終確認（10分）

```bash
# 1. PDFが生成されているか
ls reports/*.pdf
# 出力例:
#   oilchange_trainee_001_report.pdf
#   oilchange_trainee_002_report.pdf

# 2. JSONが生成されているか
ls reports/*.json

# 3. 機密ファイルがstageされていないか
git status
# → demo_videos/partner/, chunks/, reports/ が表示されないことを確認
```

### 返却メール送信（10分）

**件名**: `SOPilot評価レポート返却（{SOP名}）`

**本文テンプレート**:
```
お世話になっております。

{SOP名}作業の評価レポートを返却いたします。

### 📊 評価結果サマリー

| 動画 | スコア | 重大逸脱 | 中程度逸脱 | 軽微逸脱 |
|------|--------|----------|------------|----------|
| Trainee 001 | XX/100 | X件 | X件 | X件 |
| Trainee 002 | XX/100 | X件 | X件 | X件 |

### 📎 添付ファイル
- {SOP名}_trainee_001_report.pdf（詳細レポート）
- {SOP名}_trainee_001_report.json（構造化データ）
- {SOP名}_trainee_002_report.pdf
- {SOP名}_trainee_002_report.json

### 🎯 主な逸脱（CRITICAL）
1. Trainee 001: PPE未着用（0:04-0:08）
2. Trainee 002: 工具誤使用（1:23-1:30）

詳細はPDFレポートをご確認ください。

ご不明点があればお気軽にご連絡ください。

よろしくお願いいたします。
```

---

## 🔒 セキュリティ：毎回の確認事項

### コミット前の確認（毎回必須）

```bash
# 1. 機密ファイルが含まれていないか
git status

# 期待される出力（これらが表示されたらNG）:
#   demo_videos/partner/   ← 絶対にNG
#   chunks/                ← 絶対にNG
#   reports/               ← 絶対にNG
#   *_partner.jsonl        ← 絶対にNG

# 2. Pre-commit hookが有効か
cat .git/hooks/pre-commit
# → 存在して実行可能であることを確認

# 3. 万が一stageしてしまったら
git reset HEAD <file>  # 即座にunstage
```

---

## 📊 成功の指標（KPI）

### 回収率（最重要）
- **目標**: 依頼から7日以内にデータ受領
- **測定**: 送信日 → 受領日の日数

### 返却速度
- **目標**: 受領から7営業日以内に返却
- **測定**: 受領日 → 返却日の日数

### 追加依頼回数
- **目標**: 受領後の追加依頼は1回まで
- **測定**: 受領確認メール以降の追加依頼数

---

## 🚨 トラブルシューティング

### Q1: 動画が再生できない
```bash
# ffprobeで確認
ffprobe demo_videos/partner/filename.mp4

# コーデックが非対応の場合は再エンコード
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

### Q2: インデックス作成が遅い（>5分/本）
```bash
# 解像度を下げて再試行
python scripts/index_partner_video.py \
    --video-path <path> \
    --video-id <id> \
    --max-resolution 720  # 1080pではなく720p
```

### Q3: GT作成で迷う（どのclip_idを選ぶ？）
```
A: 最初は「明らかに一致」だけ記入して評価を回す
   → 曖昧なGTは後で追加（完璧主義で止まらない）
```

### Q4: 評価スコアが低すぎる/高すぎる
```
A: 閾値は後で調整可能（まずは1周完走させる）
   → 1周目は「動く」ことが最優先
```

---

## ✅ 完了条件（このチェックリストを終えるタイミング）

- [ ] パートナーからデータを受領
- [ ] 7営業日以内にレポート返却完了
- [ ] 相手から「わかりやすい」「使える」などの肯定的フィードバック
- [ ] 次回以降のデータ提供に合意

→ ここまで来たら、次は「2社目」または「同じ相手でSOP拡大」

---

**次のマイルストーン**: 1社目の1周完走 → 2社目への横展開 → Manufacturing-v1の82クエリ実装
