# パートナーデータ送付手順

**日付**: 2026-02-16
**目的**: PARTNER_DATA_REQUEST.mdの送付と受領オペレーション確定

---

## 送付メッセージ（コピペ用）

```
件名: SOPilot製造業パイロット - 動画データ提供依頼

お世話になっております。

SOPilot 製造業パイロットのため、動画データ（最小セット）のご提供をお願いできますか。

こちらは動画をリポジトリに入れず、ローカルでのみ評価します（匿名化前提・機密扱い）。

必要な内容は添付の通りで、受領後は **逸脱（欠落/順序/重大違反）をタイムスタンプ付きでレポート（JSON/PDF）** として返します。

詳細は添付の PARTNER_DATA_REQUEST.md をご確認ください。

---

【データ受け渡し方法】

以下のいずれかでご提供ください：
- Google Drive（共有リンク）
- Box（共有リンク）
- S3バケット（アクセス権限付与）
- 暗号化zip（パスワードは別経路で送付）

【ファイル命名規約】

以下の形式でお願いします：
{SOP名}_{役割}_{日付}.mp4

例：
- oilchange_gold_202602.mp4
- oilchange_trainee1_202602.mp4
- oilchange_trainee2_202602.mp4
- tirechange_gold_202602.mp4
- tirechange_trainee1_202602.mp4

【手順リスト】

各SOPの手順リスト（テキストまたはExcel）も併せてご提供ください。
形式：
- SOP名
- ステップ番号
- ステップ名
- 重大逸脱の定義（該当する場合）

【納期】

データご提供から約1週間でレポート（JSON/PDF）を返却します。

ご不明点がございましたらお気軽にお問い合わせください。

よろしくお願いいたします。
```

**添付ファイル**: `PARTNER_DATA_REQUEST.md`

---

## 受領方法の選択肢

### 推奨順

1. **Google Drive / Box**（最短）
   - 利点: パートナー側の操作が簡単、共有リンクですぐ受領
   - 欠点: 大容量は時間かかる

2. **S3バケット**（大容量向け）
   - 利点: 大容量（数GB）でも高速、アクセス制御明確
   - 欠点: パートナー側にAWSアカウント必要

3. **暗号化zip**（セキュリティ重視）
   - 利点: 最もセキュア（パスワード別経路）
   - 欠点: 容量制限（メール添付不可、別途アップロード必要）

---

## ファイル命名規約（詳細）

### 形式
```
{SOP名}_{役割}_{日付}.mp4
```

### 役割
- `gold`: Gold standard（模範動画）
- `trainee1`, `trainee2`, ...: 訓練者動画（複数可）

### 日付
- YYYYMM形式（例：202602）

### 例
```
oilchange_gold_202602.mp4          # オイル交換 Gold
oilchange_trainee1_202602.mp4      # オイル交換 Trainee 1
oilchange_trainee2_202602.mp4      # オイル交換 Trainee 2
tirechange_gold_202602.mp4         # タイヤ交換 Gold
tirechange_trainee1_202602.mp4     # タイヤ交換 Trainee 1
brakepad_gold_202602.mp4           # ブレーキパッド Gold
brakepad_trainee1_202602.mp4       # ブレーキパッド Trainee 1
```

---

## 受領後のチェックリスト

### 即座に実行（受領後30分以内）

- [ ] ファイル命名規約チェック（`validate_partner_videos.py`で自動化）
- [ ] 動画の再生可能性確認
- [ ] 解像度・fps・再生時間の記録
- [ ] 音声トラックの有無確認
- [ ] シーン分割数の概算（PySceneDetect）

### 受領後24時間以内

- [ ] `video_paths.local.json`にマッピング追加
- [ ] 手順リストから`manufacturing_v1.jsonl`のステップ定義作成
- [ ] Gold動画でindexing実行（`--hierarchical --embedding-model ViT-H-14`）
- [ ] Trainee動画で逸脱検出実行（`sopilot_evaluate_pilot.py`）

### 受領後1週間以内

- [ ] 全SOP × 全Trainee動画の評価完了
- [ ] レポート生成（JSON/PDF）
- [ ] パートナーへのフィードバック送付

---

## 次のアクション

1. **即座**: 上記メッセージをパートナーに送付（メール/Slack等）
2. **並行**: 受領データ検品スクリプト作成（`validate_partner_videos.py`）
3. **実データ受領後**: チェックリストに従って運用開始

---

**ステータス**: 📧 **送付準備完了、パートナー連絡待ち**
