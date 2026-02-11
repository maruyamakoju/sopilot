# 情シス・監査 向けクイックパック

## 提出物
- データ取り扱い仕様: `docs/data_handling_spec.md`
- オフライン展開手順: `docs/offline_deploy.md`
- 監査トレイルAPI: `GET /audit/trail`

## 最小セキュリティ設定
- Token認証:
  - `SOPILOT_API_TOKEN=<secret>`
- Basic認証:
  - `SOPILOT_BASIC_USER=<user>`
  - `SOPILOT_BASIC_PASSWORD=<password>`

## 監査追跡の確認
- ingest/score/training ジョブに `requested_by` を保持
- `job_id` 単位で状態遷移と結果を追跡可能

## データ削除の確認
- `DELETE /videos/{video_id}` で関連ファイルを削除
- 削除後に同タスクの検索インデックスを再構築
