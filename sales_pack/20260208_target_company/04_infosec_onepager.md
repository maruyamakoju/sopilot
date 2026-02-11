# SOPilot 情シス突破セット（1ページ）

## 1. データ保持
- 生動画: `data/raw/`
- 埋め込み/メタ: `data/embeddings/`
- レポート: `data/reports/`
- メタDB: `data/sopilot.db`

## 2. 外部送信方針
- 基本はオンプレ完結
- 外部通信は管理者が明示許可した場合のみ
- オフライン運用時は `SOPILOT_VJEPA2_SOURCE=local` を使用

## 3. 認証
- Bearer token: `SOPILOT_API_TOKEN`
- Basic auth: `SOPILOT_BASIC_USER` + `SOPILOT_BASIC_PASSWORD`
- `/health` を除き認証を要求（設定時）

## 4. 監査トレイル
- `GET /audit/trail` で履歴参照
- ingest/score/training の各ジョブに `requested_by` を記録
- 追跡単位は `job_id`

## 5. 削除
- `DELETE /videos/{video_id}` で動画と関連成果物を削除
- 削除後、同 `task_id` の検索インデックスを再構築

## 6. 提出可能資料
- データ取り扱い: `docs/data_handling_spec.md`
- オフライン展開: `docs/offline_deploy.md`
- PoCパッケージ: `docs/paid_poc_fixed_package.md`
