# データ取り扱い仕様（PoC向け）

## 1. 保存場所
- 動画: `data/raw/`
- 埋め込み/メタ: `data/embeddings/`
- レポート: `data/reports/`
- メタDB: `data/sopilot.db`

## 2. 外部送信
- 既定運用はオンプレ完結。
- 外部通信は管理者が明示的に許可した場合のみ（モデル取得など）。

## 3. 保存期間（推奨）
- 生動画: 30〜90日
- レポート: 1年
- 監査ログ: 1年以上

## 4. 削除手順
- 単体削除: `DELETE /videos/{video_id}`
- 定期削除: 管理者が期限超過データを抽出し、APIで削除
- 削除後は同タスクのインデックスを再構築

## 5. アクセス制御
- `SOPILOT_API_TOKEN` または `SOPILOT_BASIC_USER/PASSWORD` で認証を有効化
- 誰がジョブを実行したかは `requested_by` で保持
