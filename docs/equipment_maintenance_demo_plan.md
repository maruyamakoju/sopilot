# 設備保全デモ計画（1タスク固定）

## 対象タスク
- `task_id`: `maintenance_filter_swap`
- 作業: フィルタ交換（取り外し -> 清掃 -> 交換 -> 確認）

## 目的
- UIで「逸脱ジャンプ -> 根拠確認 -> PDF出力」を3分以内で実演する

## デモ用データ構成
- Gold: 1本（手順順守）
- Trainee失敗例: 5本
  - `missing`: 清掃ステップ欠落
  - `swap`: 交換と清掃の順序入替
  - `deviation`: 工具の使い方が異なる
  - `time_over`: 不自然な長時間停止
  - `mixed`: 軽微逸脱の複合

## 実施手順
1. Gold/Traineeをアップロードし、`/videos/jobs/{id}` で完了待ち
2. `/score` で採点ジョブ作成
3. `/score/{id}` で逸脱一覧確認
4. UIで逸脱マーカーをクリックし再生位置ジャンプ
5. `/score/{id}/report.pdf` をダウンロード
6. `/audit/trail` で実行履歴を確認

CLI自動実行:
- `python scripts/run_demo_flow.py --base-url http://127.0.0.1:8000 --token demo-token --task-id maintenance_filter_swap --gold demo_videos/maintenance_filter_swap/gold.mp4 --trainee demo_videos/maintenance_filter_swap/missing.mp4 --out-dir demo_artifacts`

## 成功条件（営業デモ）
- 逸脱タイプを3種類説明できる
- 逸脱区間の映像比較を即提示できる
- PDFと監査トレイルを同時に提示できる
