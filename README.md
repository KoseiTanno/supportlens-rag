# SupportLens RAG

> Citation-grounded Japanese GitHub Actions support-assist RAG API.

> **Status: work in progress.** The repository currently contains the ingestion, chunking, lexical-retrieval, API-contract, safety-gate, and deployment foundation. Dense retrieval, reranking, provider wiring, and published evaluation results are not yet complete; do not present this revision as a production-ready demo.

GitHub Actions の日本語公式ドキュメントから、根拠・URL・文書版を伴うサポート回答案を作るRAG APIです。自動返信は行わず、根拠が弱い質問には回答しません。

## Architecture

`allowlisted docs → snapshot → heading-aware chunks → dense + BM25 → RRF → rerank → evidence gate → cited answer`

## Quick start

```bash
cp .env.example .env
docker compose up --build
curl http://localhost:8000/health
```

スナップショット時は `uv run python scripts/snapshot_docs.py --docs-sha <github-docs-commit>` を使用します。`.env`や文書スナップショットは公開しません。

## Evaluation

`Recall@5`、`MRR@10`、`nDCG@10`、レイテンシ、根拠不足時の拒否率を、denseのみ・hybrid・rerankありで比較します。外部モデルの評価値は実測してから掲載します。

## Security

入力と検索文書を信頼できないデータとして扱います。文書中の命令を実行せず、APIキーを環境変数以外に保存せず、根拠が不十分なときは生成モデルを呼び出しません。

## Data attribution

GitHub Docs由来のコンテンツはCC BY 4.0です。対象URLと取得したコミットSHAはスナップショットのマニフェストに保存します。GitHub is not affiliated with or endorsing this project.

## Limitations

これは技術デモであり、GitHub公式製品ではありません。人が回答内容を確認してください。

## License

Project code is [MIT](LICENSE). External documentation remains subject to its original license.
