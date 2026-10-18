# SupportLens RAG

> 日本語の GitHub Actions 公式ドキュメントを根拠付きで探索し、サポート担当者向けの回答案を返す RAG API

SupportLens は「もっともらしい回答」ではなく、回答可能性・引用・文書バージョンを API 契約に含めることを重視したポートフォリオです。自動返信ボットではなく、利用者が根拠リンクを確認してから応答する運用を想定します。

## What it demonstrates

- 許可リスト付きの公式文書スナップショットと SHA によるデータ来歴
- 見出し階層を保持した分割、FastEmbed による日本語向けローカル埋め込み、Qdrant
- dense / BM25 / RRF hybrid 検索と、任意のローカル cross-encoder reranking
- 根拠が弱い場合に生成を止める evidence gate と、引用を必須にするレスポンス契約
- OpenAI キーなしでも動く抽出モード。キーを設定した時だけ生成モードを有効化

## Architecture

`allowlisted docs → snapshot → heading-aware chunks → dense + BM25 → RRF → rerank → evidence gate → cited answer`

## Quick start

```bash
cp .env.example .env
docker compose up --build -d
DOCS_SHA=$(git ls-remote https://github.com/github/docs.git HEAD | cut -f1)
docker compose exec api python scripts/snapshot_docs.py --docs-sha "$DOCS_SHA" --output data/snapshot
docker compose exec api python scripts/index_corpus.py
docker compose restart api
curl http://localhost:8000/health
```

初期状態の `/health` は `{"status":"not_ready"}` です。空の知識ベースを「正常」と偽装しないためで、準備後は API コンテナを再起動してください。

準備後は [http://localhost:8000](http://localhost:8000) を開くと、非エンジニア向けの質問画面を使えます。自然な日本語で質問すると、回答と参照した公式ドキュメントへのリンクを表示します。質問例の選択、キーボード送信（`⌘/Ctrl + Enter`）、待機・根拠不足・起動前エラーの各状態にも対応しています。

### 無料で自然な回答を試す（Ollama）

Ollama を入れてモデルを取得すれば、API キーなしでローカル生成を使えます。

```bash
ollama pull qwen2.5:3b
SUPPORTLENS_LLM_PROVIDER=ollama docker compose up -d --build
```

Docker Desktop では `host.docker.internal` 経由でローカルの Ollama に接続します。未設定時は外部 API を呼ばず、最上位の根拠抜粋を返す抽出モードです。OpenAI を使う場合だけ、`.env` に `SUPPORTLENS_OPENAI_API_KEY` を設定し、`SUPPORTLENS_LLM_PROVIDER=openai` を指定してください。

`SUPPORTLENS_ENABLE_RERANKER=true` は多言語 cross-encoder を追加で取得して rerank を有効にします。CPU・ディスクを使うため既定では無効で、無効時の `hybrid_rerank` は hybrid 検索にフォールバックします。

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | 索引の準備状態 |
| `POST /v1/search` | チャンク、順位、スコアを取得 |
| `POST /v1/answers` | `answered` または `insufficient_evidence` を返却 |

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
