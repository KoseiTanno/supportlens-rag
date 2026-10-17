# SupportLens RAG 設計仕様

## 1. 目的

SupportLens RAG は、GitHub Actions の日本語公式ドキュメントを根拠に、サポート担当者向けの回答案を生成する検索拡張生成（RAG）API である。利用者への自動送信、ワークフローの自動実行、または GitHub の公式製品であることを示唆する機能は提供しない。

このリポジトリの目的は、再現可能な日本語 RAG システムに必要な検索、評価、引用、セキュリティ、運用上の判断をポートフォリオとして示すことである。

## 2. 利用者と成功条件

主な利用者は、開発者向け SaaS のサポート担当者または技術支援を行うエンジニアである。質問に対して、回答案と検証可能な一次情報の引用を返す。

成功条件は次のとおり。

- `docker compose up` と取り込みコマンドだけでローカル再現できる。
- 回答には常に少なくとも 1 件の引用（URL、見出し、本文抜粋）を付ける。
- 検索根拠が不十分な質問は推測せず、`insufficient_evidence` を返す。
- dense 単独、ハイブリッド検索、reranking ありの 3 構成を同じ評価セットで比較できる。
- CI で静的検査、テスト、イメージビルドを確認する。

## 3. スコープ

### 含むもの

- GitHub Docs のうち、GitHub Actions の基本操作、ワークフロー構文、Secrets、トラブルシューティングに関する日本語 Markdown 文書
- 文書のスナップショット、ライセンス、取得コミット SHA、出典 URL の記録
- dense 検索、BM25 検索、Reciprocal Rank Fusion（RRF）、reranker
- FastAPI による検索・回答 API
- Docker Compose、pytest、GitHub Actions CI、評価レポート

### 含まないもの

- GitHub API の書き込み、ワークフロー実行、ユーザー認証
- 実運用の顧客データ、個人情報、リアルタイムで変わる障害情報
- エンドユーザーへ自動送信するチャット UI
- GitHub Docs 全体のクロール

## 4. データとライセンス

データソースは `github/docs` リポジトリの日本語コンテンツに限定する。コンテンツのライセンスは CC BY 4.0、コードは MIT として公開されている。取り込み対象、取得元コミット SHA、取得日時、元 URL、ライセンス、必要な帰属表記をマニフェストに保存する。

取り込む文書はリポジトリ上のスナップショットとして扱い、評価結果を再現できるようにする。取得・変換スクリプトは、許可リストにあるパスだけを処理する。画像、商標、第三者コンテンツは取り込まない。

## 5. アーキテクチャ

```text
GitHub Docs (Japanese Markdown)
  -> snapshot + manifest
  -> Markdown normalization / heading-aware chunking
  -> dense embedding + BM25 index
  -> Qdrant

question
  -> validation and injection screening
  -> dense top-20 + BM25 top-20
  -> RRF fusion
  -> rerank top-10
  -> evidence gate
  -> LLM answer constrained to cited context
  -> answer, citations, confidence, status, trace_id
```

### コンポーネント境界

| コンポーネント | 責務 | 依存先 |
| --- | --- | --- |
| `ingestion` | 文書取得、正規化、分割、マニフェスト作成 | ローカル文書、Qdrant |
| `retrieval` | dense/BM25 検索、RRF、reranking | Qdrant、埋め込みモデル |
| `generation` | 根拠に限定した回答生成 | LLM アダプタ |
| `api` | リクエスト検証、レスポンス、ヘルスチェック | retrieval、generation |
| `evaluation` | 指標計算、構成比較、結果出力 | retrieval、評価データ |

## 6. 検索・生成の詳細

1. 入力は Pydantic で検証し、空文字・過大な文字列を拒否する。
2. 質問を dense と BM25 の両方で最大 20 件取得する。
3. RRF で統合し、上位 10 件を多言語 reranker に渡す。
4. rerank 後の上位 3 件について、関連性しきい値と多様性を確認する。
5. evidence gate を通った場合だけ、引用済みチャンクを LLM に渡す。
6. LLM は与えられた根拠だけで回答し、根拠にない事項は不明とする。
7. evidence gate を通らない場合は LLM を呼び出さず、`insufficient_evidence` を返す。

RRF は異なるスコア尺度を直接加算しないための初期値として採用する。候補数、しきい値、reranker の導入効果は評価セット上で測定して決める。

## 7. API 契約

### `POST /v1/answers`

入力は `question`、任意の `top_k`、任意の `mode`（`hybrid_rerank` が既定）とする。レスポンスは次を含む。

```json
{
  "status": "answered",
  "answer": "回答案",
  "citations": [
    {
      "title": "見出し",
      "url": "https://docs.github.com/ja/...",
      "excerpt": "根拠の抜粋",
      "document_sha": "..."
    }
  ],
  "confidence": "high",
  "trace_id": "uuid"
}
```

根拠不足時には `status: "insufficient_evidence"`、空の `answer`、補足の `reason` を返す。

### その他のエンドポイント

- `POST /v1/search`: 生成せず検索順位、チャンク、スコアを返す。
- `GET /v1/documents`: 取り込み済み文書と版情報を返す。
- `GET /health`: API、Qdrant、設定の健全性を返す。

## 8. 評価

評価セットは、回答可能な質問 40 件と回答不可能な質問 10 件を含む。各回答可能な質問に少なくとも 1 つの正解チャンク ID を付ける。

比較する構成は、dense のみ、dense + BM25 + RRF、hybrid + reranking の 3 種類とする。検索では Recall@5、MRR@10、nDCG@10、平均・p95 レイテンシを測る。生成では引用の有無、引用が主張を支えるか、根拠不足時に拒否した割合を集計する。

モデルや候補数を評価セット全体に合わせ込まない。調整用と報告用に質問を分割し、README には評価日、文書版、構成、限界を明記する。

## 9. テストと CI

- 単体テスト: Markdown 分割、RRF、引用整形、拒否判定、設定
- 結合テスト: Qdrant を使った検索経路
- API テスト: 正常系、入力上限、未知質問、Qdrant 停止時
- 評価テスト: 固定コーパスでの指標計算
- CI: Ruff、mypy、pytest、Docker イメージビルド

外部 LLM API はテストではスタブ化する。実 API を使う評価は明示的なローカルコマンドでのみ実行する。

## 10. セキュリティと失敗時の動作

- API キーは環境変数にのみ保存し、`.env` は Git 管理しない。
- 質問と検索文書を信頼できないデータとして扱い、文書中の命令を実行しない。
- 質問の文字数、取得候補数、API 呼び出し回数を制限する。
- 構造化ログに質問原文・認証情報・全文コンテキストを保存しない。
- Qdrant または生成モデルが停止した場合、内部詳細を返さず `503` と trace ID を返す。
- README に、プロトタイプであり GitHub 非公式、最終判断は人が行うことを明記する。

## 11. 公開物

リポジトリには MIT ライセンス、README、アーキテクチャ図、Docker Compose、`.env.example`、評価データのスキーマと集計済み結果、出典・ライセンス一覧を含める。外部 API の秘密情報、取得済みの大量原文、評価用の回答キーは含めない。

README は日本語を主、冒頭に英語要約を置く。課題、設計判断、起動方法、API 例、評価結果、セキュリティ、データ帰属、制約を説明する。

## 12. 採用しない選択肢と理由

- 単一ベクトル検索: 正確な識別子や設定キーの検索が弱いため採用しない。
- LLM だけの回答: 引用と再現可能な評価を担保できないため採用しない。
- 常時最新の Web クロール: 評価の再現性と利用条件の追跡を損なうため採用しない。
- 自動返信: 誤回答の影響が大きく、ポートフォリオの第1版の範囲を超えるため採用しない。
