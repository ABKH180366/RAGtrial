# Chapter 07 - 高度RAG チャットボット

Azure OpenAI Service と Azure AI Search を使った RAG (Retrieval-Augmented Generation) チャットボットです。  
この版では、初心者向け最小実装から一歩進めて、**ハイブリッド検索 / 再ランキング / 根拠表示 / 評価 / ログ可観測性**を実装しています。

## 構成

| ファイル | 概要 |
|---|---|
| `indexer.py` | PDF(単体またはフォルダ)を構造ベースでチャンク化し、メタデータ付きで Azure AI Search に登録 |
| `orchestrator.py` | Streamlit チャット UI。ハイブリッド検索 + 再ランキング + 回答生成 + 根拠表示 + メトリクス記録 |
| `rag_core.py` | 共通設定、前処理、検索ユーティリティ、ログ記録処理 |
| `evaluator.py` | オフライン評価（Recall@k, MRR, answer hit rate） |
| `metrics_report.py` | オンラインログから無回答率・遅延・トークン量を集計 |
| `.env.example` | 環境変数テンプレート |

## 前提条件

- Python 3.10 以上
- Azure AI Search リソース（インデックス作成済み）
- Azure OpenAI Service リソース（埋め込みモデル・チャットモデルのデプロイ済み）

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env.example` をコピーして `.env` を作成してください。

```bash
cp .env.example .env
```

主な設定項目:

```
SEARCH_SERVICE_ENDPOINT=<Azure AI Search のエンドポイント>
SEARCH_SERVICE_API_KEY=<Azure AI Search の API キー。Managed Identity利用時は空でも可>
SEARCH_SERVICE_INDEX_NAME=<Azure AI Search のインデックス名>
AOAI_ENDPOINT=<Azure OpenAI Service のエンドポイント>
AOAI_API_VERSION=<API バージョン>
AOAI_API_KEY=<Azure OpenAI Service の API キー>
AOAI_EMBEDDING_MODEL_NAME=<埋め込みモデルのデプロイ名>
AOAI_CHAT_MODEL_NAME=<チャットモデルのデプロイ名>
RAG_DEFAULT_TOP_K=5
RAG_CANDIDATE_POOL_SIZE=20
RAG_MIN_RELEVANCE=0.12
RAG_LOG_PATH=logs/rag_events.jsonl
```

## Azure AI Search インデックス推奨フィールド

高度RAG機能を使うため、インデックスには最低限以下のフィールドを含めてください。

- `id` (key)
- `content` (searchable)
- `contentVector` (vector)
- `doc_id`, `chunk_id`, `source`, `title`, `page`, `section`, `language`, `updated_at` (metadata/filter用)

## 使い方

### ドキュメントのインデックス登録

PDF ファイルのパス、または PDF を含むフォルダパスを指定して `indexer.py` を実行します。

```bash
python indexer.py <PDFファイルまたはフォルダのパス>
```

### チャットボットの起動

```bash
streamlit run orchestrator.py
```

ブラウザが開き、チャット画面が表示されます。  
質問を入力すると、ハイブリッド検索 + 再ランキングされた根拠に基づいて回答が生成されます。  
回答後、画面下部に「参照した根拠チャンク（文書名/ページ/スコア）」が表示されます。

## 評価と可観測性

### オフライン評価

```bash
python evaluator.py --dataset eval_dataset.sample.jsonl --verbose
```

### オンラインログ集計

```bash
python metrics_report.py
```
