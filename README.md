# Chapter 07 - RAG チャットボット

Azure OpenAI Service と Azure AI Search を使った RAG (Retrieval-Augmented Generation) チャットボットです。PDF ドキュメントをベクトルインデックスに登録し、その内容に基づいてユーザーの質問に回答します。

## 構成

| ファイル | 概要 |
|---|---|
| `indexer.py` | PDF からテキストを抽出・チャンク分割し、Azure AI Search にベクトルインデックスとして登録する |
| `orchestrator.py` | Streamlit ベースのチャット UI。ユーザーの質問をベクトル検索し、Azure OpenAI で回答を生成する |
| `.env` | Azure リソースへの接続情報 |

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

`.env` ファイルに以下の値を設定してください。

```
SEARCH_SERVICE_ENDPOINT=<Azure AI Search のエンドポイント>
SEARCH_SERVICE_API_KEY=<Azure AI Search の API キー>
SEARCH_SERVICE_INDEX_NAME=<Azure AI Search のインデックス名>
AOAI_ENDPOINT=<Azure OpenAI Service のエンドポイント>
AOAI_API_VERSION=<API バージョン>
AOAI_API_KEY=<Azure OpenAI Service の API キー>
AOAI_EMBEDDING_MODEL_NAME=<埋め込みモデルのデプロイ名>
AOAI_CHAT_MODEL_NAME=<チャットモデルのデプロイ名>
```

## 使い方

### ドキュメントのインデックス登録

PDF ファイルのパスを引数に指定して `indexer.py` を実行します。

```bash
python indexer.py <PDFファイルのパス>
```

### チャットボットの起動

```bash
streamlit run orchestrator.py
```

ブラウザが開き、チャット画面が表示されます。質問を入力すると、インデックス登録済みドキュメントの内容に基づいて回答が生成されます。
# Chapter 07 - RAG チャットボット

Azure OpenAI Service と Azure AI Search を使った RAG (Retrieval-Augmented Generation) チャットボットです。PDF ドキュメントをベクトルインデックスに登録し、その内容に基づいてユーザーの質問に回答します。

## 構成

| ファイル | 概要 |
|---|---|
| `indexer.py` | PDF からテキストを抽出・チャンク分割し、Azure AI Search にベクトルインデックスとして登録する |
| `orchestrator.py` | Streamlit ベースのチャット UI。ユーザーの質問をベクトル検索し、Azure OpenAI で回答を生成する |
| `.env` | Azure リソースへの接続情報 |

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

`.env` ファイルに以下の値を設定してください。

```
SEARCH_SERVICE_ENDPOINT=<Azure AI Search のエンドポイント>
SEARCH_SERVICE_API_KEY=<Azure AI Search の API キー>
SEARCH_SERVICE_INDEX_NAME=<Azure AI Search のインデックス名>
AOAI_ENDPOINT=<Azure OpenAI Service のエンドポイント>
AOAI_API_VERSION=<API バージョン>
AOAI_API_KEY=<Azure OpenAI Service の API キー>
AOAI_EMBEDDING_MODEL_NAME=<埋め込みモデルのデプロイ名>
AOAI_CHAT_MODEL_NAME=<チャットモデルのデプロイ名>
```

## 使い方

### ドキュメントのインデックス登録

PDF ファイルのパスを引数に指定して `indexer.py` を実行します。

```bash
python indexer.py <PDFファイルのパス>
```

### チャットボットの起動

```bash
streamlit run orchestrator.py
```

ブラウザが開き、チャット画面が表示されます。質問を入力すると、インデックス登録済みドキュメントの内容に基づいて回答が生成されます。
