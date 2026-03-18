import sys
from pathlib import Path
from typing import Iterable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rag_core import (
    AppConfig,
    build_openai_client,
    build_search_client,
    detect_section_title,
    normalize_text,
    remove_repeated_lines,
    safe_doc_id,
    utc_now_iso,
)

# ドキュメント内のテキストをチャンクに分割する際の区切り文字を指定する。
separator = ["\n\n", "\n", "。", "、", " ", ""]

# テキストを指定したサイズで分割する関数を定義する。
def create_chunk(content: str, separators: list, chunk_size: int = 900, overlap: int = 120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=overlap,
        chunk_size=chunk_size,
        separators=separators,
    )
    return splitter.split_text(content)


def extract_pages_from_pdf(filepath: str) -> List[str]:
    print(f"{filepath} のテキストを抽出中...")
    reader = PdfReader(filepath)
    page_texts: List[str] = []
    for page in reader.pages:
        page_texts.append(normalize_text(page.extract_text() or ""))
    return remove_repeated_lines(page_texts)


def build_chunks_for_file(filepath: str) -> List[dict]:
    source_path = Path(filepath)
    source_name = source_path.name
    doc_id = safe_doc_id(source_name)
    page_texts = extract_pages_from_pdf(filepath)

    chunks: List[dict] = []
    seen = set()
    for page_num, page_text in enumerate(page_texts, start=1):
        if not page_text.strip():
            continue
        section = detect_section_title(page_text)
        page_chunks = create_chunk(page_text, separator)
        for chunk_idx, chunk in enumerate(page_chunks):
            text = normalize_text(chunk)
            if len(text) < 40 or text in seen:
                continue
            seen.add(text)
            chunks.append(
                {
                    "id": f"{doc_id}-{page_num:04d}-{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "chunk_id": chunk_idx,
                    "source": source_name,
                    "title": source_path.stem,
                    "page": page_num,
                    "section": section,
                    "language": "ja",
                    "updated_at": utc_now_iso(),
                    "content": text,
                }
            )
    print(f"{source_name}: {len(chunks)} チャンク作成")
    return chunks


def iter_target_files(path_arg: str) -> Iterable[str]:
    target = Path(path_arg)
    if target.is_file():
        yield str(target)
        return
    if target.is_dir():
        for file in sorted(target.glob("*.pdf")):
            yield str(file)
        return
    raise FileNotFoundError(f"対象が見つかりません: {path_arg}")


def index_docs(documents: List[dict], config: AppConfig):
    search_client = build_search_client(config)
    openai_client = build_openai_client(config)

    for i, doc in enumerate(documents, start=1):
        print(f"{i}/{len(documents)}件目: {doc['id']} を処理中...")
        last_error = None
        for attempt in range(1, 4):
            try:
                embedding = openai_client.embeddings.create(
                    input=doc["content"],
                    model=config.aoai_embedding_model_name,
                ).data[0].embedding
                payload = {**doc, "contentVector": embedding}
                search_client.merge_or_upload_documents([payload])
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"  失敗 (attempt {attempt}/3): {exc}")
        if last_error:
            raise RuntimeError(f"インデックス登録に失敗: {doc['id']}") from last_error

if __name__ == "__main__":
    config = AppConfig.from_env()

    if len(sys.argv) < 2:
        print("PDFファイルまたはPDFフォルダのパスを指定してください")
        sys.exit(1)

    target = sys.argv[1]
    all_documents: List[dict] = []
    for filename in iter_target_files(target):
        all_documents.extend(build_chunks_for_file(filename))

    if not all_documents:
        print("インデックス対象のチャンクが見つかりませんでした")
        sys.exit(1)

    index_docs(all_documents, config)
    print(f"インデックス登録完了: {len(all_documents)}件")
