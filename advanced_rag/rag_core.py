import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(verbose=True)


@dataclass
class AppConfig:
    search_service_endpoint: str
    search_service_index_name: str
    aoai_endpoint: str
    aoai_api_version: str
    aoai_embedding_model_name: str
    aoai_chat_model_name: str
    search_service_api_key: Optional[str]
    aoai_api_key: Optional[str]
    rag_default_top_k: int
    rag_candidate_pool_size: int
    rag_min_relevance: float
    rag_log_path: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        required = {
            "SEARCH_SERVICE_ENDPOINT": os.environ.get("SEARCH_SERVICE_ENDPOINT"),
            "SEARCH_SERVICE_INDEX_NAME": os.environ.get("SEARCH_SERVICE_INDEX_NAME"),
            "AOAI_ENDPOINT": os.environ.get("AOAI_ENDPOINT"),
            "AOAI_API_VERSION": os.environ.get("AOAI_API_VERSION"),
            "AOAI_EMBEDDING_MODEL_NAME": os.environ.get("AOAI_EMBEDDING_MODEL_NAME"),
            "AOAI_CHAT_MODEL_NAME": os.environ.get("AOAI_CHAT_MODEL_NAME", "gpt-4o-mini"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"必須環境変数が不足しています: {', '.join(missing)}")

        return cls(
            search_service_endpoint=required["SEARCH_SERVICE_ENDPOINT"],
            search_service_index_name=required["SEARCH_SERVICE_INDEX_NAME"],
            aoai_endpoint=required["AOAI_ENDPOINT"],
            aoai_api_version=required["AOAI_API_VERSION"],
            aoai_embedding_model_name=required["AOAI_EMBEDDING_MODEL_NAME"],
            aoai_chat_model_name=required["AOAI_CHAT_MODEL_NAME"],
            search_service_api_key=os.environ.get("SEARCH_SERVICE_API_KEY"),
            aoai_api_key=os.environ.get("AOAI_API_KEY"),
            rag_default_top_k=int(os.environ.get("RAG_DEFAULT_TOP_K", "5")),
            rag_candidate_pool_size=int(os.environ.get("RAG_CANDIDATE_POOL_SIZE", "20")),
            rag_min_relevance=float(os.environ.get("RAG_MIN_RELEVANCE", "0.12")),
            rag_log_path=os.environ.get("RAG_LOG_PATH", "logs/rag_events.jsonl"),
        )


def _get_search_credential(config: AppConfig):
    if config.search_service_api_key:
        return AzureKeyCredential(config.search_service_api_key)

    try:
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential()
    except Exception as exc:
        raise ValueError(
            "SEARCH_SERVICE_API_KEY が未設定で、Managed Identity も利用できません。"
        ) from exc


def build_search_client(config: AppConfig) -> SearchClient:
    return SearchClient(
        endpoint=config.search_service_endpoint,
        index_name=config.search_service_index_name,
        credential=_get_search_credential(config),
    )


def build_openai_client(config: AppConfig) -> AzureOpenAI:
    if not config.aoai_api_key:
        raise ValueError("AOAI_API_KEY が未設定です。")
    return AzureOpenAI(
        azure_endpoint=config.aoai_endpoint,
        api_key=config.aoai_api_key,
        api_version=config.aoai_api_version,
    )


def normalize_text(text: str) -> str:
    normalized = text.replace("\u3000", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def remove_repeated_lines(page_texts: List[str]) -> List[str]:
    line_frequency: Dict[str, int] = {}
    page_lines: List[List[str]] = []
    for page_text in page_texts:
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        page_lines.append(lines)
        for line in set(lines):
            line_frequency[line] = line_frequency.get(line, 0) + 1

    threshold = max(2, int(len(page_texts) * 0.7))
    repeated = {line for line, freq in line_frequency.items() if freq >= threshold}

    cleaned_pages = []
    for lines in page_lines:
        kept = [line for line in lines if line not in repeated]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


def detect_section_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) <= 60:
            return stripped
    return "本文"


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    length = min(len(v1), len(v2))
    dot = sum(v1[i] * v2[i] for i in range(length))
    norm1 = sum(v1[i] * v1[i] for i in range(length)) ** 0.5
    norm2 = sum(v2[i] * v2[i] for i in range(length)) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = {tok for tok in re.split(r"\W+", query.lower()) if len(tok) > 1}
    t_tokens = {tok for tok in re.split(r"\W+", text.lower()) if len(tok) > 1}
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def estimate_top_k(question: str, default_k: int) -> int:
    lowered = question.lower()
    if any(token in lowered for token in ["比較", "違い", "difference", "compare", "一覧"]):
        return min(default_k + 3, 12)
    if len(question) < 24:
        return max(3, default_k - 1)
    return default_k


def build_vector_query(vector: List[float], candidate_pool_size: int) -> VectorizedQuery:
    return VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=candidate_pool_size,
        fields="contentVector",
    )


def safe_doc_id(source_name: str) -> str:
    stem = Path(source_name).stem.lower()
    stem = re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-")
    return stem or f"doc-{uuid.uuid4().hex[:8]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_metric_log(path: str, payload: dict) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": utc_now_iso(), **payload}
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def timed() -> float:
    return time.perf_counter()
