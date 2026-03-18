from typing import Dict, List, Optional, Tuple

import streamlit as st
from rag_core import (
    AppConfig,
    append_metric_log,
    build_openai_client,
    build_search_client,
    build_vector_query,
    cosine_similarity,
    estimate_top_k,
    lexical_overlap_score,
    timed,
)

# AIのキャラクターを決めるためのシステムメッセージを定義する。
system_message_chat_conversation = """
あなたは根拠ベースで回答するアシスタントです。
必ず「Sources:」に含まれる情報のみで回答してください。
根拠が不十分、またはSourcesに回答に必要な情報がない場合は、必ず「すみません。わかりません。」と回答してください。
推測や一般知識で補完しないでください。
回答は簡潔に、箇条書きを優先して作成してください。
回答文中に [Source1] のような参照タグは出力しないでください。
"""

def format_chat_history(history: List[dict], max_turns: int = 4) -> str:
    recent = history[-max_turns * 2 : -1]
    lines = []
    for item in recent:
        role = "ユーザー" if item["role"] == "user" else "アシスタント"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


def rerank_results(question: str, question_vector: List[float], candidates: List[dict]) -> List[dict]:
    reranked = []
    for candidate in candidates:
        vector_score = cosine_similarity(question_vector, candidate.get("contentVector", []))
        lexical_score = lexical_overlap_score(question, candidate.get("content", ""))
        search_score = float(candidate.get("@search.score", 0.0))
        final_score = (0.55 * vector_score) + (0.25 * lexical_score) + (0.20 * search_score / 4.0)
        reranked.append({**candidate, "_score": final_score})
    reranked.sort(key=lambda item: item["_score"], reverse=True)
    return reranked


def build_filter(source_filter: Optional[str], language_filter: Optional[str]) -> Optional[str]:
    clauses = []
    if source_filter:
        escaped_source = source_filter.replace("'", "''")
        clauses.append(f"source eq '{escaped_source}'")
    if language_filter:
        escaped_language = language_filter.replace("'", "''")
        clauses.append(f"language eq '{escaped_language}'")
    if not clauses:
        return None
    return " and ".join(clauses)


def search(history: List[dict], source_filter: Optional[str], language_filter: Optional[str]) -> Tuple[str, List[dict], Dict]:
    config = AppConfig.from_env()
    question = history[-1].get("content", "").strip()
    if not question:
        return "すみません。わかりません。", [], {}

    search_client = build_search_client(config)
    openai_client = build_openai_client(config)
    start = timed()

    embedding_response = openai_client.embeddings.create(
        input=question,
        model=config.aoai_embedding_model_name,
    )
    question_vector = embedding_response.data[0].embedding

    vector_query = build_vector_query(question_vector, config.rag_candidate_pool_size)
    top_k = estimate_top_k(question, config.rag_default_top_k)
    filter_expression = build_filter(source_filter, language_filter)

    # search_text を併用することでハイブリッド検索にする。
    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        top=config.rag_candidate_pool_size,
        filter=filter_expression,
        select=[
            "id",
            "doc_id",
            "chunk_id",
            "source",
            "title",
            "page",
            "section",
            "language",
            "updated_at",
            "content",
            "contentVector",
        ],
    )
    candidates = list(results)
    ranked = rerank_results(question, question_vector, candidates)
    selected = ranked[:top_k]

    if not selected or selected[0]["_score"] < config.rag_min_relevance:
        latency_ms = int((timed() - start) * 1000)
        append_metric_log(
            config.rag_log_path,
            {
                "question": question,
                "retrieved": len(selected),
                "latency_ms": latency_ms,
                "answer_status": "no_answer",
                "top_score": selected[0]["_score"] if selected else 0.0,
            },
        )
        return "すみません。わかりません。", selected, {"latency_ms": latency_ms}

    sources = []
    for idx, result in enumerate(selected, start=1):
        source_header = f"[Source{idx}] {result.get('source', 'unknown')} p.{result.get('page', '-')}"
        sources.append(f"{source_header}\n{result.get('content', '')}")
    source_block = "\n\n".join(sources)

    conversation_summary = format_chat_history(history)
    user_message = f"""
質問:
{question}

会話履歴:
{conversation_summary if conversation_summary else "(なし)"}

Sources:
{source_block}
"""
    chat_response = openai_client.chat.completions.create(
        model=config.aoai_chat_model_name,
        messages=[
            {"role": "system", "content": system_message_chat_conversation},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    answer = chat_response.choices[0].message.content or "すみません。わかりません。"
    usage = getattr(chat_response, "usage", None)
    latency_ms = int((timed() - start) * 1000)
    append_metric_log(
        config.rag_log_path,
        {
            "question": question,
            "retrieved": len(selected),
            "latency_ms": latency_ms,
            "answer_status": "answered" if answer != "すみません。わかりません。" else "no_answer",
            "top_score": selected[0]["_score"],
            "source_filter": source_filter,
            "language_filter": language_filter,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        },
    )
    return answer, selected, {"latency_ms": latency_ms}

# ここからは画面を構築するためのコード
# チャット履歴を初期化する。
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_sources" not in st.session_state:
    st.session_state["last_sources"] = []
if "last_metrics" not in st.session_state:
    st.session_state["last_metrics"] = {}

st.sidebar.header("検索オプション")
source_filter_ui = st.sidebar.text_input("source フィルタ（任意）", value="")
language_filter_ui = st.sidebar.selectbox("language フィルタ", options=["", "ja", "en"], index=0)
language_filter = language_filter_ui or None
source_filter = source_filter_ui.strip() or None

# チャット履歴を表示する。
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ユーザーが質問を入力したときの処理を記述する。
if prompt := st.chat_input("質問を入力してください"):

    # ユーザーが入力した質問を表示する。
    with st.chat_message("user"):
        st.write(prompt)

    # ユーザの質問をチャット履歴に追加する
    st.session_state.history.append({"role": "user", "content": prompt})

    # ユーザーの質問に対して回答を生成するためにsearch関数を呼び出す。
    response, selected_sources, metrics = search(st.session_state.history, source_filter, language_filter)
    st.session_state["last_sources"] = selected_sources
    st.session_state["last_metrics"] = metrics

    # 回答を表示する。
    with st.chat_message("assistant"):
        st.write(response)

    # 回答をチャット履歴に追加する。
    st.session_state.history.append({"role": "assistant", "content": response})

if st.session_state["last_sources"]:
    with st.expander("参照した根拠チャンク"):
        for idx, item in enumerate(st.session_state["last_sources"], start=1):
            st.markdown(
                f"**Source {idx}**: `{item.get('source', 'unknown')}` / page `{item.get('page', '-')}` / score `{item.get('_score', 0):.3f}`"
            )
            st.write(item.get("content", ""))

if st.session_state["last_metrics"]:
    st.caption(f"推論時間: {st.session_state['last_metrics'].get('latency_ms', '-')} ms")
