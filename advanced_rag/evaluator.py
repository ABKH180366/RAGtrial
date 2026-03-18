import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from rag_core import (
    AppConfig,
    build_openai_client,
    build_search_client,
    build_vector_query,
    estimate_top_k,
    lexical_overlap_score,
)


def load_eval_set(path: str) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def retrieve(config: AppConfig, question: str, source_filter: Optional[str]) -> List[dict]:
    search_client = build_search_client(config)
    openai_client = build_openai_client(config)
    question_vector = openai_client.embeddings.create(
        input=question,
        model=config.aoai_embedding_model_name,
    ).data[0].embedding
    vector_query = build_vector_query(question_vector, config.rag_candidate_pool_size)
    top_k = estimate_top_k(question, config.rag_default_top_k)
    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        top=config.rag_candidate_pool_size,
        filter=build_filter(source_filter=source_filter, language_filter="ja"),
        select=["id", "source", "page", "content"],
    )
    candidates = list(results)
    for item in candidates:
        lexical = lexical_overlap_score(question, item.get("content", ""))
        score = float(item.get("@search.score", 0.0))
        item["_score"] = (0.65 * (score / 4.0)) + (0.35 * lexical)
    candidates.sort(key=lambda row: row.get("_score", 0.0), reverse=True)
    return candidates[:top_k]


def score_recall_at_k(retrieved: List[dict], expected_sources: List[str]) -> float:
    if not expected_sources:
        return 0.0
    src_set = {item.get("source") for item in retrieved}
    hit = any(source in src_set for source in expected_sources)
    return 1.0 if hit else 0.0


def score_mrr(retrieved: List[dict], expected_sources: List[str]) -> float:
    expected = set(expected_sources)
    for rank, item in enumerate(retrieved, start=1):
        if item.get("source") in expected:
            return 1.0 / rank
    return 0.0


def score_answer_hit(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 0.0
    lowered = answer.lower()
    matched = sum(1 for keyword in expected_keywords if keyword.lower() in lowered)
    return matched / len(expected_keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="jsonl形式の評価データ")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = AppConfig.from_env()
    openai_client = build_openai_client(config)
    rows = load_eval_set(args.dataset)
    if not rows:
        raise ValueError("評価データが空です。")

    recall_sum = 0.0
    mrr_sum = 0.0
    answer_hit_sum = 0.0

    for row in rows:
        question = row["question"]
        source_filter = row.get("source_filter")
        expected_sources = row.get("expected_sources", [])
        expected_keywords = row.get("expected_keywords", [])
        retrieved = retrieve(config, question, source_filter)

        source_lines = []
        for i, item in enumerate(retrieved[:5], start=1):
            source_lines.append(f"[Source{i}] {item.get('source')} p.{item.get('page')}\n{item.get('content')}")
        prompt = f"質問:\n{question}\n\nSources:\n" + "\n\n".join(source_lines)

        completion = openai_client.chat.completions.create(
            model=config.aoai_chat_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Sourcesのみを根拠に簡潔に答えてください。根拠不十分なら『すみません。わかりません。』と返してください。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        answer = completion.choices[0].message.content or ""
        recall = score_recall_at_k(retrieved, expected_sources)
        mrr = score_mrr(retrieved, expected_sources)
        answer_hit = score_answer_hit(answer, expected_keywords)
        recall_sum += recall
        mrr_sum += mrr
        answer_hit_sum += answer_hit
        if args.verbose:
            print(f"Q: {question}")
            print(f"  recall@k={recall:.2f}, mrr={mrr:.2f}, answer_hit={answer_hit:.2f}")

    n = len(rows)
    report: Dict[str, float] = {
        "samples": float(n),
        "recall_at_k": recall_sum / n,
        "mrr": mrr_sum / n,
        "answer_hit_rate": answer_hit_sum / n,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
