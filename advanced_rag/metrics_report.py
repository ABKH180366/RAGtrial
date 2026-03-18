import json
from pathlib import Path


def load_events(path: str):
    events = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def main():
    path = Path("logs/rag_events.jsonl")
    if not path.exists():
        print("ログファイルがありません: logs/rag_events.jsonl")
        return

    events = load_events(str(path))
    if not events:
        print("イベントがありません。")
        return

    total = len(events)
    no_answer = sum(1 for e in events if e.get("answer_status") == "no_answer")
    answered = total - no_answer
    avg_latency = sum(int(e.get("latency_ms", 0)) for e in events) / total
    avg_retrieved = sum(int(e.get("retrieved", 0)) for e in events) / total
    avg_prompt_tokens = (
        sum(int(e.get("prompt_tokens") or 0) for e in events) / max(answered, 1)
    )
    avg_completion_tokens = (
        sum(int(e.get("completion_tokens") or 0) for e in events) / max(answered, 1)
    )

    report = {
        "total_requests": total,
        "answered": answered,
        "no_answer_rate": no_answer / total,
        "avg_latency_ms": avg_latency,
        "avg_retrieved_docs": avg_retrieved,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
