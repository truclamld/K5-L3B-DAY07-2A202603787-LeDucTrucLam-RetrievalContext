"""Run a real model-backed agent over a converted Day 6 question file."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .agentic import OpenAIController, run_agent
from .retrieval import Day6Retriever, read_records


class DualRetriever:
    """Expose dense vector and TF-IDF search as two agent-selectable tools."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.tools = {
            "vector_search": Day6Retriever("dense", chunks=chunks),
            "keyword_search": Day6Retriever("tfidf", chunks=chunks),
        }

    def search(self, query: str, *, filters: dict[str, Any] | None,
               top_k: int, tool: str) -> list[dict[str, Any]]:
        if tool not in self.tools:
            raise ValueError(f"Unsupported retrieval tool: {tool}")
        return self.tools[tool].search(query, filters=filters, top_k=top_k)


def _public_question(row: dict[str, Any]) -> dict[str, Any]:
    """Remove gold and reference answers before constructing model inputs."""
    return {key: row[key] for key in ("id", "question", "filter") if key in row}


def run(chunks_path: Path, questions_path: Path, *, split: str, method: str,
        model: str | None, max_searches: int, top_k: int, budget: int,
        output: Path) -> dict[str, Any]:
    chunks = read_records(chunks_path)
    questions = [q for q in read_records(questions_path) if q.get("split") == split]
    if not questions:
        raise ValueError(f"Không có câu hỏi trong split {split}")
    controller = OpenAIController(model)
    if method != "dual":
        raise ValueError("Agent runner hiện dùng hai tool: vector_search và keyword_search (method='dual')")
    index_started = time.perf_counter()
    retriever = DualRetriever(chunks)
    index_build_ms = round((time.perf_counter() - index_started) * 1000, 2)
    traces = []
    cases = []
    for row in questions:
        q = _public_question(row)
        answerable = row.get("answerable", bool(row.get("gold")))
        cases.append({"id": row["id"], "split": split, "answerable": bool(answerable)})
        started = time.perf_counter()
        usage_before = dict(controller.usage)
        try:
            result = run_agent(q["question"], controller, retriever,
                filters=q.get("filter"), max_searches=max_searches,
                top_k=top_k, budget=budget)
            status = result["status"]
            record = {"query_id": q["id"], **result}
        except Exception as exc:
            # Keep the failed query and continue so one provider error doesn't
            # erase already collected evidence. No synthetic answer is added.
            status = "error"
            record = {"query_id": q["id"], "status": status, "answer": None,
                      "citations": [], "context": "", "context_ids": [],
                      "events": getattr(exc, "agent_events", []) +
                                [{"event": "error", "error": f"{type(exc).__name__}: {exc}"}]}
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        record["model"] = controller.model
        record["usage_cumulative"] = dict(controller.usage)
        # This per-case value is useful for cost accounting; prices depend on
        # provider/model and are intentionally not guessed.
        record["usage_delta"] = {
            key: controller.usage[key] - usage_before[key]
            for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens")
        }
        traces.append(record)
        print(f"{q['id']}: {status} ({record['elapsed_ms']} ms)", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tool_call_counts = {"vector_search": 0, "keyword_search": 0}
    calls_per_query: dict[str, int] = {}
    for trace in traces:
        count = 0
        for event in trace.get("events", []):
            if event.get("event") not in ("search", "search_error"):
                continue
            count += 1
            if event.get("tool") in tool_call_counts:
                tool_call_counts[event["tool"]] += 1
        calls_per_query[str(count)] = calls_per_query.get(str(count), 0) + 1
    report = {
        "split": split, "method": method, "model": controller.model,
        "tools": {"vector_search": "dense cosine", "keyword_search": "TF-IDF"},
        "embedding_model": getattr(retriever.tools["vector_search"].embedder, "model_name", None),
        "embedding_dimensions": getattr(retriever.tools["vector_search"].embedder, "dimensions", None),
        "index_build_ms": index_build_ms,
        "config": {"max_searches": max_searches, "top_k": top_k, "context_budget_characters": budget},
        "questions": len(questions), "status_counts": {
            key: sum(t["status"] == key for t in traces) for key in ("answered", "abstain", "error")},
        "elapsed_ms_total": round(sum(t["elapsed_ms"] for t in traces), 2),
        "tool_call_counts": tool_call_counts,
        "calls_per_query": calls_per_query,
        "model_usage": controller.usage,
        "cost": None,
        "cost_note": "Chưa ước tính: cần bảng giá đúng provider/model; usage token của agent được lưu trong traces, thời gian index/embedding được ghi riêng.",
        "trace_file": "traces.json", "cases_file": "cases.json",
    }
    (output / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Chạy agent thật và ghi trace để audit")
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--method", choices=("dual",), default="dual", help="Hai tool: dense vector và TF-IDF keyword")
    p.add_argument("--model", help="Mặc định lấy AGENT_MODEL từ .env")
    p.add_argument("--max-searches", type=int, default=3)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--budget", type=int, default=5000, help="Ngân sách context tính theo ký tự")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = run(a.chunks, a.questions, split=a.split, method=a.method,
        model=a.model, max_searches=a.max_searches, top_k=a.top_k,
        budget=a.budget, output=a.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
