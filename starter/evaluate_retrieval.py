"""Compare Day 6 dense, TF-IDF, and RRF retrieval on Day 7 dev questions."""
import argparse
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import json

# When invoked as `python3 starter/evaluate_retrieval.py`, Python places the
# starter directory (rather than the project root) on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.baseline import hit, pack_context
from starter.retrieval import Day6Retriever, read_records
from starter.day06_starter.src.embeddings import MyEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, default=Path("starter/results/retrieval-dev"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-k-improved", type=int, default=5,
                        help="Larger hybrid result set for the top-k experiment")
    parser.add_argument("--rerank-candidates", type=int, default=10,
                        help="RRF candidate pool size for dense-score reranking")
    parser.add_argument("--context-budget", type=int, default=5000,
                        help="Maximum packed context size in characters")
    args = parser.parse_args()
    chunks = read_records(args.chunks)
    questions = [q for q in read_records(args.questions) if q.get("split") == args.split]
    if not questions:
        raise ValueError(f"No {args.split} questions in {args.questions}")
    if args.top_k <= 0 or args.top_k_improved <= args.top_k:
        raise ValueError("Require 0 < --top-k < --top-k-improved")
    if args.rerank_candidates < args.top_k:
        raise ValueError("--rerank-candidates must be at least --top-k")
    if args.context_budget <= 0:
        raise ValueError("--context-budget must be positive")

    traces, summaries = [], {}
    # Embed local documents once and reuse their vectors for dense and hybrid.
    env_path = os.getenv("DAY6_ENV_FILE", "starter/.env")
    load_dotenv(env_path)
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not model:
        raise ValueError(f"Set EMBEDDING_MODEL in {env_path}")
    embedder = MyEmbedder(model)
    document_vectors = []
    for start in range(0, len(chunks), 64):
        document_vectors.extend(embedder.embed_documents(
            [chunk["text"] for chunk in chunks[start:start + 64]]))
    # Embed every query once. Reuse the same vector across configurations, while
    # including that shared API latency in dense/hybrid query-time metrics.
    query_vectors: dict[str, list[float]] = {}
    query_embedding_ms: dict[str, float] = {}
    for question in questions:
        qtext = question["question"]
        started = time.perf_counter()
        query_vectors[qtext] = embedder.embed_query(qtext)
        query_embedding_ms[qtext] = round((time.perf_counter() - started) * 1000, 2)
    retrievers = {
        method: Day6Retriever(method, chunks=chunks, embedder=embedder,
                              document_vectors=document_vectors, query_vectors=query_vectors)
        for method in ("dense", "tfidf", "hybrid")
    }
    configurations = [
        ("dense", "dense", args.top_k, None, None),
        ("tfidf", "tfidf", args.top_k, None, None),
        ("hybrid", "hybrid", args.top_k, None, None),
        (f"hybrid_top{args.top_k_improved}", "hybrid", args.top_k_improved, None, None),
        (f"hybrid_rerank{args.rerank_candidates}_top{args.top_k}", "hybrid",
         args.top_k, args.rerank_candidates, "dense"),
    ]
    for name, method, result_k, candidate_k, rerank in configurations:
        retriever = retrievers[method]
        method_traces = []
        for question in questions:
            gold = question.get("gold")
            if gold is None:
                gold = [{"source_id": question["gold_source_id"], "page": question["gold_page"],
                         "evidence": question["evidence_quote"]}]
            filter_spec = question.get("filter") or {}
            filters = filter_spec if isinstance(filter_spec, dict) and filter_spec else None
            started = time.perf_counter()
            results = retriever.search(question["question"], top_k=result_k,
                                       filters=filters, candidate_k=candidate_k, rerank=rerank)
            retrieval_ms = round((time.perf_counter() - started) * 1000, 2)
            embedding_ms = query_embedding_ms[question["question"]] if method != "tfidf" else 0.0
            elapsed_ms = round(retrieval_ms + embedding_ms, 2)
            ranked = [{**row, "hit": hit(row, gold)} for row in results]
            context, context_ids = pack_context(results, budget=args.context_budget)
            context_evidence_hit = any(
                hit(row, gold) and row["id"] in context_ids for row in results
            )
            trace = {"query_id": question["id"], "split": args.split, "method": name,
                     "base_method": method, "question": question["question"], "top_k": result_k,
                     "candidate_k": candidate_k, "reranker": rerank, "gold": gold,
                     "filter": filters, "results": ranked, "hit1": bool(ranked and ranked[0]["hit"]),
                     "hit_at_k": any(row["hit"] for row in ranked), "elapsed_ms": elapsed_ms,
                     "retrieval_ms": retrieval_ms, "query_embedding_ms": embedding_ms,
                     "context_budget_chars": args.context_budget, "context": context,
                     "context_ids": context_ids, "context_chars": len(context),
                     "context_evidence_hit": context_evidence_hit}
            method_traces.append(trace)
            traces.append(trace)
        n = len(method_traces)
        summaries[name] = {"split": args.split, "questions": n, "top_k": result_k,
                             "candidate_k": candidate_k, "reranker": rerank,
                             "hit1": f"{sum(t['hit1'] for t in method_traces)}/{n}",
                             "evidence_hit_at_k": f"{sum(t['hit_at_k'] for t in method_traces)}/{n}",
                             "context_evidence_rate": f"{sum(t['context_evidence_hit'] for t in method_traces)}/{n}",
                             "context_budget_chars": args.context_budget,
                             "avg_context_chars": round(sum(t["context_chars"] for t in method_traces) / n, 1),
                             "median_query_ms": sorted(t["elapsed_ms"] for t in method_traces)[n // 2]}

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
