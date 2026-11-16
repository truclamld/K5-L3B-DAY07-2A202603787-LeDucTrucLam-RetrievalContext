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

from starter.baseline import hit
from starter.retrieval import Day6Retriever, read_records
from starter.day06_starter.src.embeddings import MyEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, default=Path("starter/results/retrieval-dev"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    chunks = read_records(args.chunks)
    questions = [q for q in read_records(args.questions) if q.get("split") == args.split]
    if not questions:
        raise ValueError(f"No {args.split} questions in {args.questions}")

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
    for method in ("dense", "tfidf", "hybrid"):
        retriever = Day6Retriever(method, chunks=chunks,
                                  embedder=embedder, document_vectors=document_vectors)
        method_traces = []
        for question in questions:
            gold = question.get("gold")
            if gold is None:
                gold = [{"source_id": question["gold_source_id"], "page": question["gold_page"],
                         "evidence": question["evidence_quote"]}]
            filter_spec = question.get("filter") or {}
            filters = filter_spec if isinstance(filter_spec, dict) and filter_spec else None
            started = time.perf_counter()
            results = retriever.search(question["question"], top_k=args.top_k,
                                       filters=filters)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            ranked = [{**row, "hit": hit(row, gold)} for row in results]
            trace = {"query_id": question["id"], "split": args.split, "method": method,
                     "question": question["question"], "top_k": args.top_k, "gold": gold,
                     "filter": filters, "results": ranked, "hit1": bool(ranked and ranked[0]["hit"]),
                     "hit_at_k": any(row["hit"] for row in ranked), "elapsed_ms": elapsed_ms}
            method_traces.append(trace)
            traces.append(trace)
        n = len(method_traces)
        summaries[method] = {"split": args.split, "questions": n,
                             "hit1": f"{sum(t['hit1'] for t in method_traces)}/{n}",
                             "evidence_hit_at_k": f"{sum(t['hit_at_k'] for t in method_traces)}/{n}",
                             "median_query_ms": sorted(t["elapsed_ms"] for t in method_traces)[n // 2]}

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
