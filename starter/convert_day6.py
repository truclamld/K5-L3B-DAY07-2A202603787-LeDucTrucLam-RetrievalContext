"""Convert Day 6 JSON/JSONL exports to the Day 7 chunks/questions schema."""
import argparse
import json
import re
from pathlib import Path


def read_records(path):
    """Read either a JSON array or JSON Lines file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list):
        raise ValueError(f"{path}: đầu vào phải là JSON array hoặc JSON Lines")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: mỗi bản ghi phải là object JSON")
    return value


def convert_chunk(chunk, index):
    source_id = chunk.get("source_id") or chunk.get("document_id")
    page = chunk.get("page")
    text = chunk.get("text")
    source = chunk.get("source")
    if not source_id or page is None or not text or not source:
        raise ValueError(f"Chunk #{index}: cần document_id/source_id, page, text và source")

    # Keep an existing chunk ID; otherwise use section's chapter number (e.g. chapter-15 -> c15).
    section = chunk.get("section")
    chapter_match = re.search(r"(?:chapter[-_ ]?)?(\d+)$", str(section or ""), re.IGNORECASE)
    if not chapter_match:
        raise ValueError(
            f"Chunk #{index}: không tạo được ID vì thiếu section có số chapter (ví dụ chapter-01)"
        )
    chapter_id = chapter_match.group(1).lstrip("0") or "0"
    chunk_id = chunk.get("id") or chunk.get("chunk_id") or f"{source_id}:p{page}:c{chapter_id}"
    metadata = dict(chunk.get("metadata") or {})
    # Day 6 exports commonly store these alongside document_id/page/text.
    for key in ("section", "version"):
        if key in chunk:
            metadata[key] = chunk[key]

    return {
        "id": str(chunk_id),
        "text": text,
        "source_id": str(source_id),
        "source": str(source),
        "page": page,
        "metadata": metadata,
    }


def convert_question(question, index):
    quote = question.get("evidence_quote")
    gold = question.get("gold")
    if gold is None:
        if not quote:
            raise ValueError(f"Question #{index}: thiếu gold hoặc evidence_quote")
        gold = [{
            "source_id": question.get("gold_source_id"),
            "page": question.get("gold_page"),
            "evidence": quote,
        }]
        # Additional Day 6 quotes are alternative valid evidence spans.
        if question.get("evidence_quote_2"):
            gold.append({
                "source_id": question.get("gold_source_id"),
                "page": question.get("gold_page"),
                "evidence": question["evidence_quote_2"],
            })

    result = {
        "id": question.get("id"),
        "split": question.get("split"),
        "question": question.get("question"),
        "gold": gold,
        "expected_answer": question.get("expected_answer"),
    }

    # Day 6 filter carries UI/explanation fields; Day 7 expects metadata key/value.
    filter_value = question.get("filter")
    if isinstance(filter_value, dict) and "field" in filter_value and "value" in filter_value:
        result["filter"] = {filter_value["field"]: filter_value["value"]}
    elif filter_value is not None:
        result["filter"] = filter_value
    return result


def main():
    parser = argparse.ArgumentParser(description="Chuyển dữ liệu Day 6 sang schema Day 7")
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    chunks = [convert_chunk(row, i) for i, row in enumerate(read_records(args.chunks), 1)]
    questions = [convert_question(row, i) for i, row in enumerate(read_records(args.questions), 1)]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("chunks.json", chunks), ("questions.json", questions)):
        (args.output / name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Đã chuyển {len(chunks)} chunks và {len(questions)} questions vào {args.output}")


if __name__ == "__main__":
    main()
