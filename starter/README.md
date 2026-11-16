# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |
| `retrieval.py` | Dense, TF-IDF và hybrid RRF trên cùng file chunks local đã convert |
| `evaluate_retrieval.py` | So sánh ba retriever trên split dev/test và lưu trace |

Starter **chưa phải solution hoàn chỉnh**. Bạn cần cấu hình embedding provider thật và đánh giá kết quả; không dùng mock để báo cáo benchmark.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

Trong workspace hiện tại, dữ liệu Day 6 nằm ở `data/day6_chunks.jsonl` và `data/day6_questions.jsonl`. Converter nhận cả JSON array lẫn JSONL và tạo `chunks.json`/`questions.json` local. Dùng đúng cặp file converted cho cả baseline và ba retriever để giữ corpus/chunk nhất quán.

## 4. Ba phương pháp truy xuất trên file local

`retrieval.py` đọc trực tiếp cùng file `chunks.json` đã convert và dùng chung tập chunk cho cả ba phương pháp:

1. **Dense:** embed các chunk local và câu hỏi bằng cùng model/chiều, rồi xếp hạng cosine trong bộ nhớ. Không cần PostgreSQL; cần kết nối embedding API.
2. **TF-IDF:** xếp hạng từ khóa trên đúng các chunk local, dùng cùng `TfidfIndex` như baseline. Tokenizer đơn giản nên cần nêu hạn chế với tiếng Việt.
3. **Hybrid RRF:** hợp nhất ranking dense và TF-IDF bằng Reciprocal Rank Fusion, `1/(60 + rank)`. RRF tránh cộng trực tiếp score từ hai thang đo khác nhau.

Đặt `.env` tại `starter/day06_starter/.env` theo `.env.example`; dense cần API key, `EMBEDDING_MODEL` và đúng `EMBEDDING_DIMENSIONS`. PostgreSQL/`DATABASE_URL` không được dùng ở benchmark local này. Không commit `.env`. Cài dependencies trong môi trường Python đang chạy Day 7 bằng `python3 -m pip install -r starter/day06_starter/requirements.txt`.

```bash
python3 -c 'from starter.retrieval import Day6Retriever; r=Day6Retriever("dense", "starter/data/converted/chunks.json"); print(r.search("What does floor division do?", top_k=3))'
```

Thay `dense` lần lượt bằng `tfidf` và `hybrid` để xem từng phương pháp. Kết quả gồm ID, nội dung, nguồn, trang, metadata và score; hybrid có thêm rank thành phần. Với `tfidf`, không cần cấu hình embedding.

Chạy so sánh trên chính các file đã convert. Ngoài ba phương pháp top-3, evaluator tự chạy hai thí nghiệm: hybrid top-5 và hybrid RRF lấy 10 candidates rồi rerank theo cosine dense để trả top-3. Query embedding được tạo một lần cho mỗi câu rồi dùng chung cho mọi cấu hình; thời gian API query được tính vào thời gian dense/hybrid. Context Evidence Rate dùng cùng hàm đóng gói context của baseline; mặc định budget 5.000 ký tự, có thể đổi bằng `--context-budget`:

```bash
python3 starter/evaluate_retrieval.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/retrieval-dev
```

Summary tách cấu hình `hybrid_top5` (Evidence Hit@5) và `hybrid_rerank10_top3` (Hit@1/Evidence Hit@3) khỏi `hybrid` top-3. Mỗi trace lưu context đã đóng gói, ID chunk trong context, số ký tự và cờ `context_evidence_hit`; summary có tỷ lệ context evidence và độ dài context trung bình. Reranker ở đây dùng cosine dense trên candidate pool RRF, không dùng thêm model. Có thể đổi top-k, candidate pool và budget bằng các tham số CLI tương ứng.

Đổi hai đường dẫn nếu baseline TF-IDF của bạn dùng thư mục converted khác. Câu hỏi có thể theo schema converted (mảng `gold`) hoặc schema Day 6 JSONL (`gold_source_id`, `gold_page`, `evidence_quote`).

Chỉ chạy test sau khi đã chốt lựa chọn bằng dev; test Day 6 đã từng được xem ở lab trước, nên không xem đó là holdout độc lập nếu có dùng để chọn/chỉnh cấu hình.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 5. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
