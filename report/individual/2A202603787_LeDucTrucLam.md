# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: Python Tutorial 3.7.0, Python Software Foundation License Version 2, 98 trang nội dung, 98 chunks; file nguồn `data/Tutorial_EDIT.pdf`.
- Vector store, embedding model, phiên bản và số chiều: Không dùng vector database; embedding chunks local và xếp hạng cosine trong bộ nhớ. `text-embedding-3-small`, 1536 chiều. Phiên bản model chưa ghi nhận.
- Agent model, cách chạy và phân chia dev/test: Agent chưa được benchmark. Retrieval được so sánh bằng `starter/evaluate_retrieval.py`, top-3, trên cùng 98 chunks; benchmark có 15 câu dev và 5 câu test. Kết quả dưới đây là dev.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: Bộ câu hỏi hiện có 20 câu với gold evidence; chưa có câu thiếu evidence. Retrieval được tính hit khi chunk thuộc đúng `source_id` và `page`, đồng thời chứa nguyên văn evidence (chuẩn hóa khoảng trắng). Hit@1 lấy chunk đầu; Evidence Hit@3 tính có ít nhất một chunk trúng trong top-3.

## Retrieval và thử nghiệm

| Cấu hình         | Phương pháp, top-k, filter, rerank, context budget            | Giả thuyết / thay đổi                                                               | Split | Hit@1        | Evidence Hit@3 | Evidence Hit@5 | Context Evidence Rate | Thời gian / chi phí                           |
|------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------|-------|--------------|----------------|----------------|-----------------------|-----------------------------------------------|
| Baseline         | TF-IDF; top-3; không filter/rerank                            | Mốc lexical để so sánh với các phương pháp khác.                                    | dev   | 4/15 (26,7%) | 9/15 (60,0%)   | -              | 6/15 (40,0%)          | Median 0,05 ms/query; chi phí 0               |
| Phương pháp 2    | Dense cosine; top-3; không filter/rerank                      | Embedding ngữ nghĩa có thể tìm evidence diễn đạt khác từ khóa trong câu hỏi.        | dev   | 3/15 (20,0%) | 10/15 (66,7%)  | -              | 6/15 (40,0%)          | Median 325.68 ms/query; chi phí chưa ghi nhận |
| Phương pháp 3    | Hybrid RRF (dense + TF-IDF, k=60); top-3; không filter/rerank | Hợp nhất hai thứ hạng có thể cải thiện độ phủ evidence so với từng phương pháp riêng.| dev   | 6/15 (40,0%) | 11/15 (73,3%)  | -              | 9/15 (60,0%)          | Median 325.43 ms/query; chi phí chưa ghi nhận |
| Cải thiện 1      | Hybrid RRF top-5                                              | Tăng top-k của Hybrid RRF lên 5                                                     | dev   | 6/15 (40,0%) | 11/15 (73,3%)  | 13/15 (86,7%)  | 9/15 (60,0%)          | Median 325.41 ms/query; chi phí chưa ghi nhận |
| Cải thiện 2      | Hybrid RRF top-10, rerank bằng dense cosine, lấy top-3        | Lấy top 10 từ Hybrid RRF, dùng reranker bằng dense cosine xếp lại rồi trả top 3     | dev   | 3/15 (20,0%) | 10/15 (66,7%)  | -              | 6/15 (40,0%)          | Median 325.38 ms/query; chi phí chưa ghi nhận |
| Cấu hình đã chọn | Hybrid RRF top-5                                              | Lựa chọn từ kết quả dev                                                             | test  | 2/5 (40,0%)  | -              | 4/5 (80,0%)    | 3/5 (60,0%)           | Median 371.39 ms/query; chi phí chưa ghi nhận |
| Baseline         | TF-IDF; top-3; không filter/rerank                            | Kết quả đối chứng với cấu hình được chọn                                            | test  | 2/5 (40,0%)  | 3/5 (60,0%)    | -              | 3/5 (60,0%)           | Median 0.11 ms/query; chi phí 0               |

Kết quả dev cho thấy Hybrid RRF top-5 đạt Evidence Hit@5 13/15 (86,7%), tăng từ 11/15 (73,3%) ở hybrid top-3; Hit@1 giữ nguyên 6/15 (40%). Với budget context 5.000 ký tự, Context Evidence Rate của hai cấu hình đều 9/15 (60%). Top-5 được chọn vì tăng độ phủ retrieval trên dev, dù context rate không tăng theo cách đóng gói hiện tại.

Khi tính Context Evidence Rate, kết quả từ mỗi câu hỏi được đóng gói theo thứ hạng bằng `pack_context`, với ngân sách 5.000 ký tự. Chunk chỉ được đưa vào context nếu cả block vừa ngân sách; chunk không vừa bị bỏ qua. Một câu được tính là có context evidence nếu context chứa chunk đúng nguồn và trang, có chứa gold evidence sau khi chuẩn hóa khoảng trắng. Context Evidence Rate là số câu đạt tiêu chí này chia cho số câu có đáp án. Đây là context mô phỏng, chưa phải context thực tế gửi cho agent.

Phân tích hai lỗi dev:

- **q04 — retrieval miss:** Câu hỏi hỏi xử lý slice index vượt giới hạn; gold nằm trong chunk `p20:c3`, nhưng chunk này không có trong top-5 Hybrid RRF. Top-5 gồm các trang 9, 107, 22, 15 và 112. Câu hỏi tiếng Việt còn corpus tiếng Anh; TF-IDF không có từ khóa tương ứng trực tiếp, còn dense trả về các đoạn liên quan rộng tới slice/sequence nhưng không tìm đúng đoạn. Đây có vẻ là lỗi xếp hạng do lệch ngôn ngữ và truy vấn ngắn; context budget không phải nguyên nhân vì gold chưa được retrieve.
- **q05 — context packing miss:** Gold chunk `p20:c3` được retrieve ở hạng 2, nhưng không lọt vào context. Block hạng 1 dài 3.214 ký tự, gold block dài 1.823 ký tự; cộng thêm dấu phân cách thì vượt budget 5.000 ký tự, nên `pack_context` bỏ qua gold chunk và chuyển sang chunk hạng sau. Đây là lỗi đóng gói whole-chunk theo thứ hạng, không phải retrieval miss. Có thể thử cách chọn context/budget khác trên dev, nhưng cần giữ budget cố định khi so sánh và đánh giá lại bằng test mới nếu đổi cấu hình sau khi đã xem test.

## Agent và đánh giá

- Tool: `vector_search` (dense cosine) và `keyword_search` (TF-IDF).
- Số lượt gọi: 3
- Điều kiện dừng: Trả lời khi context đủ evidence; nếu chưa đủ thì tìm tiếp, tối đa 3 lượt retrieval, sau đó trả lời từ context hoặc abstain. Lỗi provider/retriever được ghi vào trace. Thực tế test có lặp query ở q2 và false-01.
- Cách chọn context và xử lý lỗi: Gộp kết quả theo thứ tự tìm, loại chunk trùng và giữ context tối đa 5.000 ký tự gồm nguồn/trang. Trace lưu kết quả, context, câu trả lời, citation, lỗi và thời gian; citation được đối chiếu thủ công.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Hai tool: vector_search (dense cosine), keyword_search (TF-IDF); test; gpt-4.1-mini; top-5; tối đa 3 lượt tìm; context 5.000 ký tự | 10/10 (100%) | 22 lượt: 4 câu × 1, 6 câu × 3; TB 2,2; vector 0, keyword 22 | 0: 0; 1: 0; 2: 5; TB 2,0/2 | 6/7 (85,7%) | 6/6 (100%) | 4/5 (80%) | 0/5 (0%) |

| Query ID | Loại câu     | Tool call và kết quả                                                                         | Câu trả lời và citation                                                                                                                                                                                                                                                                                                                  | Nhãn, lý do                                                                                                                                  |
|----------|--------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| q1       | Có đáp án    | keyword_search                                                                               | Tuple là bất biến (immutable), không thể thay đổi các phần tử của tuple sau khi tạo. Tuy nhiên, tuple có thể chứa các đối tượng có thể thay đổi được (mutable objects) như list. Do đó, mặc dù không thể thay đổi tuple, nhưng có thể thay đổi nội dung của list bên trong tuple sau khi tuple được tạo. [python-tutorial-3-7-0:p42:c5]" | answer_grade: 2; Câu trả lời chính xác, đầy đủ và có chứa citation hỗ trợ, chỉ đúng đến vị trí trong tài liệu                                |
| q2       | Có đáp án    | keyword_search x3; truy vấn "tuple one element syntax" bị lặp lại; context có p42:c5         | Để tạo tuple một phần tử, theo sau giá trị phải có dấu phẩy; chỉ đặt giá trị trong ngoặc đơn là chưa đủ. Ví dụ `singleton = 'hello',`. [python-tutorial-3-7-0:p42:c5] | answer_grade: 2; citation có trong context và hỗ trợ trực tiếp cú pháp dấu phẩy. |
| false-01 | Không đáp án | keyword_search x3; cùng truy vấn bị lặp lại; context không có evidence về asyncio/event loop | Không đủ bằng chứng để trả lời. | abstention_correct: true; tài liệu không nêu event loop mặc định trên Windows hay phiên bản policy bắt đầu. |
| false-02 | Không đáp án | keyword_search x3                                                                            | Trên bản build CPython 64-bit mà tutorial dùng, một list rỗng chiếm chính xác 64 byte theo sys.getsizeof([]). [python-tutorial-3-7-0:p53:c6]                                                                                                                                                                                             | 	abstention_correct: false; Câu trả lời không hề tồn tại trong tài liệu, citation cũng không có dữ liệu mà agent trả lời. Agent đã bịa câu trả lời này. |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: `starter/results/agent-two-tools-test/`; 43,24 giây tổng thời gian truy vấn, dựng index 2,12 giây, 32 request model, 40.109 tokens; lỗi 0; chi phí chưa ước tính. Tool use 10/10; 22 lượt retrieval đều là keyword_search, vector_search không được chọn.


## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:
