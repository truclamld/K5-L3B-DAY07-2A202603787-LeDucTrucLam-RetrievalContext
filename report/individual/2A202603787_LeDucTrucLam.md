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

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:

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
