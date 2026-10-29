# LifeMate: khảo sát ban đầu — 29/10/2026

## Dữ liệu thực tế trong workspace

| Thuộc tính | Train | Public test |
|---|---:|---:|
| Mẫu/hội thoại | 1.545 | 634 |
| Audio, gồm các lượt user trong history | 5.036 | 1.933 |
| Tổng giờ audio | 6,83 | 2,76 |
| Độ dài audio trung vị | 4,11 giây | 4,27 giây |
| P95 độ dài audio | 9,86 giây | 10,77 giây |
| Audio dài nhất | 22,33 giây | 20,24 giây |
| App trong catalog | 46 | 36 |
| Tool trong catalog | 147 | 122 |

Tất cả audio được tham chiếu đều tồn tại, đọc header được, là PCM 16-bit mono 16 kHz. Không có audio trùng hash trong từng split. Đây là kiểm tra file/header, không phải nghe kiểm định toàn bộ transcript hay chất lượng âm thanh.

Train có **1.217 mẫu thực thi (78,8%)**, **217 mẫu hỏi thiếu thông tin (14,0%)**, **111 mẫu không gọi tool (7,2%)**. Nhóm empty có thể bao gồm yêu cầu ngoài phạm vi hoặc một lượt không cần hành động; không suy diễn rằng tất cả đều là OOS. Có tối đa 6 gold calls cho một yêu cầu mới, nhưng history có thể chứa 12 lượt user (13 audio nếu tính yêu cầu mới).

Train có `audio_text` cho cả yêu cầu hiện tại và các lượt user trong history; public test không có. History assistant đã có text, tool calls và tool results. Nên dùng toàn bộ các audio train/history làm cặp giám sát ASR, sau khi chia theo hội thoại.

Catalog public chứa **34 tool của 11 app không có trong catalog train**: ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks, train_bus_tickets, translator. Đây là so sánh schema công khai, không phải gắn nhãn test. Agent phải đọc schema động và tổng quát hóa; không giới hạn vào các app đã có nhãn train. Cũng không được suy rằng private test sẽ chỉ có app của public test.

## Kiểm tra execution đã chạy

Replay gold calls trên 1.545 mẫu train bằng runtime BTC: **0 lỗi schema, 0 lệch state hash** so với trạng thái đích được ghép từ initial state và `label.final_state`.

`train_1696`, call index 2 có `ok=true, done=false` vì không tìm thấy chat ID; trạng thái cuối vẫn khớp. Không tự loại dữ liệu chỉ vì gặp `done=false`, và không coi `ok=true` đồng nghĩa với hành động thành công.

Split seed 42: **1.236 train / 309 dev**, giữ hội thoại nguyên vẹn và gộp yêu cầu mới trùng văn bản sau chuẩn hóa. ASR train có 4.035 lượt / 5,40 giờ; ASR dev có 1.001 lượt / 1,43 giờ. Không có metadata speaker/template đáng tin để chứng minh speaker-disjoint hay template-disjoint; các câu history ngắn trùng nhau có thể xuất hiện hai phía. Dùng thêm một bài kiểm tra giữ lại nhóm app/tool trong TRAIN nếu bắt đầu fine-tune LLM.

Pilot 48 dev gồm 16 mẫu mỗi nhóm để debug; cần báo cáo lại trên đủ 309 dev với phân phối tự nhiên.

## Đánh giá hướng model

**PhoWhisper-medium là điểm xuất phát phù hợp để đo.** PhoWhisper đã được fine-tune trên 844 giờ tiếng Việt theo [model card VinAI](https://huggingface.co/vinai/PhoWhisper-medium). Data cuộc thi nhỏ hơn nhiều; chưa có bằng chứng rằng LoRA sẽ cải thiện kết quả. Không có thông tin vùng giọng/speaker/source trong JSONL để thống kê đáng tin theo các nhóm đó.

**Qwen3-4B no thinking phù hợp làm baseline agent cần kiểm chứng.** Dùng `enable_thinking=False` trong chat template, theo [model card Qwen](https://huggingface.co/Qwen/Qwen3-4B). Mã dùng greedy cho cả hai model theo yêu cầu tái lập của cuộc thi; chưa có đo chất lượng trên LifeMate.

Luật đọc ngày 29/10/2026: tổng model <6B, LLM <5B, context ≤16.384; offline, 30 giây/mẫu trên A30. Chỉ truy cập app qua một `ToolEnv`/mẫu. Tham khảo [Rule và Evaluation chính thức](https://challenge.zalo.ai/portal/lifemate); baseline kiểm tra số tham số thực khi load. Hai T4 phục vụ phát triển, chưa chứng minh tốc độ trên A30.

Điểm phụ thuộc trạng thái cuối và F1 call. Thừa read-call có thể làm giảm precision; sai một giá trị quan trọng có thể khiến cả mẫu mất điểm do state hash. Cần đo end-to-end, không tối ưu WER riêng lẻ.

## Thứ tự thử nghiệm và điều kiện quyết định

1. **B0 / oracle-text:** Qwen3-4B nhận transcript chuẩn, context, full history và schema. Đo state match, exact-call F1 chẩn đoán, empty/missing-info, lỗi thời gian/ID/schema và số vòng gọi model.
2. **B1 / audio:** giữ nguyên agent và dev, thay toàn bộ transcript user bằng PhoWhisper. Đo WER/CER và coverage, đồng thời so từng mẫu với B0. Chênh lệch B0−B1 là tín hiệu lỗi lan truyền qua ASR, không phải trần hiệu năng hay quy kết nhân quả tuyệt đối.
3. **Nếu B0 yếu:** xem lỗi đọc history, thời gian tương đối, tool schema, ID và điều kiện trước. Sửa prompt/luồng, cân nhắc SFT/LoRA LLM bằng train trajectories nếu còn lỗi nhất quán. Test giữ lại app/tool trong train để tránh học thuộc schema.
4. **Chỉ pilot LoRA ASR khi** nhiều mẫu B0 đúng/B1 sai thực sự chứa lỗi transcript ở tên người, số tiền, thời gian, địa chỉ hoặc phủ định. Dùng ASR train 5,40 giờ, không trộn dev, không tự gắn nhãn public test. Đặt số bước/checkpoint/budget trước khi chạy.
5. **Giữ adapter** khi cải thiện downstream trên dev cố định cùng mức tài nguyên và không làm hỏng nhóm empty/missing-info hay latency. **Dừng/pivot** nếu chỉ WER đẹp hơn mà end-to-end không tốt lên, hoặc model quá khớp cách đọc/giọng train. Đây là tiêu chí đề xuất, chưa có run LoRA.
6. **Đóng gói sau profile:** đo median/P95/max thời gian theo số lượt history, token vào/ra và VRAM. Khóa phiên bản/model, thử cùng một GPU, kiểm tra offline rồi mới hoàn thiện Docker theo hướng dẫn BTC.

## Ranh giới bằng chứng

Đã chạy: audit file/header/hash, thống kê JSONL, replay gold; 9 kiểm tra tích hợp CPU đều pass, gồm kiểm tra evaluator bằng replay. Các code cell của notebook biên dịch được; chưa thực thi notebook. Test dùng output model giả lập, không đo chất lượng model.

Chưa chạy: inference PhoWhisper/Qwen trên GPU, điểm dev/public, huấn luyện LoRA, đo A30, xác minh Docker. `evaluate.py` chỉ có exact-call diagnostic, chưa phải scorer fuzzy chính thức. Không có kết luận rằng cặp model này đủ 30 giây hoặc có điểm cạnh tranh.
