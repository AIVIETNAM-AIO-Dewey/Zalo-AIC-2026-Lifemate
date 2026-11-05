# Bộ prompt chia việc sinh dataset bằng ChatGPT hoặc Gemini

Mục tiêu là một quy trình thủ công dễ giao cho teammate: dùng context prompt cố định, chọn gói chủ đề, rồi sinh từng batch năm episode nháp. Không cần teammate tự nghĩ schema hoặc API. Prompt không phụ thuộc một model ChatGPT/Gemini cụ thể; cần thử batch nhỏ trên model thực tế đang dùng.

## Bắt đầu từ bộ ZIP

Trong repo, chạy `python3 scripts/build_team_generation_kit.py` để tạo `outputs/team_dataset_kit_v1.zip`. Bộ đã tạo không có data train/dev/test, audio hoặc environment BTC; chỉ có prompt tự viết, quota và tool schema/runtime tham khảo. Runtime/catalog được sao vào outputs đã Git-ignore, không commit các bản sao này.

Giải nén ZIP và làm ba bước trong một chat mới cho mỗi chủ đề:

1. Dán `context_prompt.txt`. Chat sẽ trả READY_FOR_TOPIC_AND_BATCH.
2. Đính kèm hoặc dán đúng một file `topics/TOPIC_ID.txt`. Không gửi cả 12 packet vào cùng chat.
3. Dán `batch_prompt.txt`, thay TOPIC_ID, TEAM_ID và BATCH_ID. Giữ N=5 ở vòng đầu.

Ví dụ batch request:

```text
BATCH_REQUEST
TOPIC_ID: HOME
TEAM_ID: T01
BATCH_ID: B001
N: 5
VARIANTS: lookup_device, history_id_reuse, room_floor_disambiguation, shared_read_multiple_writes, user_self_cancellation
USED_SCENES: []

Sinh đúng 5 dòng JSONL theo context prompt và topic packet. Giữ verification_status="not_run".
```

Khi đổi chủ đề, mở chat mới, dùng lại context prompt và đổi packet cùng TOPIC_ID. Khi sinh batch tiếp theo cùng chủ đề, tăng BATCH_ID. Có thể yêu cầu sửa một dòng sai thay vì sinh lại cả batch. Nếu chat quá dài, mở chat mới với cùng hai thành phần context/packet và danh sách scene đã dùng. Chỉ đổi tên chủ đề mà không cung cấp schema tương ứng sẽ dễ làm model bịa tool.

## Chủ đề và quota

Số bên dưới là episode hoàn chỉnh đã được chấp nhận sau kiểm chứng, không phải số dòng ChatGPT/Gemini xuất ra và không phải số decision samples SFT. Pilot nằm trong V1; không cộng 600 + 2.000 thành 2.600.

| TOPIC_ID | Chủ đề | Pilot | V1 |
|---|---|---:|---:|
| HOME | Đèn, quạt, điều hòa; ID và phòng/tầng | 100 | 300 |
| SMS_PHONE | Tìm người, SMS, gọi điện | 65 | 240 |
| CHAT | Tìm hội thoại và gửi tin nhắn | 35 | 140 |
| TRANSFER | Chuyển ngân hàng/ví, người nhận, số tiền | 90 | 300 |
| BILLS | Tra cứu, thanh toán và tự động thanh toán hóa đơn | 60 | 180 |
| CALENDAR | Tạo, dời, hủy lịch; event trong history | 70 | 220 |
| ALARM_REMINDER | Báo thức, nhắc việc, ngày giờ và lặp lại | 60 | 180 |
| TASKS_NOTES | Thêm/hoàn thành việc, tạo/nối ghi chú | 30 | 100 |
| WEATHER | Tra cứu thời tiết, điều kiện, bằng chứng trong history | 30 | 100 |
| INFO | Giá vàng, xăng, số dư, trả lời từ history | 20 | 80 |
| MULTI | Hai việc liên quan, dependency và làm phần độc lập trước khi hỏi | 20 | 100 |
| STOP | Tự hủy, thiếu capability, nói chuyện, không cần call | 20 | 60 |
| **Tổng** | | **600** | **2.000** |

Đây là phân bổ đề xuất cho quy trình teammate thủ công, chưa có bằng chứng là tối ưu. Nó chia nhỏ các nhóm của thiết kế trước để dễ phân công. Config riêng: `configs/team_dataset_topics.json`; không thay đổi config pilot tự động trước đó. Các tool khác trong catalog train là backlog, chưa coi 12 chủ đề này bao phủ toàn bộ cuộc thi. Tool chỉ xuất hiện trong catalog public nằm ngoài vòng sinh đầu.

Ví dụ chia bốn người: T01 nhận HOME + TASKS_NOTES + STOP (460 episode V1); T02 nhận SMS_PHONE + CHAT + MULTI (480); T03 nhận TRANSFER + BILLS (480); T04 nhận CALENDAR + ALARM_REMINDER + WEATHER + INFO (580). Đây là chia theo khối lượng nội dung, có thể điều chỉnh theo độ khó và số người thật; TEAM_ID giúp tránh trùng ID.

## Phân phối kỹ năng bên trong chủ đề

Quota theo chủ đề và quota theo hành vi là hai trục khác nhau. Trong 2.000 episode, mục tiêu khởi đầu có thể là 1.560 execute/lookup (78%), 300 ask_missing_info (15%) và 140 no_action (7%). STOP đóng góp 60 no_action, phần còn lại nằm trong các family khác; không coi mọi no_action đều ngoài phạm vi.

Trong nhóm execute, cần cả lookup chỉ đọc, read rồi write, dùng lại history để write trực tiếp và một read phục vụ nhiều write. Chọn khoảng 10–15% tổng episode cho lookup chỉ đọc, khoảng 50–65% có history hữu ích, khoảng 10–15% có điều kiện và khoảng 10% có hai subgoal; đây là các tag giao nhau, không cộng thành 100%. Điều chỉnh sau khi review và đánh giá dev, không ép tỷ lệ làm sai ngữ nghĩa tool.

Các cảnh phải khác nhau về grounding hoặc quyết định. Năm câu paraphrase cùng một scene không tương đương năm scene mới. Những paraphrase hoặc counterfactual cùng scene seed ở cùng lineage_group và cùng split.

## Đầu ra ChatGPT/Gemini là bản nháp

Cloud UI thường không có runtime BTC. Vì vậy prompt yêu cầu đề xuất thế giới synthetic và calls, để observations, tool_results và state_hash cho replay local. Không bảo model tự viết observation rồi đánh dấu validated. JSON hợp lệ chưa đủ để được train.

Draft có snapshot **before_history** và history_plan. Pipeline local phải thực thi history trước, lấy kết quả thật, chụp snapshot **after_history**, rồi mới chạy đề xuất cho yêu cầu mới. Master Schema v1 vẫn dùng after_history; không nhầm hai ranh giới này. Với một yêu cầu mới, chỉ có một Session và không replay lại history trong agent loop.

Model xây nháp được nhìn state để thiết kế scene. Vì vậy proposed_tool_calls có tính chất nhãn nháp từ planner; trước khi dùng làm SFT phải kiểm tra mỗi quyết định có thể suy ra từ request/context/history/observations đã có, không từ hidden state hoặc kết quả tương lai. Nếu phải sửa quyết định theo observation thực tế, đưa đúng prefix nhìn thấy cho một lượt teacher riêng hoặc reviewer; không dùng lại cuộc chat planner đã chứa hidden state để khẳng định causal-valid.

Các packet giữ distractor tools để mẫu có lựa chọn gần inference. Với app không liên quan và không bị gọi ở cả history/main plan, prompt cho phép {} làm placeholder ở bản nháp để giảm công nhập liệu. Model phải ghi review_flags=distractor_state_placeholder; reviewer cần quyết định dựng state thực tế hơn trước khi chấp nhận. Không tự coi placeholder là environment đã kiểm chứng đầy đủ.

Teammate nộp JSONL gốc và ghi log ngắn: TEAM_ID, TOPIC_ID, BATCH_ID, ngày sinh, model hiển thị trong UI nếu biết, prompt version và các scene đã dùng. Quota chỉ tăng sau khi replay, kiểm tra nguồn ID, ngữ nghĩa, side effects, terminal và review. Không yêu cầu teammate tự điền hash hoặc verification pass.

Đầu mối kỹ thuật có thể chạy `PYTHONPATH=src python3 -m lifemate.replay_drafts --input data/synthetic/drafts/T01_HOME_B001.jsonl --output outputs/draft_checks/T01_HOME_B001_run01`. Hướng dẫn, demo và giới hạn của kiểm tra tự động nằm trong `docs/REPLAY_SYNTHETIC_DRAFTS.md`. Automatic pass chỉ chuyển mẫu sang bước review; không tự phê duyệt SFT.

## Quy mô fine-tune và A100

Đề xuất đầu tiên là 2.000 synthetic episode được chấp nhận, cộng dữ liệu thật của phần train. A100 cho phép chọn training setup thuận tiện hơn; nó không quyết định cần bao nhiêu mẫu và không thay đổi giới hạn số tham số được triển khai.

Lộ trình: một batch năm mẫu mỗi chủ đề (60 nháp) để sửa prompt; mở lên 600 episode đã kiểm chứng; train pilot và so với real-only; sau đó mới hoàn tất V1 2.000. Nếu tiếp tục cải thiện ở các nhóm lỗi cần thiết, mở lên khoảng 5.000; không cần sinh hàng chục nghìn ngay từ đầu. Các mốc này là ngân sách thực nghiệm, không phải cam kết đủ để đạt một mức điểm.

Với V1 scratch synthetic, chia khoảng 1.800 train / 200 synthetic validation theo lineage, trước khi tách decision samples. Giữ dev BTC 309 mẫu độc lập để đánh giá. Khi dùng đủ 1.236 real train episode, tổng khoảng 3.036 episode để train. Replay gold hiện tại của real train tương ứng 3.876 next-decision samples nếu có DONE cuối bình thường và không thêm DONE sau sys.ask_missing_info. Nếu synthetic train trung bình 3–4 decision/episode, tổng dự kiến khoảng 9.276–11.076 SFT rows; phần synthetic chưa sinh nên đây chỉ là dự toán.

Nếu synthetic bắt nguồn trực tiếp từ một real train seed, descendants phải đi cùng seed. Không đặt descendants vào synthetic validation trong khi vẫn dùng seed để train. Có thể dùng scene tự dựng cho phần synthetic validation, và không dùng dev/test làm seed.

Fine-tune thông thường cập nhật trọng số trong kiến trúc hiện có; số mẫu không thêm layer hoặc parameter. LoRA thêm hai ma trận low-rank cho các module đã chọn, số tham số bổ sung phụ thuộc rank và module, không phụ thuộc số mẫu. Merge LoRA chuẩn vào weight gốc giữ kích thước ma trận như ban đầu khi không thay kiến trúc/vocabulary. Quantization giảm số bit lưu trọng số, không biến giới hạn số tham số thành giới hạn kích thước file.

Qwen3-4B có khoảng 4,02B parameters theo metadata checkpoint. Nếu tổng hệ thống hiện gần 5B do cộng LLM và ASR, thêm dữ liệu không làm tổng đó tăng. Kiểm tra lại parameter count của artifact triển khai cuối, gồm các model còn dùng và adapter nếu chưa merge; không lấy riêng số trainable parameters hoặc số phần tử tensor packed 4-bit làm tổng tham số gốc. Trang luật BTC không trích xuất được trong lần này nên chưa xác nhận cách BTC tính adapter riêng; khái niệm dataset size độc lập với model size vẫn giữ nguyên.

Trên A100 có thể bắt đầu bằng LoRA BF16 để thử nghiệm; không bắt buộc chọn 4-bit chỉ vì lo VRAM. Full fine-tune cũng không tự tăng parameter count. Lựa chọn LoRA/full fine-tune cần so sánh chất lượng; chưa chạy training nào từ bộ prompt này.

## Nguồn cho phần giải thích

- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B): checkpoint/model metadata.
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora): low-rank adapters và merge.
- [OpenAI reasoning best practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices): chỉ dẫn rõ, delimiters, không yêu cầu xuất chain-of-thought.

Các quota và tỷ lệ dữ liệu là đề xuất thực nghiệm của dự án, không được suy ra như khuyến nghị chính thức từ ba nguồn trên.
