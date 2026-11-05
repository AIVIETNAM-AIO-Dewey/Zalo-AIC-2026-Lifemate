# Kiểm chứng batch JSONL synthetic

Đây là bước cho đầu mối kỹ thuật sau khi teammate sinh bản nháp bằng bộ prompt. Có hai việc khác nhau: replay/kiểm tra trạng thái bằng Python và review xem nhãn có đúng ý định người dùng không. Lệnh dưới đây chỉ làm phần tự động và chuẩn bị bằng chứng cho phần review; không phê duyệt train hay xuất SFT.

## 1. Đặt file đầu vào

Lưu nguyên JSONL theo context prompt v1, ví dụ `data/synthetic/drafts/T01_HOME_B001.jsonl`. Mỗi dòng là một object có `environment_draft`, `history_plan`, `proposed_tool_calls`, `expected` và các trường metadata đã quy định. Giữ `verification_status="not_run"`. Không đặt code fence Markdown vào file.

Chạy lệnh từ thư mục repo. Cần Python 3.10+ và gói BTC hiện có ở `data/train/zaic_tools/` cùng catalog train. Không cần GPU, Qwen, cloud API hoặc thư viện Python bổ sung. Các state, observation và nội dung user chỉ ghi dưới outputs, không commit.

## 2. Chạy batch thật

```bash
PYTHONPATH=src python3 -m lifemate.replay_drafts \
  --input data/synthetic/drafts/T01_HOME_B001.jsonl \
  --output outputs/draft_checks/T01_HOME_B001_run01
```

Output phải là thư mục mới. Khi sửa batch rồi chạy lại, dùng `run02`, `run03`... Mã thoát 1 nghĩa là có ít nhất một dòng fail kiểm tra tự động; các báo cáo vẫn được tạo. Mã thoát 0 chỉ nghĩa là mọi dòng qua kiểm tra tự động, không phải mọi nhãn đã được duyệt.

## 3. Script thực hiện những gì?

1. Kiểm tra JSON, ID trùng trong file, định dạng envelope và sự nhất quán giữa context, tool khả dụng và snapshot.
2. Dùng Session của runtime BTC chạy history_plan trên snapshot synthetic before_history. Lấy observations thật và chụp snapshot after_history.
3. Tạo Session mới từ after_history, chỉ chạy calls của yêu cầu mới. History không bị tính lại thành call mới hay thực hiện write lần hai.
4. Kiểm tra từng call bằng schema BTC, rồi lấy observation và state hash thật. Ngoài luật BTC, v1 loại cả arguments không được mô tả trong schema vì runtime có thể bỏ qua chúng.
5. So raw final state với state_assertions và allowed_mutation_paths do draft đề xuất. Kiểm tra cả những field text mà canonical state hash BTC bỏ qua.
6. Xuất report, bằng chứng để review và template quyết định. `accepted_for_sft` luôn false, `semantic_valid`/`causal_valid`/`call_policy_valid` vẫn not_run.

Script dùng snapshot tự tạo trong draft và code runtime/catalog train. Nó không đọc hoặc sửa environment gốc của bất kỳ sample BTC nào. Giai đoạn dựng history là công việc chuẩn bị fixture, tách khỏi Session xử lý yêu cầu mới. Đây không phải chạy model agent, benchmark hay scorer chính thức.

## 4. Đọc file nào?

| File output | Nội dung |
|---|---|
| `summary.json` | Số dòng pass/fail tự động, checksum đầu vào/runtime/catalog/validator |
| `review.md` | Yêu cầu, context, history có kết quả thật, các call mới, observations, assertions, mutation và cảnh báo |
| `report.jsonl` | Toàn bộ bằng chứng máy đọc, gồm snapshot sau history, final state và state hash |
| `review_decisions.jsonl` | Chỗ reviewer ghi quyết định và lý do, gắn với hash bản nháp |

Ví dụ báo cáo có `automatic_pass=3, automatic_fail=2, accepted_for_sft=0` nghĩa là hai dòng cần sửa/loại ngay, ba dòng đủ điều kiện để đọc và duyệt nhãn, chưa có dòng nào được phép train.

## 5. Review nhãn như thế nào?

Mở `review.md`, đối chiếu từng dòng với yêu cầu gốc và prefix model thực sự thấy. Reviewer được xem snapshot/final state trong báo cáo để debug, nhưng các phần state ẩn này không được đưa vào prompt agent.

| Câu hỏi review | Ví dụ cần phát hiện |
|---|---|
| Có đúng ý định không? | Người dùng nói tắt nhưng nhãn lại bật |
| ID và điều kiện có bằng chứng nhìn thấy không? | ID chỉ tồn tại trong hidden state nhưng không có trong request/history/read result |
| Có chọn đúng đối tượng không? | Hai người cùng tên, nhãn tự chọn một người dù chưa đủ quan hệ phân biệt |
| Tool calls có hợp lý không? | Đọc lại ID đã có trong history hoặc hỏi dù có thể tra cứu |
| Ngày, tiền và phủ định có đúng không? | Dùng ngày hệ thống; 50 nghìn thành 500 nghìn; bỏ từ "đừng" |
| Dependency và phần việc độc lập có đúng không? | Gửi tin về event chưa tạo được; bỏ phần đủ thông tin trước khi hỏi |
| Có dừng đúng lúc không? | Gọi thêm sau sys.ask_missing_info; thực hiện nhánh điều kiện sai |
| Assertions/phạm vi mutation có đúng yêu cầu không? | Draft tự cho phép sửa toàn app để che một write ngoài ý định |

Trong `review_decisions.jsonl`, điền reviewer, các trường kiểm tra và notes, rồi đặt decision là `approve`, `fix` hoặc `reject`. Cần ghi bằng chứng cho các quyết định mơ hồ, không chỉ tick true. File này là biên bản review; v1 chưa đọc nó để tự xuất Master/SFT. Khi sửa JSONL đầu vào, chạy lại và review lại vì hash nguồn đã đổi.

`done=false` hoặc danh sách rỗng được cảnh báo, không tự loại: có thể đây là kết quả hợp lệ của lookup hoặc failure branch. Ngược lại, `ok=true` cũng không chứng minh hành động đạt mục tiêu. Intent, causal evidence và gold-style call policy không được script tự khẳng định bằng cách nhìn state hay khớp tên tool.

## 6. Chạy demo trước

Đã có năm bản nháp tự tạo để minh họa; có lỗi cố ý và tuyệt đối không dùng train:

```bash
PYTHONPATH=src python3 -m lifemate.replay_drafts \
  --input schemas/examples/cloud_drafts_review_demo.jsonl \
  --output outputs/draft_checks/demo_run01
```

| ID | Kết quả mong đợi |
|---|---|
| DEMO_GOOD | Pass tự động; vẫn cần review |
| DEMO_WRONG_ID | Call đúng schema nhưng không đạt state đích; fail |
| DEMO_HISTORY | History có write và read; yêu cầu mới không replay history; pass tự động |
| DEMO_EXTRA_ACTION | Đèn đúng nhưng tạo thêm báo thức ngoài yêu cầu; fail |
| DEMO_SEMANTIC_TRAP | User nói tắt, calls và expected cùng ghi bật; pass tự động nhưng reviewer phải reject |

Demo có 3 pass / 2 fail tự động và 0 được duyệt train. Mã thoát 1 của demo là kết quả dự kiến vì có lỗi cố ý. Case SEMANTIC_TRAP cho thấy chính xác giới hạn của việc dùng expected do teacher tự viết: nhãn và hành động có thể khớp nhau nhưng cùng sai với yêu cầu.

## 7. Phạm vi hiện tại

V1 đã có replay/structure/schema/state/mutation checks và báo cáo phục vụ reviewer. Chưa có semantic judge, chứng minh causal provenance tổng quát, call-policy validator, dedup giữa nhiều batch, split planner hoặc production Master/SFT exporter. Chưa chạy trên batch cloud thật vì chưa có file do teammate gửi trong chat này. Các assertions và snapshot cùng đến từ teacher vẫn phải được review độc lập.

Kiểm tra regression CPU:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_replay_drafts.py' -v
```
