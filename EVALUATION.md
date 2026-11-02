# Đánh giá LifeMate trên train/dev

`python -m lifemate.evaluate` thực hiện quy tắc công khai tại
[LifeMate → Evaluation](https://challenge.zalo.ai/portal/lifemate), kiểm tra ngày
02/11/2026. Đây là bản triển khai local; BTC chưa cung cấp mã server scorer.

## Quy tắc

- Replay toàn bộ call **theo thứ tự** qua đúng `zaic_tools` của split, gồm cả call
  bị schema từ chối. Kết quả replay là căn cứ so state với nhãn; hash người dùng
  gửi khác hash replay được ghi riêng vào `hash_reproducible`/`hash_mismatches`.
- Nếu gold không có call: điểm bằng 1 khi prediction cũng không có call, ngược lại
  bằng 0. Thiếu cả prediction không được coi là một prediction rỗng hợp lệ.
- Nếu gold có call và replay state sai: điểm bằng 0.
- Tên tool và mọi argument không fuzzy phải giống hệt, kể cả sự có mặt của các
  argument tùy chọn. Không tự thêm default hay bỏ argument thừa để tăng điểm.
- Chỉ argument đánh dấu `x-match: fuzzy` trong catalog mới được fuzzy match.
  Ngưỡng là **token-F1 > 0,6**, không phải `>= 0,6`. `sys.ask_missing_info`
  được chấm như tool khác, không có điểm riêng cho việc hỏi lại.
- Ghép call một-một bằng maximum bipartite matching; thứ tự call không ảnh hưởng
  bước ghép. Call trùng hoặc bị từ chối vẫn nằm trong mẫu số precision; call bị từ
  chối không được match. Với M cặp khớp, P call nộp, G call gold, F1 = 2M/(P+G).
- Điểm tổng `score_percent` = 100 × trung bình điểm từng mẫu. Không tự cho 0
  dựa trên `stop_reason`. Thời gian và các giới hạn của môi trường thi được kiểm
  tra riêng; có điểm local không chứng minh hệ thống đáp ứng budget thi.

## Phần chưa được BTC đặc tả đầy đủ

Trang BTC nêu lowercase, Unicode NFC và token-F1, nhưng không nêu thuật toán
tách token hay xử lý token lặp/dấu câu. Phiên bản `lifemate-published-rules-v1`
dùng `NFC(text.lower()).split()` và đếm token theo multiset; giữ dấu tiếng Việt,
dấu câu và token lặp. Hai chuỗi không có token được coi là khớp; một chuỗi không
có token thì không khớp chuỗi có token. Những lựa chọn này được ghi trong JSON
kết quả. Không tuyên bố khớp tuyệt đối với server khi BTC chưa xác nhận chi tiết.

Không dùng `_tokens`/`_fold` của runtime làm scorer: đó là helper tìm kiếm,
có bỏ dấu tiếng Việt, khác với chuẩn hóa được công bố cho fuzzy call match.

## Chấm lại dự đoán đã lưu, không chạy lại model

Từ thư mục repo, dùng Python có `src` trong module path:

```bash
PYTHONPATH=src python3 -m lifemate.evaluate \
  --data-root data \
  --run /duong/dan/b0_oracle \
  --output runs/b0_rescored
```

`--output` phải là thư mục mới; giữ nguyên run gốc. Nếu bỏ `--output`, hai file
`evaluation.json` và `per_sample_evaluation.json` trong run được cập nhật như
trước đây. `diagnostics.jsonl` không bắt buộc để chấm calls/state; nếu có, evaluator
thêm metric ASR, latency và stop reasons. Chỉ chấm nhãn train/dev, không tune test.

Metric chính là `score_percent`, `mean_call_f1` và `by_category.*.score_percent`.
Các field `mean_exact_call_f1`/`state_gated_exact_f1_percent` được giữ làm chẩn đoán
exact match; field gated cũng bỏ hình phạt stop reason, nên không so trực tiếp
với field cùng tên của evaluator cũ. `state_match_rate` chỉ phản ánh canonical
state; runtime loại các trường text tự do khỏi hash, nên không thay thế điểm call.

Notebook nhúng source cũ cần được đồng bộ snapshot trước khi chạy lại. Có thể
dùng CLI trên với source mới để chấm ZIP cũ ngay, không cần chạy lại inference.

## Kiểm tra và kết quả chấm lại

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_evaluation.py' -v
```

Bộ test kiểm ngưỡng 0,6, NFC/lowercase, argument exact, schema fuzzy lồng nhau,
call trùng, matching không phụ thuộc thứ tự, rejected calls, empty gold,
missing prediction, hash replay và stop reasons. Integration tests cần data train
local và tự skip nếu chưa có. Có thể đặt `LIFEMATE_TEST_DATA_ROOT` để dùng data
ngoài checkout.

Gold self-check với runtime BTC: **1.545/1.545 mẫu được điểm 1**. Đây là kiểm tra
evaluator bằng nhãn, không phải chất lượng model.

Chấm lại ZIP `b0_dev_20261101T125855Z_24f9960d.zip` trên cùng 309 prediction:

| Cách tính | Điểm (%) |
|---|---:|
| Exact + stop-reason penalty cũ | 8,4936 |
| Exact, bỏ stop-reason penalty | 10,2870 |
| Fuzzy theo quy tắc công khai, tokenizer local đã khai báo | 11,6678 |

State match vẫn là 87/309; hash replay khớp 309/309; 37 mẫu chạy quá 30 giây
trên L4 với deadline tắt. Thay đổi điểm đến từ evaluator, không phải model tốt lên.
WER/CER bằng 0 vì input oracle, không phải kết quả ASR. Báo cáo tổng hợp và
checksum nằm trong `reports/evaluation/b0_dev_20261101_rescore.json`; raw prediction,
trace, nhãn và environment giữ ngoài Git.
