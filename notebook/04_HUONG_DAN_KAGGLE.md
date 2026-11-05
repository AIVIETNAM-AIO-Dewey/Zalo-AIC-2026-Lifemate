# Hướng dẫn chạy notebook 04 trên Kaggle

Notebook: `notebook/04_asr_pipeline_full_kaggle.ipynb`. Notebook tự fine-tune PhoWhisper-medium bằng LoRA trên data BTC, chọn checkpoint có **ít câu sai nhất trên dev**, rồi xuất model đã merge để dùng như PhoWhisper bình thường.

## 0. Chuẩn bị tài khoản (làm một lần)
- Kaggle → Settings → **Phone verification**. Chưa xác minh số điện thoại thì không bật được GPU và Internet.

## 1. Upload dữ liệu BTC thành dataset **Private**
1. Kaggle → **Datasets → New Dataset**.
2. Kéo file `H:\lab\ZaloAIC\train+public_test.zip` (876 MB) vào. Kaggle sẽ tự giải nén.
3. Đặt tên, ví dụ `zaic2026-lifemate-data`. Visibility: **Private**, vì BTC không cho phép chia sẻ dữ liệu → **Create**.
4. Đợi trạng thái chuyển sang *Ready*.

Không cần upload thư mục `model/PhoWhisper-medium` (3 GB): khi Internet bật, notebook tự tải `vinai/PhoWhisper-medium` về. Nếu vẫn muốn dùng bản local, hãy upload thư mục đó thành một dataset riêng (giữ nguyên tên thư mục `PhoWhisper-medium`). Notebook sẽ tự tìm thấy `PhoWhisper-medium/config.json` và ưu tiên dùng bản này.

## 2. Tạo notebook
1. Kaggle → **Code → New Notebook** → menu **File → Import Notebook** → chọn `04_asr_pipeline_full_kaggle.ipynb`.
2. Panel phải → **Session options**:
   - Accelerator: **GPU T4 x2** hoặc **GPU P100**. Notebook chỉ dùng 1 GPU.
   - Internet: **On**.
   - Persistence: **Files only**.
3. **Add Input → Your Work → Datasets →** chọn `zaic2026-lifemate-data`.

## 3. Chạy thử (SMOKE) trước, khoảng vài phút
1. Ở ô cấu hình đầu tiên, đặt `SMOKE = True`.
2. Bấm **Run All**. Notebook chạy được tới dòng `done` ở ô cuối là pipeline thông suốt. Ở chế độ này các con số chưa có ý nghĩa vì chỉ chạy trên 24 câu dev.
3. Nếu gặp lỗi, copy khoảng 20 dòng cuối của ô bị lỗi và gửi lại cho tôi.

## 4. Chạy full ở chế độ nền
1. Đặt lại `SMOKE = False`.
2. Bấm **Save Version** (góc phải trên) → chọn **Save & Run All (Commit)** → **Save**.
3. Notebook chạy nền, bạn có thể tắt trình duyệt. Theo dõi tiến độ ở **View Active Events** hoặc tab **Logs** của version đó.
4. Giới hạn của Kaggle: mỗi phiên chạy tối đa khoảng 12 giờ, và GPU có quota theo tuần (xem số còn lại ở Settings) `[CHƯA XÁC MINH: con số quota thay đổi theo thời điểm]`. Nếu sợ hết giờ, giảm `EPOCHS`. Notebook đã có early stop (`PATIENCE = 3`).

## 4b. Chạy tiếp từ checkpoint, không train lại (RESUME)
Dùng khi lần chạy trước đã train xong nhưng lỗi ở bước cuối:
1. Import lại notebook 04 bản mới nhất.
2. **Add Input → Your Work → Notebooks →** chọn notebook đã chạy trước, đúng version có `asr_pipeline/best_lora.pt`. Giữ nguyên dataset data.
3. Ở ô cấu hình, đặt `RESUME = True` và `SMOKE = False` → **Save & Run All (Commit)**.
4. Notebook sẽ bỏ qua bước baseline và train, nạp `best_lora.pt`, rồi chạy bước so greedy với beam, merge, lưu, kiểm tra và báo cáo.

## 5. Đọc kết quả
Mở version đã chạy xong → tab **Output** → thư mục `asr_pipeline/`:

| File | Nội dung |
|---|---|
| `summary.csv` | So sánh trước/sau: `n_wrong` (số câu sai), `WER`, `CER`, độ chính xác chuỗi số, recall phủ định |
| `results.json` | Như trên, kèm thông tin điểm xuất phát, epoch tốt nhất, số beam, kết quả kiểm tra reload, latency |
| `history.json` | Từng epoch: loss, `n_wrong`, WER |
| `wrong_final.csv` | Toàn bộ câu dev còn sai, đã phân loại (số / phủ định / thêm-thiếu từ / thay từ) |
| `dev_*.csv` | Transcript từng câu của mỗi lần chấm |
| `phowhisper_medium_ft/` | **Model đã fine-tune và merge (fp16, safetensors)** |

Cách so sánh trước/sau:
- **Trước:** dòng `base` (và `adapter` nếu có).
- **Sau:** dòng `final`.
- Mục tiêu của bạn tương ứng với cột `n_wrong` của dòng `final` < 10.

Gửi tôi `summary.csv` và `wrong_final.csv` để tôi phân tích bước tiếp theo.

## 6. Dùng trọng số mới
1. Trong version đã chạy xong → tab **Output** → **New Model** (hoặc **New Dataset**). Kaggle sẽ tạo model/dataset private từ output, không cần tải về rồi upload lại.
2. Trong notebook inference, chọn **Add Input** → model/dataset đó, rồi nạp:
   ```python
   from transformers import WhisperForConditionalGeneration, WhisperProcessor
   p = "/kaggle/input/<tên>/asr_pipeline/phowhisper_medium_ft"   # sửa theo đường dẫn thật
   model = WhisperForConditionalGeneration.from_pretrained(p).half().cuda().eval()
   processor = WhisperProcessor.from_pretrained(p)
   ```
   Model có cùng kiến trúc với PhoWhisper-medium, nên pipeline hiện có chỉ cần đổi đường dẫn.

## Ghi chú trung thực
- Dev gồm các câu của 309 hội thoại train mà model **không được học** (đúng split `reports/data_audit/dev_ids.json`). Public test không có transcript nên không chấm ASR trên đó được.
- Data không có nhãn vùng miền hay speaker, nên notebook **không** báo cáo được kết quả riêng cho từng vùng Bắc, Trung, Nam. `wrong_final.csv` là nơi để xem lỗi theo từng câu.
- Chưa có kết quả trên GPU Kaggle. Mục tiêu dưới 10 câu sai chỉ được xác nhận khi `summary.csv` cho thấy điều đó.
