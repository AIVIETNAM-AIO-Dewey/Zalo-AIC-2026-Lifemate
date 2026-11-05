# Bàn giao: chạy submission public test LifeMate trên Google Colab

## Bạn nhận được gì
| Thứ | Ở đâu |
|---|---|
| Source (pipeline của team, đã sửa để nạp đúng dtype gốc) | GitHub, nhánh `feat/asr-ft-native-dtype` |
| Notebook Colab | `notebook/06_public_submission_colab.ipynb` trong nhánh trên, hoặc file cùng tên trong thư mục Drive bàn giao |
| Model ASR đã fine-tune (PhoWhisper-medium, **float32**, 3,06 GB) | Thư mục Drive `ZaloAIC_LifeMate_handoff`: `phowhisper_medium_ft_v1_fp32.zip` kèm file `.sha256` |
| LLM Qwen3-4B (**bfloat16**) | Notebook tự tải từ Hugging Face |
| Dữ liệu BTC `train+public_test.zip` | **Không có trong thư mục bàn giao.** Tải từ portal bằng tài khoản đội. BTC không cho phép phát tán dữ liệu ra ngoài đội |

## Các bước
1. Mở thư mục Drive được share (`ZaloAIC_LifeMate_handoff`) → chuột phải → **Organize → Add shortcut** → chọn **My Drive**.
2. Đặt file `train+public_test.zip` vào **My Drive**. Nếu để chỗ khác thì sửa biến `DATA_ZIP` trong ô cấu hình.
3. Mở Colab → **File → Upload notebook** → chọn `06_public_submission_colab.ipynb`. Cách khác: **File → Open notebook → GitHub**, dán link repo và chọn nhánh `feat/asr-ft-native-dtype`.
4. **Runtime → Change runtime type → GPU L4** (24 GB, có bf16; cấu hình này gần với GPU A30 của BTC) hoặc **A100**. Không nên dùng T4.
5. **Runtime → Run all**. Cho phép Colab truy cập Drive khi được hỏi.
   - Ô *Pilot* chạy 48 mẫu dev có nhãn và in ra `state_match_rate`, `latency_seconds`, `stop_reasons`. Nếu có nhiều mẫu timeout, gửi số liệu này cho nhóm trước khi chạy public test.
   - Ô *Public test* chạy 634 mẫu. Mỗi mẫu được tối đa 30 giây, nên trường hợp xấu nhất mất khoảng 5 giờ. Kết quả ghi thẳng vào Drive, thư mục `My Drive/lifemate_out/`.
6. Gửi lại cho nhóm:
   - `My Drive/lifemate_out/public_asr_ft_v1/submission.jsonl`: file để nộp.
   - Cả thư mục `public_asr_ft_v1/` và `pilot_asr_ft_v1/`: để nhóm phân tích timeout và lỗi.
   - Dòng `Đỉnh VRAM: ... MB` mà notebook in ra.

## Nếu gặp lỗi
- **`Sai checksum model`**: file trên Drive bị tải dở. Xóa file đó rồi chạy lại ô.
- **CUDA OOM**: GPU đang dùng nhỏ hơn 20 GB. Đổi sang L4 hoặc A100.
- **Colab ngắt giữa chừng khi đang chạy public test**: chạy lại ô *Public test*. `run.py` cần một thư mục output mới, nên ô này sẽ xóa thư mục cũ và chạy lại từ đầu.
- Lỗi khác: gửi cho nhóm khoảng 20 dòng cuối của ô bị lỗi.

## Ghi chú kỹ thuật
- Model ASR là LoRA (r=32, notebook 04, epoch 9) đã merge vào PhoWhisper-medium fp32. Trên 1.001 câu dev, giải mã greedy cho 502 câu sai và WER 8,0%, so với 866 câu sai và WER 26,5% của PhoWhisper gốc.
- `src/lifemate/models.py` nạp model bằng `torch_dtype="auto"`, tức là dùng dtype ghi trong config: PhoWhisper chạy fp32, Qwen3-4B chạy bf16. Trước đây code ép cả hai về fp16.
- Chưa có số đo end-to-end (điểm, VRAM, latency) trên GPU 24 GB. Lần chạy này sẽ là lần đo đầu tiên.
