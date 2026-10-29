# Quy tắc cộng tác

## Phạm vi hiện tại

Team tập trung vào **agent dùng transcript chuẩn với Qwen3-4B no thinking**. Giữ các thay đổi ASR, LoRA và đóng gói cuối thành đầu việc riêng để baseline có thể so sánh được.

## Nhánh và pull request

1. Chốt đầu việc trong GitHub issue hoặc trao đổi của team, ghi rõ mục tiêu và người phụ trách.
2. Từ `main` mới nhất, tạo nhánh ngắn gọn:
   - `feat/<chu-de>`: chức năng.
   - `fix/<chu-de>`: sửa lỗi.
   - `docs/<chu-de>`: tài liệu.
   - `exp/<chu-de>`: thí nghiệm.
3. Mỗi pull request (PR) tập trung vào một việc; mô tả vấn đề, thay đổi và cách kiểm tra. Ghi rõ phần chưa chạy.
4. Có ít nhất một teammate review trước khi merge. Không push trực tiếp hoặc force-push lên `main`.
5. Tác giả giải quyết góp ý và xung đột trên nhánh của mình. Sau khi merge, có thể xóa nhánh đã hoàn tất.

Tên commit nên thể hiện thay đổi, ví dụ `docs: add teammate setup guide` hoặc `fix: preserve tool results in history`. Tránh commit gộp refactor, đổi split và thay prompt cùng lúc.

Ví dụ làm việc sau khi clone:

```bash
git switch main
git pull --ff-only
git switch -c feat/tool-validation
```

Sau khi hoàn tất, kiểm tra diff, chỉ stage các file liên quan rồi commit, push nhánh và mở PR.

## Data và file được commit

| Commit vào repo | Giữ ngoài Git |
|---|---|
| Source, cấu hình không chứa secret, tài liệu, notebook đã xóa output | Data/audio/environment/runtime gốc trong `data/` |
| Split IDs cố định và checksum của dữ liệu | Model weights, checkpoints, adapters |
| Báo cáo thống kê và bảng metric nhỏ | Raw transcript, nhãn thô, trace, submission, cache ASR |
| Test và fixture nhỏ tự tạo | Token, mật khẩu, `.env`, thông tin đăng nhập Kaggle/Hugging Face |

Mỗi thành viên tự tải data từ nguồn BTC. Không chỉnh sửa `zaic_tools` hay `*_environment/`; nếu phát hiện vấn đề, ghi nhận bằng báo cáo tái hiện. Không thêm giấy phép cho data/model của bên thứ ba thay cho giấy phép gốc.

`.gitignore` chỉ giúp bỏ qua file chưa được theo dõi. Trước khi push, luôn xem `git status`, `git diff --cached --stat` và nội dung các file đã stage. Nếu lỡ commit secret, thu hồi/đổi secret và xử lý lịch sử Git với người quản lý repo.

## Giữ phép so sánh có ý nghĩa

- Dùng split IDs trong `reports/data_audit/`, seed 42. Pilot 48 mẫu là tập debug cân bằng; báo cáo chính trên đủ 309 dev.
- Không đưa nhãn, `final_state` hay dữ liệu trạng thái trực tiếp vào prompt agent. Phần đánh giá train/dev được phép đọc nhãn để chấm và debug, tách khỏi inference.
- Agent chỉ truy cập app qua runtime BTC, một `ToolEnv` mỗi mẫu và ghi đủ call đã thực hiện. Không replay các hành động history.
- Không gắn nhãn public/private test để train hoặc tune. Không chọn cấu hình bằng nhãn test.
- Thay đổi split hoặc cách tính metric phải có giải thích riêng và ghi rõ kết quả cũ nào không còn so sánh trực tiếp được.
- Mỗi lần chạy lưu vào thư mục mới trong `runs/`; không ghi đè kết quả cũ.

Một báo cáo thí nghiệm cần có: commit source (nếu có thay đổi chưa commit thì ghi rõ), model ID/revision, prompt/config, tập IDs, seed, phiên bản thư viện, phần cứng, metric và latency. Nêu rõ dùng transcript chuẩn hay ASR, có cache hay không.

Không gọi kiểm tra source, replay gold hoặc test bằng model giả lập là điểm baseline. Exact-call F1 local chưa phải scorer fuzzy chính thức; timing Kaggle chưa chứng minh timing A30.

## Kiểm tra trước khi mở PR

- Với tài liệu: kiểm tra đường dẫn, câu lệnh và tính nhất quán của nội dung; không cần chạy GPU.
- Với mã: chạy kiểm tra phù hợp với phần thay đổi. Test hiện có được hướng dẫn trong [GETTING_STARTED.md](docs/GETTING_STARTED.md).
- Không có data/GPU thì ghi rõ chưa chạy phần nào; không thay kết quả thật bằng số giả lập.
- Notebook cần xóa execution output, không lưu credential hoặc dữ liệu thô trong cell. Thay đổi có nội dung dùng lại nên nằm trong `src`, notebook gọi lại phần đó.

## Thiết lập trên GitHub

Quy tắc review ở trên là thỏa thuận của team. Người quản lý repo cần cấu hình bảo vệ nhánh trên GitHub nếu muốn bắt buộc bằng hệ thống. Xem [GITHUB_SETUP.md](docs/GITHUB_SETUP.md).
