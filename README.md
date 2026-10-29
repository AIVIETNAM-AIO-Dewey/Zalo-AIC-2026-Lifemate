# Zalo AI Challenge 2026 — LifeMate

Workspace của team cho track **LifeMate**: hiểu yêu cầu tiếng Việt và thực thi hành động qua các công cụ do ban tổ chức (BTC) cung cấp.

**Ưu tiên hiện tại: baseline agent dùng text.** Dùng transcript chuẩn trong dữ liệu train để đánh giá Qwen3-4B ở chế độ no thinking. Audio/ASR và fine-tuning là các bước thử nghiệm sau.

## Trạng thái hiện tại

- Đã có khung mã agent, notebook Kaggle và bộ đánh giá chẩn đoán; cần chạy và đánh giá model trên GPU.
- Audit local: 1.545 mẫu train, 634 mẫu public test. Replay gold của 1.545 mẫu train khớp trạng thái đích.
- Đã có 9 kiểm tra tích hợp CPU bằng đầu ra model giả lập. Đây không phải kết quả chất lượng Qwen.
- Chưa có điểm baseline GPU, kết quả LoRA hoặc xác nhận runtime trên A30. Chưa có Docker đã xác minh.

## Luồng baseline

```text
Transcript chuẩn + context + history + schema tool
                        ↓
              Qwen3-4B (no thinking)
                        ↓
           Chọn một tool và arguments
                        ↓
           Kiểm tra schema → ToolEnv
                        ↓
       Đưa kết quả về Qwen → tiếp tục hoặc dừng
                        ↓
              Lưu calls + state hash
                        ↓
                 Đánh giá trên dev
```

Mỗi mẫu có một `ToolEnv`. Agent dùng kết quả công cụ để quyết định bước sau; các hành động đã có trong history không được chạy lại. Nhãn đáp án chỉ được dùng bởi phần đánh giá.

## Cấu trúc hiện tại

```text
.
├── README.md
├── CONTRIBUTING.md                # Quy ước cộng tác của team
├── AGENTS.md                      # Phạm vi và quy tắc cho coding agent
├── requirements.txt
├── .gitignore
├── data/                          # Data BTC tự tải; không đưa lên Git
│   ├── train/
│   └── public_test/
├── notebook/
│   └── 01_lifemate_baseline_kaggle.ipynb
├── src/lifemate/
│   ├── data.py                    # Đọc dữ liệu và chia tập
│   ├── models.py                  # Nạp và gọi model
│   ├── agent.py                   # Prompt, tool schemas, vòng lặp agent
│   ├── run.py                     # Điểm chạy thí nghiệm
│   ├── evaluate.py                # Đánh giá chẩn đoán
│   └── audit.py                   # Khảo sát data và replay gold train
├── tests/                         # Kiểm tra luồng xử lý
├── reports/                       # Phân tích và metadata có thể tái lập
│   └── data_audit/                # Split IDs, thống kê; bỏ qua manifest chứa text
└── runs/                          # Sinh khi chạy; không đưa lên Git
```

Cây trên mô tả source hiện có và nơi lưu kết quả khi chạy. Nếu cần tách prompt, cấu hình hoặc module khác, thống nhất phạm vi trong issue/PR trước khi thay cấu trúc.

## Teammate bắt đầu từ đâu?

1. Đọc [quy tắc cộng tác](CONTRIBUTING.md).
2. Clone repo sau khi được mời; tự tải dữ liệu BTC và giải nén vào `data/`.
3. Làm theo [hướng dẫn bắt đầu](docs/GETTING_STARTED.md). B0 chỉ dùng text; notebook mặc định `RUN_AUDIO=False`.
4. Tạo nhánh riêng, làm một thay đổi có phạm vi rõ, rồi mở pull request (PR) về `main`.

Data, model và kết quả chạy không đi kèm bản clone. Split cố định là **1.236 train / 309 dev**, seed 42; pilot 48 mẫu cân bằng ba nhóm chỉ dùng để debug. Evaluator hiện dùng exact-call F1 và state match, **chưa phải scorer fuzzy chính thức**.

## Quy tắc chung

- Mỗi đầu việc dùng một nhánh và một PR; nhờ ít nhất một teammate review trước khi merge vào `main`.
- Không commit data BTC, model weights, token, mật khẩu hoặc log chứa transcript/nhãn thô.
- Giữ nguyên runtime và environment của BTC; agent truy cập app qua `ToolEnv`.
- Giữ split và seed cố định khi so sánh. Tách rõ kiểm tra CPU, điểm dev và điểm do BTC chấm.
- Notebook cần xóa output trước khi commit. Báo cáo kết quả kèm cấu hình, phiên bản model, phần cứng và cách đánh giá.

Chi tiết: [CONTRIBUTING.md](CONTRIBUTING.md). Quy tắc cho trợ lý lập trình: [AGENTS.md](AGENTS.md).

## Đưa lên GitHub

Xem [hướng dẫn thiết lập GitHub](docs/GITHUB_SETUP.md). Các bước tạo repository, nối remote, push, mời thành viên và bật bảo vệ `main` cần được thực hiện trên repo của team; các file hướng dẫn không tự cấu hình chúng.

Khảo sát đã lưu: [INITIAL_FINDINGS.md](reports/INITIAL_FINDINGS.md) và [audit.json](reports/data_audit/audit.json).

Trang cuộc thi: [LifeMate — Zalo AI Challenge 2026](https://challenge.zalo.ai/portal/lifemate).
