# Thiết kế dataset LifeMate v1

Lấy capability trong catalog BTC làm giới hạn nội dung. Mỗi episode tập trung vào một quyết định chính và tối đa hai subgoal có liên quan; tool support chỉ xuất hiện khi cần dữ kiện hoặc giải quyết dependency. Tình huống sinh ra cần trả lời được: model phải quyết định điều gì, bằng chứng nào cho phép quyết định đó, và vì sao cần hoặc không cần từng call?

Đây là đặc tả và kế hoạch pilot, chưa phải dataset cloud đã sinh hoặc kết quả fine-tune. Hai JSON Schema, ví dụ tự tạo và thống kê catalog đã có; generator cloud, production exporter, semantic validator và training runner chưa được triển khai.

## Tệp và ranh giới

| Tệp | Vai trò |
|---|---|
| `schemas/master.schema.json` | Một yêu cầu mới nhất, context/history, tool khả dụng, snapshot, mục tiêu và trajectory |
| `schemas/sft.schema.json` | Một quyết định kế tiếp: `prompt`, `completion`, `tools` và lineage |
| `schemas/examples/master.jsonl` | Ví dụ tự tạo, observation được lấy từ runtime train BTC |
| `schemas/examples/sft.jsonl` | Minh họa chuyển các ví dụ thành decision samples; không dùng train |
| `configs/synthetic_pilot.json` | Tám nhóm capability, biến thể và ngân sách thử nghiệm |
| `reports/synthetic_design/TOOL_COVERAGE.md` | Kiểm kê 147 tool train, required, alternative groups, nguồn ID và tần suất |
| `reports/synthetic_design/tool_inventory.json` | Thống kê đầy đủ và checksum nguồn |

Thống kê chọn scenario chỉ dùng 1.236 IDs train cố định; dev 309 mẫu vẫn là nơi đánh giá. Inventory chỉ đọc catalog public để thấy capability mới, không đọc public rows hoặc environments. Pilot chưa mở rộng sang 34 tool chỉ có trong catalog public. Không sửa runtime hay environment gốc. Snapshot synthetic ghi dưới `data/synthetic/`, được Git bỏ qua.

## Chọn tool rồi chọn quyết định

Đừng sinh một chủ đề rộng như "trợ lý gia đình". Chọn một focus tool hoặc một nhóm tool cùng phục vụ một mục tiêu, đọc parameter schema và runtime, rồi viết scenario có điều kiện kiểm chứng được.

| Nhóm | Quyết định chính | Biến thể cần có |
|---|---|---|
| Liên hệ và giao tiếp | Chọn đúng người/kênh hoặc hỏi | Tên trùng; quan hệ phân biệt; số điện thoại trực tiếp; ID từ history |
| Chuyển tiền | Đúng người, số tiền và điều kiện | Thiếu amount; nhiều người nhận; số dư; đã có beneficiary ID |
| Nhà thông minh | Đúng thiết bị, phòng và tham số | Đọc thiết bị; dùng lại ID; một read phục vụ nhiều writes; tự hủy |
| Calendar | Tìm và sửa đúng event | Dời giờ; ngày tương đối; nhiều event; ID trong history |
| Alarm/reminder/task | Chuẩn hóa thời gian đúng | Giờ rõ; "mai"; thiếu time bắt buộc; bỏ due tùy chọn |
| Hóa đơn | Đúng hóa đơn và hành động | Tìm hóa đơn; ID cũ; điều kiện hạn đóng; runtime noop |
| Tra cứu và điều kiện | Đọc hay dùng lại bằng chứng | Lookup; điều kiện đúng/sai; history sai ngày/vị trí |
| Hai subgoal | Thực hiện phần độc lập, chặn phần phụ thuộc | Làm phần đủ dữ kiện rồi hỏi; không thực hiện bước phụ thuộc đang bị chặn |

Với mỗi tool, xác định required, `x-one-of`, enum, format, `x-ref`, loại read/write/meta, kết quả read và trạng thái write thay đổi. `x-one-of` của BTC nghĩa là ít nhất một nhóm tham số được cung cấp; không tự dịch thành JSON Schema `oneOf` có nghĩa đúng một nhánh. Runtime BTC là nguồn xác nhận ràng buộc thực tế; JSON Schema tổng quát không thay thế `env.validate`.

Không đồng nhất read đứng sát write với read cung cấp dữ liệu cho write. Một read có thể phục vụ nhiều bước; ID có thể đã nằm trong history. Counts đồng xuất hiện trong inventory là gợi ý kiểm tra, chưa chứng minh dependency.

## Kiểm soát context

Planner nhận scenario card tập trung: focus tool, biến thể, subgoal, dependency, quy tắc ngày giờ và điều kiện đích. Card này dùng để xây thế giới và nhãn, không đưa vào prompt agent.

Teacher chọn hành động nhận đúng phần thông tin agent sẽ có: yêu cầu, context, history, tools đã cài và observations trước bước hiện tại. Không cho teacher chọn hành động bằng snapshot ẩn, nhãn target hoặc observation tương lai. Planner được xem state để tạo tình huống nhưng action rollout phải chạy trong cuộc hội thoại riêng, không mang theo nội dung planner.

Tập trung scenario không có nghĩa là chỉ đưa tool đúng vào prompt. SFT phải giữ toàn bộ tools đã cài cộng `sys.ask_missing_info`, gồm distractors thực tế. Dùng cấu hình cài app của train làm gợi ý phân phối, rồi dựng snapshot synthetic nhất quán; không đưa tất cả 147 tools vào mọi mẫu. Nếu chỉ cung cấp 1–2 tools đúng, model sẽ không học chọn tool ở điều kiện inference thật.

History phải phục vụ một kỹ năng xác định: cung cấp ID, xác định đối tượng, ngày giờ, thông tin đã hỏi hoặc bằng chứng cho điều kiện. Canary bắt đầu bằng history ngắn; tăng độ dài bằng các lượt hợp lệ ở giai đoạn sau. Có distractors vừa đủ để luyện chọn đúng bằng chứng, không thêm đoạn văn chủ đề bất kỳ. Không dùng nhãn dev để chọn nội dung synthetic.

Các biến thể đối chứng chỉ thay một yếu tố khi có thể: ID đã biết/chưa biết; tên trùng nhưng quan hệ đủ/chưa đủ; có/không có capability; điều kiện đúng/sai; dự báo đúng/sai ngày. Counterfactual siblings, paraphrases và parent episodes ở cùng split.

## Master: ràng buộc trường

`input` là phần được phép chuyển thành prompt. Context giữ các trường BTC: `current_time`, `timezone`, `location`, `user_profile`, `installed_apps`. Mỗi app chỉ chứa danh sách tool trong phần nhìn thấy; không nhét app state vào context. History được chuẩn hóa thành user `text` và assistant `text`/`tool_calls`/`tool_results`, không cần audio cho SFT text.

`tools` là schema thật của toàn bộ tool khả dụng trong mẫu; không sửa required/enum/description để làm bài dễ hơn. Khi serialize, giữ cả các extension của parameter schema. Membership phải khớp `installed_apps` cộng system tool. Không dùng `focus_tools` làm whitelist inference.

`environment` chỉ cho verifier, gồm snapshot inline hoặc reference bất biến, hash snapshot, catalog và runtime. Snapshot dùng format Session BTC (`_meta` + app states). Ranh giới luôn là **sau khi history đã hoàn tất, trước yêu cầu mới**. Khi rollout yêu cầu mới chỉ tạo một Session và không chạy lại history. Nếu cần dựng history synthetic, tạo và thực thi nó ở giai đoạn xây fixture trước, rồi chụp snapshot sau history; bước này tách khỏi rollout yêu cầu mới.

Hash object (snapshot): SHA256 của JSON UTF-8 `ensure_ascii=False`, `sort_keys=True`, `separators=(",", ":")`, `allow_nan=False`. Catalog SHA256 là hash bytes tệp gốc. Runtime SHA256 là hash JSON canonical của danh sách `[relative_path, sha256(bytes)]` cho các tệp `*.py` dưới package, sắp xếp theo relative path; bỏ cache. Không nhầm các hash provenance này với `state_hash` canonical của BTC: state hash bỏ một số trường text và app thông tin.

`target.outcome` mô tả hình thức kết thúc, không phải trạng thái mọi subgoal:

| Outcome | Trajectory mới | Terminal |
|---|---|---|
| `execute` | Có call read/write, không gọi hỏi | `DONE`; có thể hoàn tất lookup hoặc dừng vì điều kiện sai |
| `ask_missing_info` | Có call hỏi ở cuối; có thể có read/write trước đó | Runtime dừng sau call hỏi |
| `no_action` | Không có call mới | `DONE`; ví dụ tự hủy, thiếu capability, đáp án hoặc điều kiện đã có trong history |

`target.subgoals` lưu dependency và trạng thái đích riêng: completed, awaiting_info, skipped_condition_false, unsupported hoặc cancelled. Khi hỏi đúng vì thiếu thông tin, không bắt buộc mọi subgoal completed. `terminal.reason` giải thích vì sao dừng; điều kiện sai sau một read là execute/condition_false, điều kiện sai đã biết từ history có thể là no_action/condition_false.

`state_assertions` dùng RFC 6901 JSON Pointer vào app-state không có `_meta`: `eq` so giá trị cuối, `unchanged` so với đầu, `exists` kiểm tra tồn tại. `allowed_mutation_paths` là các subtree được phép đổi; phải kiểm tra mọi thay đổi nằm trong phạm vi đó và assertions đủ chặt. Chỉ ghi allowed path không tự chứng minh write đúng. Với lookup/condition-false/no-action, mặc định không cho mutation. Việc chỉ đạt state hash không đủ bảo đảm text, read calls hay đối tượng được chọn đúng.

Mỗi `trajectory` step là một call thật và observation runtime trả về, kèm state hash sau call. `argument_sources` ghi nguồn tham số bằng JSON Pointer vào Master. Với nguồn observation, `source_step` phải nhỏ hơn step hiện tại; với history, reference chỉ vào history đã hoàn tất. Các nguồn derived/default_policy cần kiểm tra ngữ nghĩa riêng. Không được dùng reference tới target/environment làm bằng chứng model có thể thấy. Không sinh các giờ mặc định chưa có policy được kiểm chứng.

JSON Schema kiểm tra hình dạng và một số quan hệ outcome/terminal. Các ràng buộc liên trường, tool arguments động, causal provenance, thứ tự step, dependency, snapshot/history consistency và ngữ nghĩa phải do validator khác kiểm tra. V1 chưa cung cấp production semantic validator. Trường verification do validator/reviewer ghi, không lấy trực tiếp từ teacher.

## SFT: một quyết định tiếp theo

Dùng `prompt` + `completion` + `tools`, một assistant call hoặc `DONE` trong completion. Các cột `episode_id`, `decision_index`, `split_group`, `partition` và `id` chỉ phục vụ lineage, không được serialize vào nội dung model. Training loader chỉ lấy ba cột model input tương ứng.

Prompt đầu có cùng system instruction của baseline và user payload JSON gồm `context`, `history`, `latest_request`. History đã hoàn tất vẫn nằm trong payload như baseline hiện tại. Các call/observation của yêu cầu mới được nối sau đó bằng assistant/tool messages. Không nối nhãn target, snapshot, review hay focus_tools vào prompt.

Episode N calls kết thúc bình thường sinh N + 1 samples: N next-call và một DONE. Episode không có calls vẫn sinh một DONE. Episode có sys.ask_missing_info sinh sample dự đoán call hỏi rồi dừng; không thêm DONE sau call hỏi vì agent hiện tự dừng. Không dùng completion rỗng hoặc chuỗi `[]` thay DONE. Không học observation như output assistant.

Các ví dụ trong `schemas/examples/` chỉ minh họa định dạng; toàn bộ partition=example, semantic/call-policy review vẫn not_run. SFT examples không có nghĩa là Master đã được duyệt train. Production exporter phải chặn các bản ghi chưa qua gate.

TRL hỗ trợ [tool calling và conversational prompt-completion](https://huggingface.co/docs/trl/dataset_formats#tool-calling). Bắt đầu bằng completion-only loss. Trước GPU run, khóa transformers/TRL/datasets versions và Qwen model/tokenizer revision; render 10–20 mẫu đại diện và decode vùng label khác -100. Kiểm tra prefix assistant dùng [Qwen3 no thinking](https://huggingface.co/Qwen/Qwen3-4B#enable_thinkingfalse), không có reasoning của teacher, đúng tool-call serialization, có token kết thúc cần học và prompt inference tương thích. Chưa xác minh training stack trong repo này.

Với arguments/tool schemas có kiểu JSON khác nhau, cấu hình feature JSON theo phiên bản datasets đã chọn; đừng ép mọi arguments vào một Arrow struct cố định. Không cài stack mới chỉ để áp dụng một ví dụ API từ tài liệu phiên bản khác.

Tính token bằng tokenizer Qwen trên prompt đã render đầy đủ. Budget pilot đề xuất là 16.384 token tổng và chừa 512 token cho completion; đây là bound thiết kế, chưa phải đo token hoặc xác minh luật thi mới. Nếu vượt, loại hoặc thiết kế lại episode; không cắt mù history, tools hay completion. Decision samples lặp prefix nên phải báo cáo tổng token và phân phối số quyết định/episode, không chỉ số dòng. Dùng sampling theo episode/family để episode dài không vô tình chiếm quá nhiều trọng số.

## Generator và acceptance gate

1. Chọn family/variant và scene seed; ấn định split_group trước mọi paraphrase/counterfactual.
2. Dựng yêu cầu/context/history/snapshot và mục tiêu có thể kiểm chứng. Card xác định phần độc lập/phụ thuộc, đối tượng và phạm vi mutation.
3. Teacher action rollout đề xuất một call, runtime validate rồi thực thi và trả observation thật. Không để teacher viết observation; không dùng target hidden để chọn ID.
4. Dừng đúng DONE hoặc sys.ask_missing_info; fail nếu vượt cap. `ok=true, done=false`/kết quả rỗng cần diễn giải theo mục tiêu; không tự gắn nhãn thành công hoặc tự loại tất cả.
5. Kiểm tra structure, replay, causal evidence, semantic goal/subgoal, side effects, call policy và terminal. Không bỏ/thêm read chỉ để rút ngắn trajectory; call policy tham chiếu train và bằng chứng thực tế, không khớp mù mọi gold nghi lỗi.
6. Review canary thủ công, giữ evidence. Chỉ xuất train nếu bảy checks đều pass và review approved. `not_run` không đồng nghĩa pass. Gate phải do pipeline thực thi; schema không ngăn teacher tự ghi chữ pass nên không tin các cờ do teacher gửi.
7. Xuất SFT và kiểm tra template/loss mask/token budget trước fine-tune.

Schema-valid không chứng minh semantic-valid. Replay-valid không chứng minh call sequence tối ưu F1. Canonical state hash có thể bỏ text nên luôn kiểm tra arguments/free-text và raw state assertions khi cần. Semantic judge cloud chỉ là một tín hiệu; các mẫu mơ hồ hoặc có hành động ngoài yêu cầu cần review hoặc quarantine.

## Pilot và split

Canary đầu: 32 episode, bốn mẫu mỗi family. Đây là bước tìm lỗi generator/validator; chưa train. Sau canary đã review và kiểm chứng mới mở ngân sách 600 episode được chấp nhận theo config. Các coverage targets là tags giao nhau, không cộng thành 600 và chưa có bằng chứng đây là tỷ lệ tối ưu.

Scratch synthetic có thể chia khoảng 480 train / 120 synthetic_val theo connected lineage. Một counterfactual sibling ở train thì mọi paraphrase và parent cùng group cũng ở train. Không hứa template-disjoint nếu vẫn dùng chung template tổng quát.

Nếu synthetic bắt nguồn trực tiếp từ một BTC train sample, giữ toàn bộ descendants ở train. Nếu muốn dùng descendants làm synthetic_val thì phải loại cả seed và họ hàng khỏi real training; nếu không, validation bị lộ qua parent. Dev BTC giữ cố định và không làm seed sinh dữ liệu. Challenge holdout app/tool có thể làm ở vòng sau bằng một protocol riêng trong phần train; chưa thay split hiện tại.

So sánh B0, SFT real-only và SFT real+synthetic trên cùng dev/config/optimization budget; báo cáo cả token đã train, tỷ lệ real/synthetic, seed và family sampling. Đánh giá state match, exact-call F1 chẩn đoán, read/write calls thừa/thiếu, ask/no-action, nguồn ID, ngày giờ, số vòng model và latency. Exact-call local chưa phải scorer fuzzy BTC; timing phát triển chưa chứng minh runtime thi.

Không mở rộng sang mọi app ngay. Sau pilot, chọn family cần bổ sung từ lỗi dev ở mức nhóm; không paraphrase lại chính các câu dev. Capability mới chỉ từ catalog public có thể thử ở giai đoạn riêng bằng state tự tạo và runtime đúng phiên bản, không dùng public requests/labels/environments; tách provenance và báo cáo rõ.

## Kiểm tra hiện có và cách chạy

```bash
python3 scripts/audit_synthetic_tools.py
python3 -m pip install -r requirements-schema.txt
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_dataset_spec.py' -v
```

Schema tests kiểm tra Draft 2020-12, fixture hợp lệ, một số mẫu sai cấu trúc, replay observation/hash và prefix SFT không chứa observation tương lai. Tests dùng fixture tự tạo với runtime train local; không gọi cloud/GPU và không đánh giá chất lượng Qwen. Cần có data/runtime BTC local và dependency jsonschema để chạy đầy đủ.

Chưa chạy: generation cloud, production semantic/causal validator, token/template/loss-mask smoke bằng tokenizer Qwen, fine-tune và đo chất lượng/latency. Không coi bốn fixture minh họa là canary đã hoàn tất.
