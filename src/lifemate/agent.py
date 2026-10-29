"""Single-session, sequential tool agent; no labels or direct state access."""
from __future__ import annotations

import json
import re
import time

from .data import catalog_map


SYSTEM = """Bạn là trợ lý LifeMate thực thi yêu cầu tiếng Việt bằng công cụ.
Chỉ giải quyết yêu cầu MỚI NHẤT dựa trên context và history. History là các lượt
đã hoàn tất; môi trường đã chứa trạng thái sau history. Không chạy lại các lệnh cũ.
Chỉ dùng công cụ được cung cấp. Mỗi lần chỉ gọi MỘT công cụ, chờ kết quả rồi
quyết định bước tiếp theo. Không bịa ID, kết quả tìm kiếm, thời gian hay tham số.
Lấy ID từ công cụ đọc/tìm kiếm, hoặc từ kết quả công cụ đã có trong history.
Ngày tương đối dựa trên context.current_time và timezone, không dựa vào ngày hệ thống.
Đọc kỹ điều kiện, phủ định, sửa đổi và tham chiếu các lượt trước. Không gọi công cụ
thừa để kiểm tra lại nếu đã đủ dữ kiện. Kết quả ok=true, done=false là chưa thực hiện
được hành động; đọc message để xử lý, không tự coi là thành công.
Nếu thiếu thông tin cần thiết mà không thể tra từ context/history/công cụ, gọi
sys.ask_missing_info với tool_name và params đúng schema; sau đó dừng.
Nếu không có ứng dụng hỗ trợ yêu cầu, không gọi công cụ nào. Khi hoàn tất, trả đúng
DONE, không giải thích và không gọi thêm công cụ. Dùng định dạng tool calling chuẩn.
"""


def tools_for(row, catalog):
    available = {t for app in row["context"]["installed_apps"].values() for t in app["tools"]}
    available.add("sys.ask_missing_info")
    specs = catalog_map(catalog)
    missing = available - specs.keys()
    if missing:
        raise ValueError(f"Installed tools absent from catalog: {sorted(missing)}")
    return [{"type": "function", "function": {"name": name,
             "description": specs[name].get("description", ""),
             "parameters": specs[name]["parameters"]}} for name in sorted(available)]


def make_messages(row, transcripts):
    history = []
    for turn in row.get("history", []):
        if turn["role"] == "user":
            history.append({"role": "user", "text": transcripts[turn["audio"]]})
        else:
            history.append({k: turn[k] for k in ("role", "text", "tool_calls", "tool_results") if k in turn})
    payload = {"context": row["context"], "history": history, "latest_request": transcripts[row["audio"]]}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def parse_action(text):
    if text.strip() == "DONE":
        return None
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
    if len(matches) != 1:
        raise ValueError("Return exactly ONE <tool_call>JSON</tool_call>, or DONE")
    # Do not accept a truncated second call or unrelated text outside the block.
    if re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip():
        raise ValueError("Unexpected text outside tool call")
    call = json.loads(matches[0])
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
        raise ValueError("Tool call requires name:string and arguments:object")
    return {"name": call["name"], "arguments": call["arguments"]}


def run_agent(row, transcripts, env, catalog, model, deadline, max_calls=8, max_repairs=2):
    messages = make_messages(row, transcripts)
    tools = tools_for(row, catalog)
    trace, repairs = [], 0
    reason = "call_limit"
    for _ in range(max_calls + max_repairs + 1):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            reason = "timeout"
            break
        output, generation = model.generate(messages, tools, remaining)
        step = {"output": output, "generation": generation}
        trace.append(step)
        messages.append({"role": "assistant", "content": output})
        if time.perf_counter() >= deadline:
            reason = "timeout"
            break
        try:
            if generation.get("output_cap_reached"):
                raise ValueError("Output reached token cap; no potentially truncated action was executed")
            call = parse_action(output)
            if call is None:
                reason = "done"
                break
            if len(env.get_tool_calls()) >= max_calls:
                reason = "call_limit"
                break
            check = env.validate(call["name"], call["arguments"])
            if not check["ok"]:
                raise ValueError(check["error"])
        except (ValueError, TypeError, KeyError) as error:
            step["validation_error"] = str(error)
            repairs += 1
            if repairs > max_repairs:
                reason = "repair_limit"
                break
            messages.append({"role": "user", "content": "Chưa chạy công cụ. Sửa lỗi định dạng/schema: " + str(error)})
            continue
        # Exactly one session; no rollback, no hidden calls, no direct reads of state.
        result = env.call(call["name"], call["arguments"])
        step["tool_result"] = result
        messages.append({"role": "tool", "name": call["name"],
                         "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))})
        if call["name"] == "sys.ask_missing_info":
            reason = "missing_info"
            break
    return {"id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}, {
        "stop_reason": reason, "trace": trace, "repairs": repairs,
    }
