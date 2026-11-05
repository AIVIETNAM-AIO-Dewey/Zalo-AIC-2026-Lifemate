"""Build aggregate tool coverage from TRAIN IDs and catalog-only public metadata.

Never reads public-test rows/environments or exports transcripts, labels or states.
Counts earlier reads as observations, not as proven causal dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_map(path):
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    return {tool["name"]: tool for app in catalog["apps"] for tool in app["tools"]}


def build_report(data_root, ids_path, dev_ids_path):
    data_root = Path(data_root)
    train_catalog = data_root / "train/train_tools_catalog.json"
    train_rows = data_root / "train/train.jsonl"
    tools = tool_map(train_catalog)
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
    dev_ids = set(json.loads(Path(dev_ids_path).read_text(encoding="utf-8")))
    if len(ids) != len(set(ids)) or set(ids) & dev_ids:
        raise ValueError("Train IDs must be unique and disjoint from fixed dev IDs")
    selected = set(ids)
    rows = [json.loads(line) for line in train_rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["id"] in selected]
    if len(rows) != len(ids) or {row["id"] for row in rows} != selected:
        raise ValueError("Missing or duplicate train records")

    occurrences, samples, installed = Counter(), Counter(), Counter()
    preceding, immediate = {}, {}
    outcomes = Counter()
    for row in rows:
        available = {t for app in row["context"]["installed_apps"].values() for t in app["tools"]}
        installed.update(available | {"sys.ask_missing_info"})
        calls = row["label"]["tool_calls"]
        names = [call["name"] for call in calls]
        outcomes["no_action" if not names else "ask_missing_info" if "sys.ask_missing_info" in names else "execute"] += 1
        occurrences.update(names)
        samples.update(set(names))
        for index, name in enumerate(names):
            if tools[name].get("kind") != "write":
                continue
            # Each preceding read name counted at most once per write occurrence.
            earlier_reads = {n for n in names[:index] if tools[n].get("kind") == "read"}
            preceding.setdefault(name, Counter()).update(earlier_reads)
            if index and tools[names[index - 1]].get("kind") == "read":
                immediate.setdefault(name, Counter()).update([names[index - 1]])

    public_path = data_root / "public_test/public_test_tools_catalog.json"
    public_tools = tool_map(public_path) if public_path.exists() else {}
    shared = tools.keys() & public_tools.keys()
    changed = sorted(name for name in shared if tools[name]["parameters"] != public_tools[name]["parameters"])
    entries = []
    for name, spec in sorted(tools.items()):
        params = spec["parameters"]
        entries.append({
            "name": name,
            "app": spec.get("app", name.split(".")[0]),
            "kind": spec.get("kind"),
            "installed_train_samples": installed[name],
            "gold_train_samples": samples[name],
            "gold_train_call_occurrences": occurrences[name],
            "required": params.get("required", []),
            "x_one_of": params.get("x-one-of", []),
            "reference_parameters": {key: prop["x-ref"] for key, prop in params.get("properties", {}).items() if "x-ref" in prop},
            "earlier_reads_per_write_occurrence": dict(sorted(preceding.get(name, {}).items())),
            "immediate_reads_per_write_occurrence": dict(sorted(immediate.get(name, {}).items())),
        })
    return {
        "scope": "Fixed train IDs only; public catalog metadata only; no dev labels or test rows",
        "counts_are": "Call occurrences and sample counts are distinct. Earlier read co-occurrence is NOT a dependency proof.",
        "sources": {
            "train_jsonl_sha256": digest(train_rows),
            "train_ids_sha256": digest(ids_path),
            "dev_ids_sha256": digest(dev_ids_path),
            "train_catalog_sha256": digest(train_catalog),
            "public_catalog_sha256": digest(public_path) if public_tools else None,
        },
        "train_samples": len(rows),
        "outcomes": dict(sorted(outcomes.items())),
        "train_catalog_tools": len(tools),
        "tools_with_gold_train_calls": sum(samples[name] > 0 for name in tools),
        "train_catalog_tools_without_gold_train_calls": [name for name in sorted(tools) if not samples[name]],
        "public_catalog_only_tools": sorted(public_tools.keys() - tools.keys()),
        "shared_tools_with_parameter_schema_changes": changed,
        "tools": entries,
    }


def markdown(report):
    lines = [
        "# Tool coverage cho thiết kế synthetic dataset",
        "",
        f"Chỉ dùng {report['train_samples']} IDs train cố định. Không dùng nhãn dev hoặc dòng dữ liệu public test.",
        "Public chỉ được đọc catalog. Đây là thống kê cấu trúc, chưa phải kết quả model hay dependency đã kiểm chứng.",
        "",
        "Mỗi hàng là một tool. Occurrences đếm số call; samples đếm số hội thoại có call.",
        "`x-ref` là gợi ý nguồn ID; cần đọc runtime và kiểm tra provenance trước khi xây scenario.",
        "",
        "| Tool | Kind | Samples / calls | Installed samples | Required | Alternative groups | ID references |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for t in report["tools"]:
        required = ", ".join(t["required"]) or "—"
        alternatives = " / ".join(" + ".join(group) for group in t["x_one_of"]) or "—"
        refs = "; ".join(f"{key} → {value}" for key, value in t["reference_parameters"].items()) or "—"
        lines.append(f"| `{t['name']}` | {t['kind']} | {t['gold_train_samples']} / {t['gold_train_call_occurrences']} | {t['installed_train_samples']} | {required} | {alternatives} | {refs} |")
    lines.extend([
        "", "## Catalog public có thêm", "",
        "Chỉ là danh sách capability công khai để lập backlog. Pilot chưa sinh dữ liệu cho các tool này.", "",
        ", ".join(f"`{name}`" for name in report["public_catalog_only_tools"]), "",
        "## Schema tham số thay đổi giữa hai catalog", "",
        ", ".join(f"`{name}`" for name in report["shared_tools_with_parameter_schema_changes"]) or "Không có.", "",
        "Chi tiết counts trước/sát write và checksum nguồn nằm trong `tool_inventory.json` cùng thư mục.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--train-ids", type=Path, default=Path("reports/data_audit/train_ids.json"))
    parser.add_argument("--dev-ids", type=Path, default=Path("reports/data_audit/dev_ids.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/synthetic_design"))
    args = parser.parse_args()
    report = build_report(args.data_root, args.train_ids, args.dev_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tool_inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "TOOL_COVERAGE.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["train_samples", "train_catalog_tools", "tools_with_gold_train_calls", "outcomes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
