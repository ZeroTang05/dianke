"""比较两份由 check_answers.py 生成的报告，验证每次只改一个开关。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d01.inputs import write_json


def delta(base: dict, trial: dict, fields: tuple[str, ...]) -> dict:
    """每项同时保留两轮数值和试验轮减去基准轮的差。"""
    result = {}
    for field in fields:
        a, b = base[field], trial[field]
        change = None if a is None or b is None else b - a
        result[field] = {"base": a, "trial": b, "delta": change}
    return result


def metric_deltas(base: dict, trial: dict) -> dict:
    if base.keys() != trial.keys():
        raise ValueError("两份报告的评分项目不同")
    return {metric: delta(values, trial[metric], ("matched", "missing", "extra", "f1"))
            for metric, values in base.items()}


def coverage_deltas(base: dict, trial: dict) -> dict:
    if base.keys() != trial.keys() or any(v["expected"] != trial[k]["expected"] for k, v in base.items()):
        raise ValueError("两份报告的候选覆盖统计口径不同")
    return {key: {"expected": row["expected"], **delta(row, trial[key], ("covered",))}
            for key, row in base.items()}


def compare(base: dict, trial: dict) -> dict:
    """先核对比较范围，再汇总原始评分、留出组和调用量。"""
    if not all(report.get("format_checked") and {"run", "reference_digest"} <= report.keys()
               for report in (base, trial)):
        raise ValueError("需要两份由 --gold 和 --work-run 生成的评分报告")
    left, right = base["run"], trial["run"]
    if left["input_digests"] != right["input_digests"]:
        raise ValueError("两轮运行的输入不同")
    if base["reference_digest"] != trial["reference_digest"]:
        raise ValueError("两轮评分的标准答案不同")
    a, b = left["settings"], right["settings"]
    if {k: v for k, v in a.items() if k != "options"} != {k: v for k, v in b.items() if k != "options"}:
        raise ValueError("源码版本、模型设置或并发数不同")
    if a["options"].keys() != b["options"].keys():
        raise ValueError("两轮的实验开关集合不同")
    changes = [(key, value, b["options"][key]) for key, value in a["options"].items() if value != b["options"][key]]
    if len(changes) != 1:
        raise ValueError(f"每轮应只改变一个开关，当前改变了 {len(changes)} 个")
    key, before, after = changes[0]

    steps = {"base": left["requests"] + left["cache_hits"], "trial": right["requests"] + right["cache_hits"]}
    result = {
        "changed_option": {"name": key, "base": before, "trial": after},
        "metrics": metric_deltas(base["metrics"], trial["metrics"]),
        "aligned_instances": metric_deltas(base["instance_alignment"]["metrics"], trial["instance_alignment"]["metrics"]),
        "candidate_coverage": coverage_deltas(base["candidate_coverage"], trial["candidate_coverage"]),
        "observed_run": {**delta(left, right, ("requests", "cache_hits", "observed_tokens", "elapsed_seconds")),
                         "requests_plus_cache_hits": {**steps, "delta": steps["trial"] - steps["base"]}},
    }
    split_key = "group_splits"
    if (split_key in base) != (split_key in trial):
        raise ValueError("完整验证的两份报告都需要使用同一个 --holdout-from")
    if split_key in base:
        if base["pilot_digests"] != trial["pilot_digests"]:
            raise ValueError("两份报告使用了不同的试跑分组")
        result[split_key] = {}
        for name in ("pilot_groups", "held_out_groups"):
            a_group, b_group = base[split_key][name], trial[split_key][name]
            if a_group["sources"] != b_group["sources"]:
                raise ValueError("两份报告的留出分组范围不同")
            result[split_key][name] = {"metrics": metric_deltas(a_group["metrics"], b_group["metrics"]),
                                       "candidate_coverage": coverage_deltas(a_group["candidate_coverage"], b_group["candidate_coverage"])}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="基准报告")
    parser.add_argument("trial", type=Path, help="试验报告")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        base, trial = (json.loads(path.read_text(encoding="utf-8")) for path in (args.base, args.trial))
        result = compare(base, trial)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"对照检查失败：{exc}\n")
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
