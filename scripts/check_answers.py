"""验收比赛输出，或在本地对比参考标注。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d01.contracts import Schema, Task1
from d01.evaluation import (INSTANCES, TIMELINES, Answers, align_generated_ids, candidate_coverage, catalog_extension_candidates, comparison_differences,
                            comparison_evidence_scores, instance_differences, pilot_group_reports, scores,
                            split_group_reports)
from d01.inputs import Sources, material_digest, write_json
from d01.workspace import schema_version


TASK2_OUTPUTS = ("task2/labels-t1.json", "task2/labels-t2.json")
TASK2_METRICS = {"instance_rows", "instance_fields", "instance_links", "timeline_values", "pseudo_changes"}


def load_task2(answers: Path, run: Path) -> Answers:
    """用本轮定稿的表结构校验任务二实例和时序。"""
    saved = json.loads((run / "schema.json").read_text(encoding="utf-8"))
    schema = Schema.model_validate(saved["model"])
    if schema_version(schema) != saved["version"]:
        raise ValueError("表结构内容与本轮保存的版本不一致")
    instances, timelines = ((answers / name).read_bytes() for name in TASK2_OUTPUTS)
    return Answers(Task1(model=schema, conflicts=[]), INSTANCES.validate_json(instances),
                   TIMELINES.validate_json(timelines))


def inspect_run(run: Path, answers: Path, materials: dict[str, dict[str, str]], task: str) -> tuple[dict, dict[str, list[set[str]]]]:
    """读取这次运行的候选窗口和统计，并检查窗口使用了本次评估的来源。"""
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    if not summary["complete"] or settings["task"] != task:
        raise ValueError("请选择已成功生成所选任务输出的运行目录")
    input_digests = {task: material_digest(texts) for task, texts in materials.items()}
    if input_digests != json.loads((run / "inputs.json").read_text(encoding="utf-8")):
        raise ValueError("评分输入与该次运行使用的材料不一致")
    outputs = TASK2_OUTPUTS if task == "task2" else ("task1/labels.json", *TASK2_OUTPUTS)
    output_digests = {name: sha256((answers / name).read_bytes()).hexdigest() for name in outputs}
    if output_digests != json.loads((run / "outputs.json").read_text(encoding="utf-8")):
        raise ValueError("评分答案与该次运行生成的文件不一致")
    candidates = {}
    for kind in (("task2",) if task == "task2" else materials):
        rows = json.loads((run / kind / "candidates.json").read_text(encoding="utf-8"))
        candidates[kind] = [set(row["refs"]) for row in rows]
        if any(refs - materials[kind].keys() for refs in candidates[kind]):
            raise ValueError(f"{task} 的候选窗口使用了评估范围之外的来源")
    details = {"elapsed_seconds": summary["elapsed_seconds"], "counts": summary["counts"],
               "requests": sum(summary["calls"].values()), "cache_hits": sum(summary["cache_hits"].values()),
               "observed_tokens": sum(summary["total_tokens"].values()),
               "settings": settings, "input_digests": input_digests}
    return details, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "task2"), default="all")
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--pilot", action="store_true", help="将完整标准答案按首轮输入限定范围")
    parser.add_argument("--holdout-from", type=Path, help="完整验证时传入首轮输入目录，另列首轮组和留出组结果")
    parser.add_argument("--work-run", type=Path, help="具体运行目录；比较筛选覆盖与调用量")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (args.pilot or args.holdout_from) and not args.gold:
        parser.error("分组评估需要 --gold")
    if args.pilot and args.holdout_from:
        parser.error("--pilot 和 --holdout-from 分别用于首轮和完整验证")
    if args.task == "task2" and not args.work_run:
        parser.error("单独验收任务二需要 --work-run，供评分时读取定稿的表结构")
    materials = {task: Sources.load(args.in_dir, task).texts for task in ("task1", "task2")}
    candidates = None
    report = {}
    if args.work_run:
        report["run"], candidates = inspect_run(args.work_run, args.answers, materials, args.task)
    answer = load_task2(args.answers, args.work_run) if args.task == "task2" else Answers.load(args.answers)
    answer.verify(args.in_dir)
    report.update(format_checked=True, instances=len(answer.instances), timelines=len(answer.timelines))
    if args.task == "all":
        report["conflicts"] = len(answer.task1.conflicts)
    if args.gold:
        reference = Answers.load(args.gold)
        if args.pilot:
            reference = reference.for_inputs(args.in_dir)
        reference.verify(args.in_dir)
        report["comparison"] = "对照参考标注做规范化精确匹配；目录扩展项单列，需回查原文"
        reference_key = [reference.task1.model_dump(), [row.model_dump() for row in reference.instances],
                         [row.model_dump() for row in reference.timelines]]
        report["reference_digest"] = sha256(json.dumps(reference_key, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        report["metrics"] = scores(answer, reference)
        report["catalog_extension_candidates"] = catalog_extension_candidates(answer, reference)
        report["comparison_evidence"] = comparison_evidence_scores(answer, reference)
        report["comparison_differences"] = comparison_differences(answer, reference)
        report["instance_differences"] = instance_differences(answer, reference)
        aligned, counts = align_generated_ids(answer, reference, materials["task2"])
        report["instance_alignment"] = {"changes": counts, "metrics": {
            key: value for key, value in scores(aligned, reference, record_bound_fields=True).items()
            if key in {"instance_rows", "instance_fields", "instance_links"}
        }}
        if candidates is not None:
            report["candidate_coverage"] = candidate_coverage(reference, candidates)
        if args.pilot:
            report["pilot_groups"] = pilot_group_reports(aligned, reference, materials, candidates)
        if args.holdout_from:
            pilot_ids, pilot_digests = {}, {}
            for task, texts in materials.items():
                trial = Sources.load(args.holdout_from, task).texts
                if any(texts.get(ref) != text for ref, text in trial.items()):
                    raise ValueError(f"{task} 的试跑材料与完整输入不一致")
                pilot_ids[task] = set(trial)
                pilot_digests[task] = material_digest({ref: text for ref, text in texts.items() if ref in trial})
            report["group_splits"] = split_group_reports(aligned, reference, materials, pilot_ids, candidates)
            report["pilot_digests"] = pilot_digests
        if args.task == "task2":
            keep_task2_scores(report)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def keep_task2_scores(report: dict) -> None:
    """只保存本轮实际评估的任务二项目。"""
    def select(section: dict) -> None:
        section["metrics"] = {key: value for key, value in section["metrics"].items() if key in TASK2_METRICS}
        section["comparison_evidence"].pop("conflict_source_sets")
        section["comparison_differences"].pop("conflicts")

    select(report)
    if "pilot_groups" in report:
        groups = report["pilot_groups"]
        groups.pop("task1")
        for counts in groups["cross_group_findings"].values():
            counts.pop("conflicts")
    if "group_splits" in report:
        splits = report["group_splits"]
        for name in ("pilot_groups", "held_out_groups"):
            select(splits[name])
            splits[name]["sources"].pop("task1")
        for counts in splits["cross_split_findings"].values():
            counts.pop("conflicts")


if __name__ == "__main__":
    main()
