"""离线核对已完成抽取的结构和实例，可检查尚未导出结果的运行。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d01.contracts import Schema, Task1, validate_instances
from d01.evaluation import Answers, align_generated_ids, instance_differences, scores
from d01.graph import Graph
from d01.inputs import Sources, material_digest, write_json
from d01.model import ModelClient, ModelConfig, ModelError
from d01.records import Locator, Record, RecordDraft
from d01.workspace import schema_version


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect(run: Path, root: Path, gold: Path, cache: Path, endpoint: str) -> dict:
    """只使用本次输入、定稿结构、审核过的实例及同配置的响应缓存。"""
    sources = {task: Sources.load(root, task) for task in ("task1", "task2")}
    if {task: material_digest(src.texts) for task, src in sources.items()} != read(run / "inputs.json"):
        raise ValueError("材料与保存的运行不一致")
    schema_data = read(run / "schema.json")
    schema = Schema.model_validate(schema_data["model"])
    version = schema_version(schema)
    if version != schema_data["version"]:
        raise ValueError("定稿结构与保存的版本不一致")
    instance_data = read(run / "task2" / f"records-{version}.json")
    if version != instance_data["schema_version"]:
        raise ValueError("实例抽取与定稿结构版本不一致")

    settings = read(run / "settings.json")["model"]
    endpoint = endpoint.rstrip("/")
    if sha256(endpoint.encode()).hexdigest() != settings["endpoint_digest"]:
        raise ValueError("指定接口地址与该次运行不一致")
    config = ModelConfig(endpoint, settings["name"], "", settings["json_schema"], settings["thinking"],
                         settings["rpm"], settings["timeout"], settings["context_chars"])
    client = ModelClient(config, cache, cache_only=True)
    task2 = sources["task2"]
    reviews = {row["slot"]: row["accepted"] for row in instance_data["reviews"]}
    records = []
    for row in instance_data["records"]:
        locator = Locator.model_validate(row["locator"])
        draft = RecordDraft.model_validate(row["facts"])
        if not reviews[row["slot"]] or not set(locator.refs) <= task2.texts.keys() or any(
            not quote.check(task2) for fact in (*draft.facts, *draft.proposed_fields, *draft.connections)
            for quote in fact.evidence
        ):
            raise ValueError(f"实例 {row['slot']} 的审核或来源校验未通过")
        records.append(Record(row["slot"], locator, draft))
    graph = Graph(records, schema)
    items = graph.instances(task2, client)
    validate_instances(items, schema)
    checked = Answers(Task1(model=schema, conflicts=[]), items, [])
    expected = Answers.load(gold).for_inputs(root)
    expected.verify(root)
    aligned, changes = align_generated_ids(checked, expected, task2.texts)
    schema_keys = ("object_types", "properties", "link_types")
    instance_keys = ("instance_rows", "instance_fields", "instance_links")
    strict = scores(checked, expected)
    normalized = scores(aligned, expected, record_bound_fields=True)
    return {"scope": "表结构、实例及关联；不含冲突或时序成绩", "source_run": run.name,
            "instances": len(items), "unresolved_links": len(graph.unresolved),
            "strict": {key: strict[key] for key in (*schema_keys, *instance_keys)},
            "aligned": {key: normalized[key] for key in instance_keys}, "alignment_changes": changes,
            "instance_differences": instance_differences(checked, expected),
            "cache_hits": sum(client.cached.values()), "requests": sum(client.calls.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-run", type=Path, required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = inspect(args.work_run, args.in_dir, args.gold, args.cache_dir, args.base_url)
    except (OSError, ValueError, ModelError) as exc:
        parser.exit(1, f"离线核对失败：{exc}\n")
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
