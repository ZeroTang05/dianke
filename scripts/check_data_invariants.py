"""使用官方验证答案检查材料隔离、记录归属和反向连接。"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d01.contracts import Property, Schema, instance_report_date
from d01.discovery import Discovery, Findings
from d01.evaluation import (Answers, align_generated_ids, catalog_extension_candidates, comparison_evidence_scores,
                            pilot_group_reports, scores,
                            split_group_reports)
from d01.graph import Graph
from d01.inputs import Sources
from d01.records import Fact, LinkChoice, Locator, Record, RecordDraft, RecordSelection, connection_options, resolve_connections
from d01.schema_stage import EXAMPLE
from d01.stages import Evidence
from d01.vocabulary import canonicalize_schema


def check_terminology(schema: Schema) -> dict:
    """用实际结构验证更名对关系的影响，以及重名字段的处理。"""
    aliases = {"hz": "患者主档", "jzjl": "就诊", "jc": "检查项目", "jcjl": "检查记录"}
    ids = {obj.typeCode: f"local_{obj.typeCode}" for obj in schema.object_types}
    fields = {obj.typeCode: {code: f"local_{code}" for code in obj.properties} for obj in schema.object_types}
    renamed = Schema(
        object_types=[obj.model_copy(update={
            "name": aliases.get(obj.typeCode, obj.name), "typeCode": ids[obj.typeCode],
            "properties": {fields[obj.typeCode][code]: prop for code, prop in obj.properties.items()},
            "primaryKey": fields[obj.typeCode][obj.primaryKey],
            "displayKey": fields[obj.typeCode][obj.displayKey] if obj.displayKey else None,
            "foreignKeys": [fields[obj.typeCode][code] for code in obj.foreignKeys],
        }) for obj in schema.object_types],
        link_types=[link.model_copy(update={
            "sourceTypeCode": ids[link.sourceTypeCode], "targetTypeCode": ids[link.targetTypeCode],
            "sourceProperty": fields[link.sourceTypeCode][link.sourceProperty],
            "targetProperty": fields[link.targetTypeCode][link.targetProperty],
        }) for link in schema.link_types],
    )
    assert not renamed.issues()
    result, changes = canonicalize_schema(renamed)
    assert result == schema and changes
    assert canonicalize_schema(result) == (result, [])
    exam = schema.object_index()["jcjl"]
    # 两个标题都可能映射到「报告日期」，保留两个字段以供模型分辨口径。
    exam_with_alternative = exam.model_copy(update={"properties": {
        **exam.properties, "bgsj": Property(title="报告时间", type="string"),
        "lylx": Property(title="来源类型", type="string"),
    }})
    variation = schema.model_copy(update={"object_types": [exam_with_alternative if obj is exam else obj
                                                           for obj in schema.object_types]})
    assert canonicalize_schema(variation) == (variation, [])
    for value in ("2025-10-03", "2025-10-03 14:25", "2025-10-03T14:25:30"):
        assert instance_report_date(instance_report_date(value)) == instance_report_date(value)
    assert instance_report_date("2025-10-03 14:25") == "2025-10-03T14:25:00"
    return {"renames": len(changes), "links_preserved": len(result.link_types), "ambiguous_fields_preserved": 2}


def check_catalog_extensions(reference: Answers) -> dict:
    """用赛题用药示例验证目录扩展能单列，已有字段的别名不会被误列。"""
    base = reference.task1.model
    example = Schema.model_validate(EXAMPLE["结果"])
    # 保留验证集的结构，补入赛题示例中的两张表。
    schema = Schema(object_types=base.object_types + example.object_types, link_types=base.link_types + example.link_types)
    assert not schema.issues()
    prediction = Answers(reference.task1.model_copy(update={"model": schema}), reference.instances, reference.timelines)
    result = catalog_extension_candidates(prediction, reference)
    assert {row["type_code"] for row in result["object_types"]} == {"yp", "yzyy"}
    assert {row["type_code"] for row in result["properties"]} == {"yp", "yzyy"}
    assert [(link["source"], link["target"]) for link in result["link_types"]] == [("yzyy.ypbh", "yp.ypbh")]
    assert scores(prediction, reference)["object_types"]["extra"] == 2
    exam = base.object_index()["jcjl"]
    aliased = exam.model_copy(update={"properties": {
        **exam.properties, "bgrq": exam.properties["bgrq"].model_copy(update={"title": "报告时间"}),
    }})
    alias_schema = base.model_copy(update={"object_types": [aliased if obj is exam else obj for obj in base.object_types]})
    alias_prediction = Answers(reference.task1.model_copy(update={"model": alias_schema}), reference.instances,
                               reference.timelines)
    assert not any(catalog_extension_candidates(alias_prediction, reference).values())
    return {category: len(items) for category, items in result.items()}


def check_graph(gold: Answers, materials: Sources) -> dict:
    """用标准实例的实际编号和引用检查生产代码的跨组关联。"""
    schema = gold.task1.model
    types = schema.object_index()
    roles = {"hz": "patient", "jc": "shared", "jzjl": "event", "jcjl": "event"}
    visits = {row.data["jzbh"]: row for row in gold.instances if row.typeCode == "jzjl"}
    source_by_visit = defaultdict(list)
    for visit in visits:
        source_by_visit[visit] = [ref for ref, text in materials.texts.items() if visit in text]
    records = []
    for i, row in enumerate(gold.instances, 1):
        kind = row.typeCode
        key = row.data[types[kind].primaryKey]
        refs = source_by_visit.get(row.data.get("jzbh")) or []
        if not refs and kind == "hz":
            refs = [ref for ref, text in materials.texts.items() if row.data["zjhm"] in text]
        if not refs and kind == "jc":
            refs = [ref for ref, text in materials.texts.items() if row.data["jcxmmc"] in text]
        if not refs:
            raise ValueError(f"标准实例 {i} 找不到对应材料")
        # 这里使用官方已审定实例作为已抽取的图输入；模型推理另用真实接口验收。
        facts = [Fact(field=field, raw_value=str(value), value=value,
                      evidence=[Evidence(ref=refs[0], quote=str(value))])
                 for field, value in row.data.items() if field == types[kind].primaryKey or field in row.relations]
        records.append(Record(str(i), Locator(type_code=kind, label=str(key), scope=roles[kind],
                                              identity=str(key), refs=refs),
                              RecordDraft(facts=facts, connections=[], proposed_fields=[])))
    graph = Graph(records, schema)
    slots = {row.slot: row.locator for row in records}
    exam = next(row for row in records if row.locator.type_code == "jcjl")
    options = connection_options(exam.slot, slots, schema)
    assert {(field, slots[target].type_code, target_field) for field, target, target_field in options.values()} == {
        ("jzbh", "jzjl", "jzbh"), ("jcbh", "jc", "jcbh"),
    }
    owner_fact = next(fact for fact in exam.draft.facts if fact.field == "jzbh")
    option = next(key for key, target in options.items() if target[0] == "jzbh" and slots[target[1]].identity == owner_fact.value)
    selection = RecordSelection(facts=exam.draft.facts, connections=[LinkChoice(option=option, evidence=owner_fact.evidence)],
                                proposed_fields=[])
    link, = resolve_connections(selection, options).connections
    assert (link.field, link.target_slot, link.target_field) == options[option]
    exam_groups = defaultdict(set)
    patient_exam = defaultdict(set)
    for node in graph.nodes:
        if node[0] != "jcjl":
            continue
        owners = {key for key in graph.ancestors(node) if key[0] == "hz"}
        projects = {key for links in graph.links[node].values() for key, _ in links if key[0] == "jc"}
        assert len(owners) == len(projects) == 1
        patient, exam = next(iter(owners)), next(iter(projects))
        exam_groups[exam].add(patient)
        patient_exam[patient, exam].add(node)
    assert any(len(owners) > 1 for owners in exam_groups.values())
    assert any(len(checks) > 1 for checks in patient_exam.values())
    seen = Counter(check for checks in patient_exam.values() for check in checks)
    assert len(seen) == sum(row.typeCode == "jcjl" for row in gold.instances)
    assert all(count == 1 for count in seen.values())
    groups = [refs for refs in source_by_visit.values() if refs]
    discovery = Discovery(Findings(pattern=None, signals=[], priority_refs=[]), list(materials.texts), groups)
    assert all(len(candidate.patients) == 1 for candidate in graph.candidates(materials, discovery, "indexed"))
    # 从官方实例中移除一条真实就诊，检查受影响的检查记录能否按来源追查。
    project_by_record = {node: links for node, links in graph.links.items() if node[0] == "jcjl"}
    record_node = next(iter(project_by_record))
    fk, visit_node = next((field, target) for field, links in project_by_record[record_node].items()
                          for target, _ in links if target[0] == "jzjl")
    visit_slots = {row.slot for row in graph.nodes[visit_node]}
    check_slots = {row.slot for row in graph.nodes[record_node]}
    incomplete = Graph([row for row in records if row.slot not in visit_slots], schema)
    issue = next(item for item in incomplete.unresolved if set(item["slots"]) == check_slots and item.get("field") == fk)
    assert issue["reason"] == "外键原文值找不到目标"
    assert set(issue["sources"]) == {ref for row in graph.nodes[record_node] for ref in row.locator.refs}
    assert not incomplete.links[record_node].get(fk)
    return {"exam_records": len(seen), "cross_patient_shared_projects": sum(len(owners) > 1 for owners in exam_groups.values()),
            "patient_project_timeline_groups": sum(len(checks) > 1 for checks in patient_exam.values()),
            "missing_visit_reported": True}


def check_evaluation(gold: Answers, pilot: Answers, first: Sources, second: Sources, root: Path) -> dict:
    """用真实标准实例做编号替换实验，检查编号变化是否只影响原始分数。"""
    rows = []
    # 保留所有官方字段与关系，只替换患者内部编号及引用它的字段。
    for row in gold.instances:
        data = dict(row.data)
        relations = dict(row.relations)
        if row.typeCode == "hz":
            data["hzbh"] = "LOCAL_CHECK_" + data["hzbh"]
        for field, link in relations.items():
            if link.targetTypeCode == "hz" and link.targetProperty == "hzbh":
                data[field] = "LOCAL_CHECK_" + link.targetPropertyVal
                relations[field] = link.model_copy(update={"targetPropertyVal": data[field]})
        rows.append(row.model_copy(update={"data": data, "relations": relations}))
    renumbered = Answers(gold.task1, rows, gold.timelines)
    renumbered.verify(root / "full")
    assert scores(renumbered, gold)["instance_links"]["f1"] < 1
    aligned, counts = align_generated_ids(renumbered, gold, second.texts)
    assert counts["patient"] == sum(row.typeCode == "hz" for row in gold.instances)
    assert counts["ambiguous_identities"] > 0
    assert aligned.instances == gold.instances

    materials = {first.task: first.texts, second.task: second.texts}
    pilot_texts = {task: Sources.load(root / "pilot", task).texts for task in materials}
    groups = pilot_group_reports(pilot, pilot, pilot_texts, None)
    assert len(groups["task1"]) == len(groups["task2"]) == 3
    assert groups["task1"]["M000703"]["metrics"]["conflicts"]["expected"] == 2
    assert groups["task2"]["M000235"]["metrics"]["pseudo_changes"]["expected"] == 1
    splits = split_group_reports(gold, gold, materials, {task: set(texts) for task, texts in pilot_texts.items()}, None)
    assert splits["held_out_groups"]["metrics"]["timeline_values"]["expected"] == 2
    assert splits["held_out_groups"]["comparison_evidence"]["timeline_evidence_by_day"]["matched"] == 2
    assert not splits["held_out_groups"]["comparison_differences"]["timeline_values"]["missing"]
    assert splits["cross_split_findings"]["reference"] == {"conflicts": 0, "timelines": 0}

    # 用真实原文中的报告时分检验：按天比较不改变原始精度的统计。
    timed = []
    for timeline in pilot.timelines:
        states = []
        for state in timeline.values:
            exact = next((value for value in (state.time + " 13:23", state.time + " 17:58")
                          if value in pilot_texts["task2"][state.ref]), state.time)
            states.append(state.model_copy(update={"time": exact}))
        timed.append(timeline.model_copy(update={"values": states}))
    evidence = comparison_evidence_scores(Answers(pilot.task1, pilot.instances, timed), pilot)
    assert evidence["timeline_evidence"]["matched"] < evidence["timeline_evidence_by_day"]["matched"]
    assert evidence["timeline_evidence_by_day"]["matched"] == evidence["timeline_source_days"]["expected"]
    assert evidence["pseudo_change_source_pairs"]["matched"] == evidence["pseudo_change_evidence"]["expected"]
    return {"renumbered_patients_aligned": counts["patient"],
            "ambiguous_identities": counts["ambiguous_identities"],
            "pilot_groups": len(groups["task1"]) + len(groups["task2"]),
            "held_out_sources": splits["held_out_groups"]["sources"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="dev_data.py 生成的目录")
    args = parser.parse_args()
    first, second = (Sources.load(args.root / "full", task) for task in ("task1", "task2"))
    both = first.texts.keys() & second.texts.keys()
    distinct_materials = {(source.task, ref): value for source in (first, second) for ref, value in source.texts.items()}
    assert both and any(first.texts[ref] != second.texts[ref] for ref in both)
    assert len(distinct_materials) == len(first.texts) + len(second.texts)

    gold = Answers.load(args.root / "gold")
    gold.verify(args.root / "full")
    pilot = gold.for_inputs(args.root / "pilot")
    pilot.verify(args.root / "pilot")
    print(json.dumps({"shared_source_ids": len(both), "separate_sources": len(distinct_materials),
                      "pilot_gold": {"conflicts": len(pilot.task1.conflicts), "instances": len(pilot.instances),
                                     "timelines": len(pilot.timelines)}, **check_graph(gold, second),
                      "evaluation": check_evaluation(gold, pilot, first, second, args.root),
                      "catalog_extensions": check_catalog_extensions(gold),
                      "terminology": check_terminology(gold.task1.model)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
