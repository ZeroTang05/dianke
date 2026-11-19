"""对照验证集计算本地分项结果和扩展项，不参与模型推理。"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable, Iterable
from unicodedata import normalize

from pydantic import TypeAdapter

from .contracts import (Instance, ObjectType, Scalar, Task1, Timeline, instance_joins, instance_report_date,
                        validate_conflicts, validate_instances, validate_timelines)
from .inputs import Sources
from .vocabulary import CATALOG


INSTANCES = TypeAdapter(list[Instance])
TIMELINES = TypeAdapter(list[Timeline])


@dataclass(frozen=True)
class Answers:
    task1: Task1
    instances: list[Instance]
    timelines: list[Timeline]

    @classmethod
    def load(cls, root: Path) -> "Answers":
        return cls(
            Task1.model_validate_json((root / "task1/labels.json").read_bytes()),
            INSTANCES.validate_json((root / "task2/labels-t1.json").read_bytes()),
            TIMELINES.validate_json((root / "task2/labels-t2.json").read_bytes()),
        )

    def verify(self, inputs: Path) -> None:
        validate_conflicts(self.task1, Sources.load(inputs, "task1").texts)
        validate_instances(self.instances, self.task1.model)
        validate_timelines(self.timelines, self.task1.model, Sources.load(inputs, "task2").texts)

    def for_inputs(self, inputs: Path) -> "Answers":
        """把官方答案投影到试跑材料。先定位明确出现的就诊，再保留关联记录。"""
        return self.for_sources(Sources.load(inputs, "task1").texts, Sources.load(inputs, "task2").texts)

    def for_sources(self, task1: dict[str, str], task2: dict[str, str]) -> "Answers":
        """截取验证组的答案；跨组共享目录只随本组实例的关联纳入。"""
        conflicts = [row for row in self.task1.conflicts if all(c.source_id in task1 for c in row.claims)]
        timelines = [row for row in self.timelines if timeline_refs(row) <= task2.keys()]
        objects = self.task1.model.object_index()
        texts = "\n".join(task2.values())
        visits, patients = set(), set()
        for i, row in enumerate(self.instances):
            obj = objects[row.typeCode]
            role = type_role(obj.name)
            if role == "visit":
                value = field_value(obj, row, {"就诊编号", "就诊号"})
            elif role == "patient":
                value = field_value(obj, row, {"证件号码", "证件号", "身份证号", "身份证号码"})
            else:
                continue
            if value is not None and appears_in_text(value, texts):
                (visits if role == "visit" else patients).add(i)

        # 先收集本组就诊的后续记录，再补上患者和共享项目。相同患者的其他就诊留在原组。
        index = instance_index(self.instances)
        descendants: dict[int, list[int]] = defaultdict(list)
        parents = [targets(row, index) for row in self.instances]
        for i, matches in enumerate(parents):
            for target in matches.values():
                if type_role(self.instances[target].typeName) not in {"patient", "project"}:
                    descendants[target].append(i)

        def expand(selected: set[int], edges: Callable[[int], Iterable[int]]) -> None:
            pending = list(selected)
            while pending:
                new = set(edges(pending.pop())) - selected
                selected.update(new)
                pending.extend(new)

        expand(visits, lambda i: descendants[i])
        selected = visits | patients
        expand(selected, lambda i: parents[i].values())
        return Answers(Task1(model=self.task1.model, conflicts=conflicts),
                       [row for i, row in enumerate(self.instances) if i in selected], timelines)


def timeline_refs(row: Timeline) -> set[str]:
    return {v.ref for v in row.values} | {state.ref for pair in row.statusPairs for state in (pair.priorStatus, pair.postStatus)}


def candidate_coverage(answer: Answers, candidates: dict[str, list[set[str]]]) -> dict:
    """计算进入同一个候选窗口的标准证据组合，区分筛选漏项与模型漏判。"""
    groups = {
        "conflicts": ("task1", [{c.source_id for c in row.claims} for row in answer.task1.conflicts]),
        "timeline_pairs": ("task2", [{a.ref, b.ref} for row in answer.timelines for a, b in zip(row.values, row.values[1:])]),
        "pseudo_changes": ("task2", [{pair.priorStatus.ref, pair.postStatus.ref} for row in answer.timelines for pair in row.statusPairs]),
    }
    result = {}
    for name, (task, gold) in groups.items():
        if task not in candidates:
            continue
        windows = candidates[task]
        missing = [sorted(refs) for refs in gold if not any(refs <= window for window in windows)]
        result[name] = {"covered": len(gold) - len(missing), "expected": len(gold), "missing_sources": missing}
    return result


def source_group(ref: str) -> str:
    """识别官方验证集的文档组；其他材料保留自己的编号。"""
    return next((part for part in ref.split("_") if re.fullmatch(r"M[0-9]+", part)), ref)


def pilot_group_reports(prediction: Answers, reference: Answers, materials: dict[str, dict[str, str]],
                        candidates: dict[str, list[set[str]]] | None) -> dict:
    """逐组核对比较结果，同时显示候选有没有带齐证据。"""
    result: dict[str, dict] = {}
    for task, texts in materials.items():
        groups: dict[str, dict[str, str]] = defaultdict(dict)
        for ref, text in texts.items():
            groups[source_group(ref)][ref] = text
        reports = {}
        for group, sources in sorted(groups.items()):
            subset = (sources, {}) if task == "task1" else ({}, sources)
            actual = prediction.for_sources(*subset)
            expected = reference.for_sources(*subset)
            keys = ("conflicts",) if task == "task1" else ("instance_rows", "instance_fields", "instance_links",
                                                             "timeline_values", "pseudo_changes")
            metrics = scores(actual, expected, record_bound_fields=True)
            report = {"sources": len(sources), "metrics": {key: metrics[key] for key in keys}}
            report["comparison_differences"] = {key: value for key, value in comparison_differences(actual, expected).items()
                                                if key in keys}
            if candidates is not None and task in candidates:
                coverage = candidate_coverage(expected, candidates)
                selected = ("conflicts",) if task == "task1" else ("timeline_pairs", "pseudo_changes")
                report["candidate_coverage"] = {key: coverage[key] for key in selected}
            reports[group] = report
        result[task] = reports
    result["cross_group_findings"] = {
        label: {
            "conflicts": sum(len({source_group(c.source_id) for c in row.claims}) > 1 for row in answer.task1.conflicts),
            "timelines": sum(len({source_group(ref) for ref in timeline_refs(row)}) > 1 for row in answer.timelines),
        }
        for label, answer in (("prediction", prediction), ("reference", reference))
    }
    return result


def split_group_reports(prediction: Answers, reference: Answers, materials: dict[str, dict[str, str]],
                        pilot_ids: dict[str, set[str]], candidates: dict[str, list[set[str]]] | None) -> dict:
    """仅在本地对照正式试跑用过的组和其余组；共享实例可以在两边出现。"""
    result = {}
    groups = {task: {source_group(ref) for ref in ids} for task, ids in pilot_ids.items()}
    partitions = {task: {ref for ref in texts if source_group(ref) in groups[task]}
                  for task, texts in materials.items()}
    if any(ids != partitions[task] for task, ids in pilot_ids.items()):
        raise ValueError("试跑清单必须是完整验证集中的完整文档组")
    for name, keep_pilot in (("pilot_groups", True), ("held_out_groups", False)):
        subsets = {task: {ref: text for ref, text in texts.items() if (ref in partitions[task]) == keep_pilot}
                   for task, texts in materials.items()}
        actual = prediction.for_sources(subsets["task1"], subsets["task2"])
        expected = reference.for_sources(subsets["task1"], subsets["task2"])
        metrics = scores(actual, expected, record_bound_fields=True)
        report = {"sources": {task: len(rows) for task, rows in subsets.items()},
                  "metrics": {key: value for key, value in metrics.items()
                              if key not in {"object_types", "properties", "link_types"}},
                  "comparison_evidence": comparison_evidence_scores(actual, expected),
                  "comparison_differences": comparison_differences(actual, expected)}
        if candidates is not None:
            report["candidate_coverage"] = candidate_coverage(expected, candidates)
        result[name] = report

    def crossing_counts(answer: Answers) -> dict:
        def is_mixed(refs: set[str], task: str) -> bool:
            return bool(refs & partitions[task] and refs - partitions[task])
        return {
            "conflicts": sum(is_mixed({c.source_id for c in row.claims}, "task1") for row in answer.task1.conflicts),
            "timelines": sum(is_mixed(timeline_refs(row), "task2") for row in answer.timelines),
        }

    result["cross_split_findings"] = {"prediction": crossing_counts(prediction), "reference": crossing_counts(reference)}
    return result


def text_key(value: object) -> str:
    if isinstance(value, str):
        return " ".join(normalize("NFKC", value).split())
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


TYPE_ROLES = {
    "患者": "patient", "就诊": "visit", "就诊记录": "visit",
    "检查": "project", "检查项目": "project", "检查记录": "exam",
}

FIELD_TITLES = (
    ("证件号码", "患者证件号码", "证件号", "身份证号", "身份证号码"),
    ("就诊编号", "就诊号"),
    ("检查项目名称", "检查项目名", "项目名称", "检查名称"),
    ("报告日期", "报告时间", "检查报告日期", "检查报告时间"),
)


def type_role(name: str) -> str:
    return TYPE_ROLES.get(text_key(name), "")


def field_value(obj: ObjectType, row: Instance, titles: set[str]) -> Scalar | None:
    values = [row.data[k] for k, p in obj.properties.items() if text_key(p.title) in titles and k in row.data]
    return values[0] if len(values) == 1 and values[0] != "" else None


def appears_in_text(value: object, texts: str) -> bool:
    """英文或数字编号可紧贴中文，但不能命中另一个更长的编号。"""
    return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(str(value)) + r"(?![A-Za-z0-9])", texts))


def instance_index(rows: list[Instance]) -> dict[tuple, int]:
    """按字段真值查找可唯一引用的记录。"""
    matches: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        for field, value in row.data.items():
            matches[row.typeCode, field, type(value), value].append(i)
    return {key: hits[0] for key, hits in matches.items() if len(hits) == 1}


def targets(row: Instance, index: dict[tuple, int]) -> dict[str, int]:
    return {field: index[ref] for field, link in row.relations.items()
            if (ref := (link.targetTypeCode, link.targetProperty, type(link.targetPropertyVal), link.targetPropertyVal)) in index}


def business_identities(answer: Answers) -> dict[tuple, list[int]]:
    """从官方样例能回查的身份字段定位记录，保留同名或同日的歧义。"""
    objects = answer.task1.model.object_index()
    rows = answer.instances
    index = instance_index(rows)
    identities: dict[tuple, list[int]] = defaultdict(list)

    def value(i: int, title_group: int):
        row = rows[i]
        return field_value(objects[row.typeCode], row, set(FIELD_TITLES[title_group]))

    for i, row in enumerate(rows):
        role = type_role(row.typeName)
        if role in {"patient", "visit", "project"}:
            key = (value(i, {"patient": 0, "visit": 1, "project": 2}[role]),)
        elif role == "exam":
            linked = list(targets(row, index).values())
            visits = [value(j, 1) for j in linked if type_role(rows[j].typeName) == "visit"]
            projects = [value(j, 2) for j in linked if type_role(rows[j].typeName) == "project"]
            if len(visits) != 1 or len(projects) != 1:
                continue
            reported_at = value(i, 3)
            try:
                # 对齐用同一日期写法；实际评分仍逐字段核对导出的值。
                reported_at = instance_report_date(reported_at) if reported_at is not None else None
            except ValueError:
                continue
            key = (visits[0], projects[0], reported_at)
        else:
            continue
        if None not in key:
            identities[(role, *map(text_key, key))].append(i)
    return identities


def align_generated_ids(prediction: Answers, reference: Answers, task2_texts: dict[str, str]) -> tuple[Answers, dict]:
    """仅替换能双向唯一对齐的合成编号，并同步更新引用。"""
    actual, expected = business_identities(prediction), business_identities(reference)
    left_types, right_types = prediction.task1.model.object_index(), reference.task1.model.object_index()
    occupied = {(row.typeCode, type(value), value) for row in prediction.instances
                for value in (row.data[left_types[row.typeCode].primaryKey],)}
    material = "\n".join(task2_texts.values())
    replacements = {}
    counts: Counter[str] = Counter()
    for key, left in actual.items():
        right = expected.get(key, [])
        if not right:
            continue
        if len(left) != 1 or len(right) != 1:
            counts["ambiguous_identities"] += 1
            continue
        i, j = left[0], right[0]
        a, b = prediction.instances[i], reference.instances[j]
        a_pk, b_pk = left_types[a.typeCode].primaryKey, right_types[b.typeCode].primaryKey
        old, new = a.data[a_pk], b.data[b_pk]
        if old == new or type(old) is not str or type(new) is not str or a_pk in a.relations:
            continue
        if (any(appears_in_text(v, material) for v in (old, new))
                or (a.typeCode, type(new), new) in occupied):
            continue
        replacements[a.typeCode, a_pk, type(old), old] = new
        occupied.add((a.typeCode, type(new), new))
        counts[key[0]] += 1

    rows = []
    for row in prediction.instances:
        obj = left_types[row.typeCode]
        fields = dict(row.data)
        pk = fields[obj.primaryKey]
        fields[obj.primaryKey] = replacements.get((row.typeCode, obj.primaryKey, type(pk), pk), pk)
        relations = dict(row.relations)
        for field, link in relations.items():
            old = link.targetPropertyVal
            ref = (link.targetTypeCode, link.targetProperty, type(old), old)
            if ref in replacements:
                fields[field] = replacements[ref]
                relations[field] = link.model_copy(update={"targetPropertyVal": replacements[ref]})
        rows.append(row.model_copy(update={"data": fields, "relations": relations}))
    validate_instances(rows, prediction.task1.model)
    return Answers(prediction.task1, rows, prediction.timelines), dict(counts)


def instance_differences(prediction: Answers, reference: Answers) -> dict:
    """在双向唯一对应的记录上汇总字段差异；报告只写字段标题和计数。"""
    actual = business_identities(prediction)
    expected = business_identities(reference)
    left_types = prediction.task1.model.object_index()
    right_types = reference.task1.model.object_index()
    counts: dict[str, Counter] = defaultdict(Counter)
    fields: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    def values(row: Instance, obj: ObjectType) -> dict[str, Counter]:
        result: dict[str, Counter] = defaultdict(Counter)
        for field, value in row.data.items():
            if field not in {obj.primaryKey, *obj.foreignKeys}:
                result[text_key(obj.properties[field].title)][text_key(value)] += 1
        return result

    for key in sorted(actual.keys() | expected.keys()):
        role = key[0]
        a, b = actual.get(key, []), expected.get(key, [])
        if len(a) > 1 or len(b) > 1:
            status = "ambiguous"
        elif not a:
            status = "reference_only"
        elif not b:
            status = "prediction_only"
        else:
            status = "paired"
            left, right = prediction.instances[a[0]], reference.instances[b[0]]
            x = values(left, left_types[left.typeCode])
            y = values(right, right_types[right.typeCode])
            for title in sorted(x.keys() | y.keys()):
                field_status = ("missing" if title not in x else "extra" if title not in y else
                                "matched" if x[title] == y[title] else "different_value")
                fields[role][title][field_status] += 1
        counts[role][status] += 1

    return {"by_role": {role: {"identities": dict(counts[role]),
                                "fields": {title: dict(stats) for title, stats in fields[role].items()}}
                        for role in sorted(counts)},
            "without_identity": {"prediction": len(prediction.instances) - sum(map(len, actual.values())),
                                 "reference": len(reference.instances) - sum(map(len, expected.values()))}}


def item_keys(answer: Answers, *, record_bound_fields: bool = False) -> dict[str, list[tuple]]:
    """只比较可复现的规范化结果；编号和值偏差可在分项中单独定位。"""
    objects = answer.task1.model.object_index()
    names = {code: text_key(o.name) for code, o in objects.items()}

    def field_name(kind: str, field: str) -> str:
        prop = objects[kind].properties.get(field)
        return text_key(prop.title) if prop else field

    def timeline_field(code: str) -> str:
        titles = {field_name(kind, code) for kind, obj in objects.items() if code in obj.properties}
        return next(iter(titles)) if len(titles) == 1 else code

    def instance_fields(row: Instance) -> Iterable[tuple]:
        key = (names[row.typeCode],)
        if record_bound_fields:
            key += (text_key(row.data[objects[row.typeCode].primaryKey]),)
        return (key + (field_name(row.typeCode, field), text_key(value)) for field, value in row.data.items())

    return {
        "object_types": [(name,) for name in names.values()],
        "properties": [(names[code], field_name(code, field), p.type) for code, obj in objects.items() for field, p in obj.properties.items()],
        "link_types": [
            (names[source], field_name(source, sp), names[target], field_name(target, tp),
             "MANY_TO_ONE" if link.cardinality == "ONE_TO_MANY" else link.cardinality)
            for link in answer.task1.model.link_types for source, sp, target, tp in sorted(instance_joins(link, objects))
        ],
        "conflicts": [
            (names[c.owner_type], field_name(c.owner_type, c.property), c.attribute_status,
             tuple(sorted((claim.source_id, text_key(claim.value)) for claim in c.claims)))
            for c in answer.task1.conflicts
        ],
        "instance_rows": [
            (names[i.typeCode], tuple(sorted((field_name(i.typeCode, f), text_key(v)) for f, v in i.data.items())))
            for i in answer.instances
        ],
        "instance_fields": [key for row in answer.instances for key in instance_fields(row)],
        "instance_links": [
            (names[i.typeCode], field_name(i.typeCode, f), names[link.targetTypeCode],
             field_name(link.targetTypeCode, link.targetProperty), text_key(link.targetPropertyVal),
             text_key(i.data[objects[i.typeCode].primaryKey]))
            for i in answer.instances for f, link in i.relations.items()
        ],
        "timeline_values": [
            (t.zjhm, timeline_field(t.changeItem), timeline_field(t.relatedItem), text_key(t.relatedItemVal),
             v.ref, v.time, text_key(v.value))
            for t in answer.timelines for v in t.values
        ],
        "pseudo_changes": [
            (t.zjhm, timeline_field(t.changeItem), timeline_field(t.relatedItem), text_key(t.relatedItemVal),
             p.priorStatus.ref, p.priorStatus.time, text_key(p.priorStatus.value),
             p.postStatus.ref, p.postStatus.time, text_key(p.postStatus.value))
            for t in answer.timelines for p in t.statusPairs
        ],
    }


def scores(prediction: Answers, reference: Answers, *, record_bound_fields: bool = False) -> dict[str, dict]:
    actual = item_keys(prediction, record_bound_fields=record_bound_fields)
    expected = item_keys(reference, record_bound_fields=record_bound_fields)
    result = {}
    for category, keys in expected.items():
        a, b = Counter(actual[category]), Counter(keys)
        hit = (a & b).total()
        predicted, gold = a.total(), b.total()
        result[category] = {
            "matched": hit, "predicted": predicted, "expected": gold,
            "extra": predicted - hit, "missing": gold - hit,
            "precision": hit / predicted if predicted else None,
            "recall": hit / gold if gold else None,
            "f1": 2 * hit / (predicted + gold) if predicted + gold else None,
        }
    return result


def catalog_extension_candidates(prediction: Answers, reference: Answers) -> dict:
    """列出参考标注未覆盖的目录对象、字段和关联，便于回查证据。"""
    output_types, output_fields = [], []
    instance_counts = Counter(row.typeCode for row in prediction.instances)
    for term, fields in CATALOG:
        reference_types = [obj for obj in reference.task1.model.object_types if term.matches(obj.name)]
        reference_titles = {prop.title for obj in reference_types for prop in obj.properties.values()}
        new_fields = {field for field in fields if not any(field.matches(title) for title in reference_titles)}
        for obj in prediction.task1.model.object_types:
            if not term.matches(obj.name):
                continue
            if not reference_types:
                output_types.append({"type_code": obj.typeCode, "name": obj.name,
                                     "instances": instance_counts[obj.typeCode]})
            for code, prop in obj.properties.items():
                if any(field.matches(prop.title) for field in new_fields):
                    output_fields.append({"type_code": obj.typeCode, "field_code": code, "title": prop.title})
    extended_types = {item["type_code"] for item in output_types}
    extended_fields = {(item["type_code"], item["field_code"]) for item in output_fields}
    instance_links = Counter(
        (row.typeCode, field, link.targetTypeCode, link.targetProperty)
        for row in prediction.instances for field, link in row.relations.items()
    )
    schema = prediction.task1.model
    objects = schema.object_index()
    output_links = []
    for link in schema.link_types:
        for source, sp, target, tp in sorted(instance_joins(link, objects)):
            if ({source, target} & extended_types) or ({(source, sp), (target, tp)} & extended_fields):
                output_links.append({"name": link.name, "source": f"{source}.{sp}", "target": f"{target}.{tp}",
                                     "instance_links": instance_links[source, sp, target, tp]})
    return {"object_types": output_types, "properties": output_fields, "link_types": output_links}


def comparison_evidence_scores(prediction: Answers, reference: Answers) -> dict:
    """分开核对来源、日期和取值；保留原始时间精度的匹配。"""
    def day(value: str) -> str:
        # 年和月仍保持原来的精度；只去掉明确日期之后的时分秒。
        return value[:10]

    def evidence(answer: Answers) -> dict:
        items = item_keys(answer)
        return {
            "conflict_source_sets": [tuple(sorted({ref for ref, _ in key[3]})) for key in items["conflicts"]],
            "timeline_evidence": [(key[0], *key[4:]) for key in items["timeline_values"]],
            "timeline_source_days": [(k[0], k[4], day(k[5])) for k in items["timeline_values"]],
            "timeline_evidence_by_day": [(k[0], k[4], day(k[5]), k[6]) for k in items["timeline_values"]],
            "pseudo_change_evidence": [(key[0], *key[4:]) for key in items["pseudo_changes"]],
            "pseudo_change_source_pairs": [(k[0], k[4], day(k[5]), k[7], day(k[8]))
                                           for k in items["pseudo_changes"]],
        }

    actual, expected = evidence(prediction), evidence(reference)
    return {name: {"matched": (Counter(actual[name]) & Counter(keys)).total(),
                   "predicted": len(actual[name]), "expected": len(keys)}
            for name, keys in expected.items()}


def comparison_differences(prediction: Answers, reference: Answers) -> dict:
    """按精确匹配定位错项；报告只携带来源编号和日期。"""
    actual, expected = item_keys(prediction), item_keys(reference)
    result = {}
    for name in ("conflicts", "timeline_values", "pseudo_changes"):
        left, right = Counter(actual[name]), Counter(expected[name])

        def locate(key: tuple) -> dict:
            if name == "conflicts":
                return {"sources": sorted({ref for ref, _ in key[3]}), "status": key[2]}
            if name == "timeline_values":
                return {"sources": [key[4]], "time": key[5]}
            return {"sources": [key[4], key[7]], "times": [key[5], key[8]]}

        result[name] = {"missing": [locate(key) for key in (right - left).elements()],
                        "extra": [locate(key) for key in (left - right).elements()]}
    return result
