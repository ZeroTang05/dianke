"""按业务身份连接记录，给冲突和时序保留可回查的候选。"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from .contracts import Contract, Instance, InstanceLink, Scalar, Schema, instance_report_date, scalar_matches
from .discovery import Discovery
from .inputs import Sources
from .model import ModelClient
from .records import Fact, Record, identity_key, stable_code
from .stages import dump


Node = tuple[str, Scalar]


class SelectedValue(Contract):
    index: int


def record_issue(records: list[Record], reason: str, *, field: str | None = None) -> dict:
    """用记录定位号和材料编号定位问题，清单里不保存患者编号或取值。"""
    issue = {"type_code": records[0].locator.type_code, "slots": [row.slot for row in records],
             "sources": sorted({ref for row in records for ref in row.locator.refs}), "reason": reason}
    if field is not None:
        issue["field"] = field
    return issue


@dataclass(frozen=True)
class Candidate:
    key: str
    refs: list[str]
    records: list[Record]
    patients: list[Node]
    tracks: tuple[tuple[str, ...], ...] = ()

    def payload(self, sources: Sources) -> dict:
        return {"材料": sources.selected(self.refs), "已审核记录": [record.snapshot() for record in self.records]}


class Graph:
    """记录节点可共享目录；患者归属沿外键计算，经过目录时停止。"""

    def __init__(self, records: list[Record], schema: Schema):
        self.records = records
        self.schema = schema
        objects = schema.object_index()
        self.keys: dict[str, Node] = {}
        self.nodes: dict[Node, list[Record]] = defaultdict(list)
        self.links: dict[Node, dict[str, set[tuple[Node, str]]]] = defaultdict(lambda: defaultdict(set))
        self.refs: dict[str, set[Node]] = defaultdict(set)
        self.unresolved: list[dict] = []
        self.unresolved_values: list[dict] = []
        # 先取原文中可见的编号，再关联同一目录项或患者的无编号记录。
        explicit: dict[str, Scalar] = {}
        identities: dict[tuple[str, str], set[Scalar]] = defaultdict(set)
        verified_identities: set[str] = set()
        for record in records:
            loc = record.locator
            pk = objects[loc.type_code].primaryKey
            values = {f.value for f in record.draft.facts if f.field == pk
                      and any(str(f.value) in e.quote for e in f.evidence)}
            if len(values) == 1:
                explicit[record.slot] = next(iter(values))
            # 姓名只用来展示。患者跨组归并需要原文编号或证件号。
            if loc.scope == "shared" or (loc.scope == "patient" and (
                record.slot in explicit or any(
                    f.field in objects[loc.type_code].properties
                    and any(word in objects[loc.type_code].properties[f.field].title for word in ("证件", "身份证", "护照"))
                    and identity_key(str(f.value)) == identity_key(loc.identity)
                    and any(str(f.value) in e.quote for e in f.evidence)
                    for f in record.draft.facts
                )
            )):
                verified_identities.add(record.slot)
                if values:
                    identities[loc.type_code, identity_key(loc.identity)].update(values)
        for record in records:
            loc = record.locator
            obj = objects[loc.type_code]
            known = identities[loc.type_code, identity_key(loc.identity)]
            value = explicit.get(record.slot)
            if value is None and record.slot in verified_identities and len(known) == 1:
                value = next(iter(known))
            if value is None:
                if obj.properties[obj.primaryKey].type != "string":
                    self.unresolved.append(record_issue([record], "缺少非字符串主键的原文证据"))
                    continue
                basis = (loc.scope, identity_key(loc.identity))
                if record.slot not in verified_identities or len(known) > 1:
                    basis = (*basis, *sorted(loc.refs), identity_key(loc.label))
                value = stable_code(loc.type_code, *basis)
            key = (loc.type_code, value)
            self.keys[record.slot] = key
            self.nodes[key].append(record)
            for ref in loc.refs:
                self.refs[ref].add(key)
            for fact in record.draft.facts:
                for item in fact.evidence:
                    self.refs[item.ref].add(key)
        missing_targets: set[tuple[Node, str]] = set()
        for record in records:
            if record.slot not in self.keys:
                continue
            owner = self.keys[record.slot]
            for link in record.draft.connections:
                target = self.keys.get(link.target_slot)
                if target is None:
                    missing_targets.add((owner, link.field))
                    continue
                self.links[owner][link.field].add((target, link.target_field))
        self._join_exact_values(missing_targets)
        for node, fields in self.links.items():
            for field, endpoints in fields.items():
                if len(endpoints) > 1:
                    self.unresolved.append(record_issue(self.nodes[node], "同一外键存在多个目标", field=field))

    def _join_exact_values(self, missing_targets: set[tuple[Node, str]]) -> None:
        """按模型已标注的外键值跨组连接，只对同一目标唯一命中时补边。"""
        joins = self.schema.joins()
        targets: dict[tuple, set[Node]] = defaultdict(set)
        join_fields = {(kind, field) for _, _, kind, field in joins}
        objects = self.schema.object_index()
        for node, records in self.nodes.items():
            kind, pk = node
            if (kind, objects[kind].primaryKey) in join_fields:
                targets[kind, objects[kind].primaryKey, type(pk), pk].add(node)
            for record in records:
                for fact in record.draft.facts:
                    if (kind, fact.field) in join_fields and any(str(fact.value) in e.quote for e in fact.evidence):
                        targets[kind, fact.field, type(fact.value), fact.value].add(node)
        destinations: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for kind, field, target_kind, target_field in sorted(joins):
            destinations[kind, field].append((target_kind, target_field))
        for node, records in self.nodes.items():
            possible: dict[str, set[tuple[Node, str]]] = defaultdict(set)
            for row in records:
                for fact in row.draft.facts:
                    target_fields = destinations.get((node[0], fact.field))
                    if not target_fields or self.links[node].get(fact.field) or not any(str(fact.value) in e.quote for e in fact.evidence):
                        continue
                    endpoints = possible[fact.field]
                    for target_kind, target_field in target_fields:
                        for match in targets[target_kind, target_field, type(fact.value), fact.value]:
                            if match != node:
                                endpoints.add((match, target_field))
            for field, endpoints in possible.items():
                if len(endpoints) == 1:
                    self.links[node][field].update(endpoints)
                else:
                    reason = "外键原文值对应多个目标" if endpoints else "外键原文值找不到目标"
                    self.unresolved.append(record_issue(records, reason, field=field))
            for field in objects[node[0]].foreignKeys:
                if (node, field) in missing_targets and field not in possible and not self.links[node].get(field):
                    self.unresolved.append(record_issue(records, "关联目标未验收", field=field))

    def scope(self, node: Node) -> str:
        return self.nodes[node][0].locator.scope

    def ancestors(self, start: Node) -> set[Node]:
        visited: set[Node] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited or self.scope(node) == "shared":
                continue
            visited.add(node)
            stack.extend(target for links in self.links[node].values() for target, _ in links)
        return visited

    def evidence_tracks(self, records: list[Record], refs: list[str]) -> tuple[tuple[str, ...], ...]:
        """关联同一对象的事实，以及引用同一目录的同类事件。"""
        tracks: dict[tuple, set[str]] = defaultdict(set)
        order = {ref: i for i, ref in enumerate(refs)}
        for record in records:
            node = self.keys[record.slot]
            if self.scope(node) == "shared":
                continue
            projects = {target for endpoints in self.links.get(node, {}).values() for target, _ in endpoints
                        if self.scope(target) == "shared"}
            for fact in record.draft.facts:
                evidence = {item.ref for item in fact.evidence if item.ref in order}
                tracks[("node", node, fact.field)].update(evidence)
                if fact.field:
                    for target in sorted(projects, key=repr):
                        tracks[("project", node[0], target, fact.field)].update(evidence)
        return tuple(dict.fromkeys(tuple(sorted(track, key=order.__getitem__))
                                   for track in tracks.values() if len(track) >= 2))

    def candidates(self, sources: Sources, discovery: Discovery, mode: Literal["indexed", "grouped"]) -> list[Candidate]:
        """患者连接优先；原始文本分组仍保留，可覆盖证据未能归字段的冲突。"""
        containers: dict[tuple, set[str]] = defaultdict(set)
        patient_for: dict[str, set[Node]] = defaultdict(set)
        for ref, nodes in self.refs.items():
            for node in nodes:
                patient_for[ref].update(key for key in self.ancestors(node) if self.scope(key) == "patient")
        for index, group in enumerate(discovery.groups):
            group_patients = set().union(*(patient_for[ref] for ref in group))
            for ref in group:
                patients = patient_for[ref]
                if mode == "indexed" and len(group_patients) == 1 and (not patients or patients == group_patients):
                    containers[("patient", *next(iter(group_patients)))].add(ref)
                elif mode == "indexed" and len(patients) == 1:
                    containers[("patient", *next(iter(patients)))].add(ref)
                else:
                    containers[("group", index)].add(ref)
        priority = set(discovery.findings.priority_refs)
        result = []
        order = {ref: i for i, ref in enumerate(sources.texts)}
        record_order = {record.slot: i for i, record in enumerate(self.records)}
        for refs in containers.values():
            if len(refs) < 2:
                continue
            ids = sorted(refs, key=order.__getitem__)
            nodes = set().union(*(self.refs.get(ref, set()) for ref in ids))
            patients = sorted({node for ref in ids for node in patient_for[ref]}, key=repr)
            records = sorted((record for node in nodes for record in self.nodes[node]
                              if refs.intersection(record.locator.refs)), key=lambda r: record_order[r.slot])
            tracks = self.evidence_tracks(records, ids) if mode == "indexed" and len(patients) == 1 else ()
            result.append(Candidate(stable_code("group", *ids), ids, records, patients, tracks))
        result.sort(key=lambda item: not bool(priority.intersection(item.refs)))
        return result

    def instances(self, sources: Sources, client: ModelClient) -> list[Instance]:
        """用审核过的事实生成实例；关联字段从目标记录取值。"""
        objects = self.schema.object_index()
        data: dict[Node, dict[str, Scalar]] = {}
        for index, (node, records) in enumerate(self.nodes.items(), 1):
            kind, pk_value = node
            obj = objects[kind]
            facts: dict[str, list[Fact]] = defaultdict(list)
            for record in records:
                for fact in record.draft.facts:
                    if fact.field and fact.field not in {obj.primaryKey, *obj.foreignKeys}:
                        facts[fact.field].append(fact)
            data[node] = {obj.primaryKey: pk_value}
            for field, items in facts.items():
                values = [instance_report_date(fact.value) if obj.properties[field].title == "报告日期"
                          else fact.value for fact in items]
                if len({(type(value), value) for value in values}) == 1:
                    data[node][field] = values[0]
                    continue
                result = client.ask(f"task2.reconcile.{index}.{field}",
                    "为同一条实例选唯一字段值。对照已经核过原文出处的说法，选适合此对象此字段的值。"
                    "index 从 0 开始；若证据不足以选值，返回 -1，原始候选会保留。",
                    {"对象类型": obj.name, "字段": obj.properties[field].title, "候选": dump(items)}, SelectedValue,
                    max_tokens=256)
                if not -1 <= result.index < len(items):
                    raise ValueError(f"实例 {index} 字段 {field}：模型选择的序号超出候选")
                if result.index >= 0:
                    data[node][field] = values[result.index]
                else:
                    self.unresolved_values.append(record_issue(records, "多份说法不同，证据不足以选择实例取值", field=field))
        items = []
        for node, values in data.items():
            obj = objects[node[0]]
            relations = {}
            for fk, endpoints in self.links[node].items():
                if len(endpoints) != 1:
                    continue
                target, field = next(iter(endpoints))
                value = data[target].get(field)
                if value is None or not scalar_matches(value, obj.properties[fk].type):
                    self.unresolved.append(record_issue(self.nodes[node], "目标连接字段缺失或类型错误", field=fk))
                    continue
                values[fk] = value
                relations[fk] = InstanceLink(targetProperty=field, targetPropertyVal=value, targetTypeCode=target[0])
            items.append(Instance(typeCode=node[0], typeName=obj.name, data=values, relations=relations))
        return items
