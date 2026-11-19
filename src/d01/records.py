"""逐条抽取有证据的记录，保留原始事实与独立审核结果。"""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Annotated, Literal
from unicodedata import normalize

from pydantic import Field

from .contracts import Contract, Scalar, Schema, Text, instance_report_date, scalar_matches
from .discovery import Discovery
from .inputs import Sources
from .model import ModelClient, OutputError
from .stages import Evidence, FieldGap, Review, batches, dump, parallel_in_order


class Locator(Contract):
    type_code: Text
    label: Text
    scope: Literal["shared", "patient", "event"]
    identity: Text
    refs: Annotated[list[Text], Field(min_length=1)]


class Locators(Contract):
    records: list[Locator]


class Fact(Contract):
    field: str
    raw_value: Text
    value: Scalar
    evidence: Annotated[list[Evidence], Field(min_length=1)]
    raw_time: str = ""
    note: str = ""


class Connection(Contract):
    field: Text
    target_slot: Text
    target_field: Text
    evidence: Annotated[list[Evidence], Field(min_length=1)]


class RecordDraft(Contract):
    facts: list[Fact]
    connections: list[Connection]
    proposed_fields: list[Fact]


class LinkChoice(Contract):
    option: Text
    evidence: Annotated[list[Evidence], Field(min_length=1)]


class RecordSelection(Contract):
    facts: list[Fact]
    connections: list[LinkChoice]
    proposed_fields: list[Fact]


@dataclass(frozen=True)
class Record:
    slot: str
    locator: Locator
    draft: RecordDraft

    def snapshot(self) -> dict:
        return {"slot": self.slot, "locator": self.locator.model_dump(), "facts": self.draft.model_dump()}


@dataclass
class Extraction:
    records: list[Record]
    reviews: list[dict]
    missing_schema: list[FieldGap]


def stable_code(kind: str, *parts: str) -> str:
    """从业务身份生成稳定编号，重复运行时同一组材料得到相同编号。"""
    return kind.upper() + "_" + sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:16].upper()


def identity_key(value: str) -> str:
    return " ".join(normalize("NFKC", value).strip().casefold().split())


def _shape(schema: Schema) -> dict:
    return schema.model_dump(mode="json", exclude_none=True)


def _locators(sources: Sources, refs: list[str], schema: Schema, client: ModelClient, group: str) -> list[Locator]:
    objects = schema.object_index()
    feedback = ""
    for attempt in range(2):
        try:
            found = client.ask(f"{sources.task}.locate.{group}.{attempt}",
                "逐条阅读材料，识别所有应当在给定结构下抽取的实例，只列记录定位。"
                "同一次就诊不同类型的记录分别列；一段有多个项目要全找；复查按每次发生时间另列记录。"
                "重复出现的同一条记录用一个定位和多个 refs。目录如检查项目、医疗机构或药品 scope=shared，患者主档 scope=patient，其他具体记录 scope=event。"
                "identity 优先用原文业务编号；患者没有编号时用证件号；共享目录没有编号时用原文中稳定名称（如有规格差异带规格）；"
                "事件无编号时用同一就诊、项目和时间区分。label 写可从这些材料直接定位的短名称。"
                "只用已给的 typeCode 和原始材料 id，不能生成不存在的记录。",
                {"结构": _shape(schema), "材料": sources.selected(refs), "修订": feedback}, Locators, max_tokens=3800)
            if all(item.type_code in objects and set(item.refs) <= set(refs) for item in found.records):
                return found.records
            feedback = "记录类型或来源编号无效，需使用结构和本组材料中列出的值。"
        except OutputError as exc:
            feedback = str(exc)
    raise OutputError(f"{sources.task} 定位组 {group}：{feedback}")


def connection_options(slot: str, slots: dict[str, Locator], schema: Schema) -> dict[str, tuple[str, str, str]]:
    """列出当前对象能引用的目标；选项包含完整的两端字段。"""
    destinations = defaultdict(list)
    for owner, field, target_kind, target_field in sorted(schema.joins()):
        if owner == slots[slot].type_code:
            destinations[target_kind].append((field, target_field))
    allowed = [(field, key, target_field) for key, target in slots.items() if key != slot
               for field, target_field in destinations.get(target.type_code, [])]
    return {str(i): result for i, result in enumerate(allowed, 1)}


def resolve_connections(selection: RecordSelection, options: dict[str, tuple[str, str, str]]) -> RecordDraft:
    """把模型选的序号还原为已定义的字段和记录定位。"""
    connections = []
    for choice in selection.connections:
        if choice.option not in options:
            raise OutputError("关联选项不在本轮清单中")
        field, slot, target_field = options[choice.option]
        connections.append(Connection(field=field, target_slot=slot, target_field=target_field, evidence=choice.evidence))
    return RecordDraft(facts=selection.facts, connections=connections, proposed_fields=selection.proposed_fields)


def _record_issues(draft: RecordDraft, schema: Schema, locator: Locator, sources: Sources, refs: set[str]) -> list[str]:
    obj = schema.object_index()[locator.type_code]
    issues = []
    for n, fact in enumerate([*draft.facts, *draft.proposed_fields], 1):
        if any(not item.check(sources, refs) for item in fact.evidence):
            issues.append(f"第 {n} 条事实的引文或来源无效")
        if fact.raw_time and all(fact.raw_time not in sources.texts[e.ref] for e in fact.evidence if e.ref in refs):
            issues.append(f"第 {n} 条事实的原始时间应来自原文")
    for fact in draft.facts:
        p = obj.properties.get(fact.field)
        if fact.field and (p is None or not scalar_matches(fact.value, p.type)):
            issues.append(f"字段 {fact.field} 未定义或取值类型不符合结构")
        if p is not None and p.title == "报告日期":
            try:
                instance_report_date(fact.value)
            except ValueError:
                issues.append(f"字段 {fact.field} 不是有效的报告日期")
    for conn in draft.connections:
        if any(not piece.check(sources, refs) for piece in conn.evidence):
            issues.append(f"连接字段 {conn.field} 缺少可回查的引文")
    return issues


def _record(sources: Sources, refs: list[str], slots: dict[str, Locator], slot: str, schema: Schema,
            client: ModelClient, *, review_enabled: bool = True) -> tuple[Record | None, dict]:
    locator = slots[slot]
    obj = schema.object_index()[locator.type_code]
    options = connection_options(slot, slots, schema)
    context = {"任务": sources.task, "当前对象结构": obj.model_dump(), "当前定位": locator.model_dump(),
               "允许关联的选项": {key: {"外键": field, "目标字段": target_field, "目标记录": slots[target].model_dump()}
                                 for key, (field, target, target_field) in options.items()},
               "材料": sources.selected(refs)}
    feedback = []
    history = []
    previous = None
    for attempt in range(2):
        try:
            selection = client.ask(f"{sources.task}.record.{slot}.{attempt}",
                "只抽取当前定位的这一条记录。逐字段输出，证据 quote 要逐字复制，不要带原文没有的引号。"
                "facts 中的字段编码从当前对象的 properties 选择；其他对象的字段由各自记录提取。"
                "raw_value 是原始短语，value 按字段类型做必要规范化。不同说法分别作为事实保存。"
                "证件号、姓名和编号照抄，缺失编号由程序分配。报告日期按报告原文能支持的最细时间精度填写，保留已写明的时分秒。"
                "只有日期时保留日期；raw_time 逐字填写原文表达。初诊/复诊标志只取确切支持的就诊性质，医疗机构名使用原文全称。"
                "检查结果取本次报告的结果值，不加报告引导语、归档说明或复查比较。"
                "对相较初诊的描述、疑似矛盾或无法归字段的事实，用 field 空串并保留 note；"
                "有充分证据且明显属于本体范围的漏项才用 proposed_fields，字段名写中文。"
                "connections 的 option 填允许关联的选项的字符串编号。可用同一就诊编号等两段原文联合支持。"
                "如果没有合适选项但外键值出现在原文，把外键编码和原文取值填进 facts，供程序跨组查询。"
                "任何未找到的事实或关联都直接省略；没有条目时返回空列表。",
                {**context, "上一次抽取": previous, "复核反馈": feedback}, RecordSelection, max_tokens=3100)
            previous = selection.model_dump()
            draft = resolve_connections(selection, options)
            feedback = _record_issues(draft, schema, locator, sources, set(refs))
            if feedback:
                history.append({"format_issues": feedback})
                continue
            if not review_enabled:
                return Record(slot, locator, draft), {"slot": slot, "history": history, "accepted": True}
            review = client.ask(f"{sources.task}.record_review.{slot}.{attempt}",
                "独立复核这一条记录是否抓住原文所有核心字段，是否有跨患者、跨事件、错时间、误归属或漏掉证据。"
                "按选项表核对 connections 的归属。证据原文、术语和归属准确时 accept，分数至少 90，issues 留空。"
                "缺少原文未提供的业务编号不需要拒绝。任务一记录保留各方冲突说法。反馈只写字段、source ID 和错误类型。",
                {**context, "待审核": previous}, Review, max_tokens=1000)
            history.append({"review": review.model_dump()})
            if review.passed:
                return Record(slot, locator, draft), {"slot": slot, "history": history, "accepted": True}
            feedback = review.issues or ["重新核对记录的关键字段和归属"]
        except OutputError as exc:
            feedback = [str(exc)]
            history.append({"format_issues": feedback})
    return None, {"slot": slot, "type_code": locator.type_code, "sources": locator.refs,
                  "history": history, "accepted": False}


def extract(sources: Sources, discovery: Discovery, schema: Schema, client: ModelClient,
            workers: int, *, group_mode: str = "discovered", review_enabled: bool = True) -> Extraction:
    """先定位每组记录，再独立抽取和复核，线程共享同一个请求限速器。"""
    if group_mode not in {"discovered", "single"}:
        raise ValueError("未知记录分组方式")
    groups = discovery.groups if group_mode == "discovered" else [[ref] for ref in discovery.covered]
    jobs = []
    for index, refs in enumerate(groups):
        for part, rows in enumerate(batches(sources.selected(refs), client.config.context_chars // 2)):
            jobs.append((f"g{index+1}_{part+1}", [item["id"] for item in rows]))

    def locate(job: tuple[str, list[str]]) -> tuple[list[str], dict[str, Locator]]:
        group, refs = job
        found = _locators(sources, refs, schema, client, group)
        return refs, {f"{group}_r{i}": item for i, item in enumerate(found, 1)}

    records = []
    reviews = []
    proposed: dict[tuple[str, str], dict[str, Evidence]] = defaultdict(dict)
    located = parallel_in_order(locate, jobs, workers)
    work = ((refs, slots, slot) for refs, slots in located for slot in slots)

    def pull(job: tuple[list[str], dict[str, Locator], str]) -> tuple[Record | None, dict]:
        refs, slots, slot = job
        return _record(sources, refs, slots, slot, schema, client, review_enabled=review_enabled)

    for record, review in parallel_in_order(pull, work, workers):
        reviews.append(review)
        if record is not None:
            records.append(record)
            for item in record.draft.proposed_fields:
                if not item.field.strip():
                    continue
                evidence = proposed[record.locator.type_code, item.field.strip()]
                for piece in item.evidence:
                    # 截取连续引句，每份材料最多保留一个范例，审核时无需发送整份报告。
                    evidence.setdefault(piece.ref, Evidence(ref=piece.ref, quote=piece.quote.strip()[:400]))
    missing = [FieldGap(type=kind, field=field, sources=sorted(evidence),
                        evidence=[evidence[ref] for ref in sorted(evidence)[:3]])
               for (kind, field), evidence in sorted(proposed.items()) if len(evidence) >= 2]
    return Extraction(records, reviews, missing)
