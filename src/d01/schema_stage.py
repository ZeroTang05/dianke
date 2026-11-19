"""从任务一原文归纳对象、字段和关系，并在进入实例化前复审。"""

from collections import defaultdict
from itertools import zip_longest
from typing import Annotated

from pydantic import Field

from .contracts import Contract, LinkType, ObjectType, Property, Schema, Text, assert_valid
from .discovery import Discovery
from .inputs import Sources
from .model import ModelClient, OutputError
from .stages import Evidence, FieldGap, Review, batches, dump, json_size
from .vocabulary import canonicalize_schema, catalog_field, catalog_names, existing_field


MEDICAL_SCOPE = {
    "人物与机构": "患者（标识、姓名、证件、性别）；医疗机构有可辨认的身份且可被多次就诊引用时，可以单独建表。医生、科室按材料中的业务角色归属。",
    "就诊与记录": "就诊编号、患者关联、医疗机构、初/复诊；有业务区分时再考虑入院、病历、诊断、处置。",
    "检查与检验": "共享的检查/检验项目（项目编号、名称、类别）与每次执行的检查/检验记录（记录编号、就诊、项目、部位、结果、报告日期）。",
    "用药": "共享药品（编号、名称、剂型、规格、有效期状态）与医嘱/实际用药记录（归属、单次剂量、剂量单位、频次、途径、执行时间或天数）。",
}

# 来自赛题正文的药品段落，避开验证集用来打分的初赛记录组。
EXAMPLE = {
    "材料": "医嘱中的药品是左甲状腺素钠片，剂型片剂、规格50μg/片，当前有效期状态为有效。该药每次使用50μg，单位μg，频次每日两次，途径口服，执行30天。",
    "结果": {
        "object_types": [
            {"name": "药品", "description": "可由多条医嘱引用的药品", "typeCode": "yp",
             "properties": {"ypbh": {"title": "药品编号", "type": "string"}, "ypmc": {"title": "药品名称", "type": "string"},
                            "jx": {"title": "剂型", "type": "string"}, "gg": {"title": "规格", "type": "string"},
                            "yxqzt": {"title": "有效期状态", "type": "string"}},
             "primaryKey": "ypbh", "displayKey": "ypbh", "foreignKeys": []},
            {"name": "医嘱用药", "description": "一次用药安排的剂量、频次、途径及天数", "typeCode": "yzyy",
             "properties": {"yzbh": {"title": "医嘱用药编号", "type": "string"}, "ypbh": {"title": "药品编号", "type": "string"},
                            "dcjl": {"title": "单次剂量", "type": "string"}, "yppl": {"title": "用药频次", "type": "string"},
                            "tj": {"title": "给药途径", "type": "string"}, "zxts": {"title": "执行天数", "type": "number"}},
             "primaryKey": "yzbh", "displayKey": "yzbh", "foreignKeys": ["ypbh"]},
        ],
        "link_types": [{"name": "引用药品", "sourceTypeCode": "yzyy", "targetTypeCode": "yp",
                        "sourceProperty": "ypbh", "targetProperty": "ypbh", "cardinality": "MANY_TO_ONE"}],
    },
}


class Proposal(Contract):
    name: Text
    fields: list[Text]
    connected_to: list[Text]
    evidence: Annotated[list[Evidence], Field(min_length=1, max_length=3)]


class Proposals(Contract):
    items: list[Proposal]


class ProposalProof(Contract):
    name: Text
    evidence: list[Evidence]


class ProposalProofs(Contract):
    items: list[ProposalProof]


class TypeOutline(Contract):
    name: Text
    code: Text
    description: Text
    source_names: list[Text]
    identity: Text


class Outlines(Contract):
    items: Annotated[list[TypeOutline], Field(min_length=1)]


class OutlineReview(Contract):
    additions: list[TypeOutline]
    rationale: list[Text]


class Links(Contract):
    link_types: list[LinkType]


class SchemaDraft(Contract):
    schema_definition: Schema
    rationale: list[Text]


class FieldAddition(Contract):
    type_code: Text
    code: Text
    definition: Property
    evidence: Evidence


class FieldGapReview(Contract):
    reason: Text
    addition: FieldAddition | None


def build_schema(sources: Sources, discovery: Discovery, client: ModelClient, *, use_catalog: bool = True,
                 use_example: bool = True) -> tuple[Schema, dict]:
    """从分批的候选证据归纳类型；逐类型生成字段后再生成关系。"""
    signals = [item.description for item in discovery.findings.signals]
    # 对照时只移除一份提示材料，保留模型任务和初始线索。
    summary = {}
    if use_catalog:
        summary["医疗候选目录"] = MEDICAL_SCOPE
        summary["术语名称与编码"] = catalog_names()
    summary["输入线索"] = signals
    example = {"示例": EXAMPLE} if use_example else {}
    evidence: dict[str, list[Proposal]] = defaultdict(list)
    for index, rows in enumerate(batches(sources.selected(discovery.covered), max(2000, client.config.context_chars // 2)), 1):
        allowed = {row["id"] for row in rows}
        possible = client.ask(f"schema.proposals.{index}",
            "从本批原文发现可建模的对象类型及字段和关系。逐项检查记录是否同时描述一个能被多次引用的项目；"
            "有证据时分别提议项目目录和每次执行的记录。名称、类别或规格可以作为项目存在的证据；缺少项目编号不影响提议。"
            "字段按业务对象归属。仅出现在个别补充说明中的争议信息，先留作比较证据，评估其是否适合成为通用字段。"
            "原文支持时，可提议目录中的医疗机构、药品等扩展类型和字段。示例只供参考格式和划分对象的方法。"
            "归纳同义类型。每种类型给出一到三段紧挨在一起的短原文，优先复制含类型标识的半句话。"
            "逐字复制，保留原文标点，避免省略号拼接、转述或添加句号。",
            {**summary, **example, "材料": rows}, Proposals, max_tokens=4200)
        unverified = {}
        for item in possible.items:
            valid = [piece for piece in item.evidence if piece.check(sources, allowed)]
            if valid:
                evidence[item.name].append(item.model_copy(update={"evidence": valid}))
            else:
                unverified[item.name] = item
        if unverified:
            repaired = client.ask(f"schema.proofs.{index}",
                "为待核对的候选补充原文证据。name 从候选名原样复制。选择能证明该对象存在的一小段连续文字，"
                "同时复制材料的 id 到 ref。原文中找不到依据的候选省略。",
                {"材料": rows, "待核对候选": list(unverified)}, ProposalProofs, max_tokens=1800)
            for proof in repaired.items:
                if proof.name not in unverified:
                    raise OutputError("补引文只能使用待核对的候选名")
                valid = [piece for piece in proof.evidence if piece.check(sources, allowed)][:3]
                if valid:
                    evidence[proof.name].append(unverified[proof.name].model_copy(update={"evidence": valid}))
    if not evidence:
        raise OutputError("找不到有原文证据的候选对象")

    compact = {name: {"字段": sorted({f for row in rows for f in row.fields}),
                      "关联": sorted({f for row in rows for f in row.connected_to}),
                      "例证": [row.evidence[0].quote for row in rows[:2]], "出现批次": len(rows)}
               for name, rows in sorted(evidence.items())}
    outlines = client.ask("schema.types",
        "用所有批次的候选证据决定应建立的对象类型。先逐个核对候选是否是有独立编号、独立信息或承担连接作用的对象。"
        "有编号的就诊记录应单独建模。中文名称使用明确的业务名，区分项目和记录；代码用该名称的拼音首字母。"
        "把同义对象合并；患者、就诊、项目目录和具体记录有不同粒度。同一对象的属性留在该类型中。"
        "依据当前候选及其原文选择对象。目录中的类型有独立的业务含义和原文支持时可以单独建表。"
        "source_names 填被合并的原候选名，identity 说明靠哪个业务标识区分。",
        {**summary, "候选": compact}, Outlines, max_tokens=3000)
    coverage = client.ask("schema.type_coverage",
        "独立检查已选对象是否覆盖所有应建模的业务对象。逐个核对未被任何现有对象纳入的候选；"
        "如候选有独立编号、持有记录或承担连接作用，且原文有依据，补入该对象。"
        "检查当前对象是否混合了可反复引用的项目和一次执行：如果候选中有直接证据，补出独立项目对象。"
        "列出被漏掉的必要对象及其代码和候选名；已经存在的类型和同义项不要重复添加。没有漏掉的对象时 additions 为空。",
        {**summary, "原始候选与例证": compact, "现有对象": dump(outlines.items),
         "未纳入候选": sorted(evidence.keys() - {name for row in outlines.items for name in [row.name, *row.source_names]})},
        OutlineReview, max_tokens=1800)
    existing = {(item.name, item.code) for item in outlines.items}
    for item in coverage.additions:
        if any(item.name == name or item.code == code for name, code in existing):
            continue
        if set(item.source_names).intersection(evidence) or item.name in evidence:
            outlines.items.append(item)
            existing.add((item.name, item.code))

    objects = []
    for index, outline in enumerate(outlines.items, 1):
        names = set(outline.source_names) | {outline.name}
        selected = [row for name in sorted(names) for row in evidence.get(name, [])]
        if not selected:
            raise OutputError(f"表结构候选 {index} 未引用有原文证据的对象")
        payload = {**summary, "当前对象": outline.model_dump(), "所有对象": dump(outlines.items),
                   "本对象证据": dump(selected)}
        if json_size(payload) > client.config.context_chars:
            payload["本对象证据"] = dump(selected[:max(2, client.config.context_chars // 1400)])
            payload["本对象全部候选字段"] = sorted({f for row in selected for f in row.fields})
        original_budget = min(client.config.context_chars // 3, client.config.context_chars - json_size(payload))
        payload["材料原文"] = schema_review_material(sources, discovery.covered, selected, original_budget)
        obj = client.ask(f"schema.fields.{index}",
            "只定义当前对象的字段、主键、显示键和外键。优先采用文本能支持的通用业务字段；区分字段所属对象和口径。"
            "若独立目录对象已经存在，项目名称、类别和规格留在目录，执行记录通过目录编号关联，保留执行时间和结果。"
            "逐项核对原文中当前对象的标识、名称、类别、状态和归属，候选摘要可能漏提字段。"
            "医疗机构和科室可同时存在；初复诊状态与就诊时间含义不同。"
            "某个流程是否发生与该流程的意见内容含义不同；只在孤立补充说明中出现的说法先保留为比较证据。"
            "标识符用 string。可为记录建字符串主键，即使原文暂缺编号。"
            "同义概念优先使用术语目录中的名称和编码；新增字段写完整业务名，编码用拼音首字母。"
            "报告日期字段可以存带时间的原文日期，具体精度在实例抽取时决定。外键名称和目标字段保持一致。"
            "外键必须写入 properties。保持当前对象代码和名字。",
            payload, ObjectType, max_tokens=3200)
        if (obj.typeCode, obj.name) != (outline.code, outline.name):
            raise OutputError(f"表结构候选 {index} 的字段定义更改了对象代码或名称")
        objects.append(obj)
    context = {"来源举例": compact}
    links = client.ask("schema.links",
        "只在对象表列出的类型之间定义关系。关系两端的类型码和字段码都从对象表原样复制。"
        "sourceProperty 和 targetProperty 都是连接的字段。"
        "一对多时外键在 target 一侧；多对一时外键在 source 一侧。明确关系名称和基数；字段使用实际编码。",
        {**context, "对象表": dump(objects)}, Links, max_tokens=3000)
    proposals = [row for group in zip_longest(*evidence.values()) for row in group if row is not None]
    context["材料原文"] = schema_review_material(sources, discovery.covered, proposals, client.config.context_chars // 3)
    return review_schema(Schema(object_types=objects, link_types=links.link_types), context, client,
                         use_catalog=use_catalog)


def schema_review_material(sources: Sources, refs: list[str], proposals: list[Proposal], budget: int) -> list[dict]:
    """短批次给全部原文；长批次优先保留能回查候选对象的材料。"""
    rows = sources.selected(refs)
    if json_size(rows) <= budget:
        return rows
    ordered = dict.fromkeys(piece.ref for row in proposals for piece in row.evidence)
    remaining = budget - 2
    picked = []
    for row in sources.selected(list(ordered)):
        size = json_size(row) + 1
        if size <= remaining:
            picked.append(row)
            remaining -= size
    return picked


def review_field_gaps(schema: Schema, gaps: list[FieldGap], sources: Sources, client: ModelClient, *,
                      use_catalog: bool) -> tuple[Schema, dict]:
    """逐项查重和核实归属，只向已审定的表添加有证据的普通字段。"""
    objects = schema.object_index()
    history = []
    for index, gap in enumerate(gaps, 1):
        owner = objects[gap.type]
        if code := existing_field(owner, gap.field, use_catalog=use_catalog):
            history.append({"proposal": gap.model_dump(), "already_defined": f"{gap.type}.{code}"})
            continue
        # 保留引文前后的表头和归属线索，无需把全部候选的报告一并送入模型。
        excerpts = {}
        for piece in gap.evidence:
            text = sources.texts[piece.ref]
            start = text.index(piece.quote)
            excerpts[piece.ref] = text[max(0, start - 200):start + len(piece.quote) + 200]
        context = {"候选表": gap.type, "候选字段": gap.field, "出现来源数量": len(gap.sources),
                   "材料节选": excerpts, "当前表与字段": {
                       code: {"名称": obj.name, "字段": {key: prop.title for key, prop in obj.properties.items()}}
                       for code, obj in objects.items()}}
        if use_catalog:
            context["术语名称与编码"] = catalog_names()
        review = client.ask(f"schema.field_gap.{index}",
            "独立核对这一个字段候选。根据原文确认业务含义和所属表，再查这张表是否已有同义字段。"
            "项目目录描述可共享的项目定义，单次报告的结果、部位和报告日期应查单次记录的字段。"
            "已有字段、关联可表达的重复信息、孤立的争议描述或证据不足时，addition 填 null。"
            "确有缺失时，addition 仅包含一个属于现有表的普通字段，type_code 从当前表选择。"
            "字段可使用有证据的目录条目。code 用拼音首字母或目录编码。"
            "evidence 的 ref 取材料编号，quote 逐字复制一段能证明该字段的连续原文。reason 写一句判定依据。",
            context, FieldGapReview, max_tokens=800)
        entry = {"proposal": gap.model_dump(), "review": review.model_dump()}
        history.append(entry)
        addition = review.addition
        if addition is None:
            continue
        target = objects.get(addition.type_code)
        if (target is None or not addition.evidence.check(sources, set(excerpts))
                or addition.evidence.quote not in excerpts[addition.evidence.ref]):
            raise OutputError(f"schema.field_gap.{index}：补充字段的所属表或引文无效")
        code, definition = addition.code, addition.definition
        if use_catalog and (term := catalog_field(target.name, definition.title)):
            code, definition = term.code, definition.model_copy(update={"title": term.title})
        if existing := existing_field(target, definition.title, use_catalog=use_catalog):
            entry["already_defined"] = f"{target.typeCode}.{existing}"
            continue
        if code in target.properties:
            raise OutputError(f"schema.field_gap.{index}：补充字段编码与现有字段重复")
        objects[target.typeCode] = target.model_copy(update={"properties": {**target.properties, code: definition}})
        entry["added"] = f"{target.typeCode}.{code}"
    result = schema.model_copy(update={"object_types": list(objects.values())})
    assert_valid(result.issues(), "补充字段后的表结构")
    return result, {"field_gap_reviews": history}


def review_schema(initial: Schema, context: dict, client: ModelClient, *, use_catalog: bool) -> tuple[Schema, dict]:
    """程序检查引用，模型检查语义；失败修订最多两轮。"""
    proposed = initial
    history = []
    feedback = []
    if use_catalog:
        context = {**context, "术语名称与编码": catalog_names()}
    for turn in range(3):
        if feedback:
            changed = client.ask(f"schema.repair.{turn}",
                "根据原文和反馈修订整份表结构。保留正确的对象和字段编码。保证外键存在，且关系两端属性和方向正确。"
                "对象范围已在候选阶段确定。保留当前所有对象编码，按反馈修订字段和关系。"
                "只根据当前输入的证据修订；反馈中的推测须由原文支持。简述修订依据。"
                "保留原文明确支持的目录对象和属性；仅出现在术语目录里的条目不构成增删依据。"
                "暂无法归属的冲突保留在比较阶段。",
                {**context, "当前表结构": proposed.model_dump(), "修订反馈": feedback}, SchemaDraft, max_tokens=5200)
            if changed.schema_definition.object_index().keys() != proposed.object_index().keys():
                feedback = ["修订只能调整当前对象的字段和关系，保留全部对象编码"]
                history.append({"round": turn + 1, "validation": feedback})
                continue
            proposed = changed.schema_definition
            history.append({"round": turn + 1, "revision_rationale": changed.rationale})
        feedback = proposed.issues()
        if feedback:
            history.append({"round": turn + 1, "validation": feedback})
            continue
        if use_catalog:
            proposed, changes = canonicalize_schema(proposed)
            if changes:
                history.append({"round": turn + 1, "terminology_changes": changes})
        review = client.ask(f"schema.review.{turn}",
            "独立审核表结构的对象粒度、各字段归属、关系方向、外键和证据覆盖。"
            "只审查表结构字段下的当前方案。核对候选原文中的常规字段是否遗漏。不同业务含义的字段应分别保留。"
            "逐个检查已建对象：原文明确给出名称、类别或状态时，确认它们所在的字段。"
            "有原文支持的医疗领域扩展项和其他字段使用相同的审核要求。术语目录仅供查找业务名称。"
            "issues 只列需要修订的具体问题，每项附原文依据；已通过的检查和猜测不写入 issues。"
            "共享项目的名称可通过检查记录的外键取得，缺少同名副本不算漏项。"
            "一个关系两端字段的编码可以不同，程序已检查编码有效性。指出会影响抽取的具体语义问题。"
            "可直接使用且无实质问题时返回 accept、分数至少 90、issues 为空。",
            {**context, "表结构": proposed.model_dump()}, Review, max_tokens=1400)
        history.append({"round": turn + 1, "review": review.model_dump()})
        if review.passed:
            return proposed, {"history": history}
        feedback = review.issues or ["按审核要求检查有无缺失的核心对象与属性"]
    assert_valid(feedback, "表结构未通过审核")
    raise AssertionError("表结构审核未产生结果")
