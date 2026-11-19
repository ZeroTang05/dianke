"""让模型对候选证据做冲突或时序判断，再独立复核。"""

from datetime import datetime
from itertools import groupby, pairwise, product
import re
from typing import Annotated, Literal

from pydantic import Field

from .contracts import (ChangeValue, Claim, Conflict, Contract, Schema, State, StatusPair, Task1, Text, Timeline, TimelineDraft,
                        date_order, validate_conflicts, validate_timelines)
from .graph import Candidate
from .inputs import Sources
from .model import ModelClient, OutputError
from .records import Record, stable_code
from .stages import Evidence, batches, dump, json_size, parallel_in_order


class ConflictDraft(Contract):
    incompatibility: Text
    claims: Annotated[list[Claim], Field(min_length=2)]


class ModeledConflict(ConflictDraft):
    field_key: Text


class ModeledConflicts(Contract):
    items: list[ModeledConflict]


class UnmodeledConflict(ConflictDraft):
    owner_type: Text


class UnmodeledConflicts(Contract):
    items: list[UnmodeledConflict]


class AdditionalClaim(Contract):
    claim: Claim | None
    reason: Text


class TimelineStateEvidence(ChangeValue):
    raw_time: Text
    raw_value: Text


class EvidencedTimeline(TimelineDraft):
    values: Annotated[list[TimelineStateEvidence], Field(min_length=2)]


class Timelines(Contract):
    items: list[EvidencedTimeline]


class IndexedStatusPair(StatusPair):
    timeline_index: int


class StatusPairChoice(Contract):
    option: Text
    prior_quote: Text
    post_quote: Text


class StatusPairs(Contract):
    items: list[StatusPairChoice]


# 相邻选项绑定一个序列和两份报告，模型只提交编号和原文短语。
PairOptions = dict[str, tuple[int, tuple[str, str], tuple[str, str]]]


class ComparisonIssue(Contract):
    kind: Literal["missing", "incorrect"]
    reason: Text
    candidate_index: int | None
    source_ids: Annotated[list[Text], Field(min_length=1, max_length=3)]


class ComparisonReview(Contract):
    verdict: Literal["accept", "revise"]
    score: Annotated[int, Field(ge=0, le=100)]
    issues: list[ComparisonIssue]


class UnmodeledIssue(ComparisonIssue):
    evidence: Annotated[list[Evidence], Field(min_length=1, max_length=3)]


class UnmodeledReview(ComparisonReview):
    issues: list[UnmodeledIssue]


class DisputedDecision(Contract):
    verdict: Literal["supported", "refuted", "uncertain"]
    reason: Text
    evidence: Annotated[list[Evidence], Field(min_length=1, max_length=4)]


RULES = {
    "conflicts": "判断同一患者的多份材料中是否有互不相容的事实。一般随时间变化的检查结果不构成冲突；"
    "本轮只检查有明确表结构归属的属性。field_key 从字段选项原样选择，例如两条互斥的检查结果可归入检查记录的检查结果字段。"
    "同一患者的不同次记录也可能冲突：已明确发生且持续有效的事实，与其后报告中的相反事实可以构成冲突。"
    "例如原文说某装置已永久移除，之后的同一对象报告却说该装置仍在位。查证是否有恢复或替换的证据。"
    "每条冲突给出所有互斥方的原始 source_id、短取值和逐字 raw_text，incompatibility 写明为何两者无法同时成立。"
    "普通不一致、仅有缺失信息或患者身份不同的情况不输出。",
    "unmodeled_conflicts": "专门检查现有字段无法准确表达的冲突。先在同一患者或记录下找原文中的断言，再逐条核对反证。"
    "同时比较具体事件与覆盖该事件的范围性陈述，或事实与流程说明。核实其适用对象和时段；没有提及某事不等于明确否认。"
    "例如一份设备日志说此次校准做了双人签核，另一份材料写有第二人的签核意见；适用于这次校准的说明明确说该类校准无第二人签核。"
    "前两份材料分别与说明构成两条证据配对。签核意见字段只描述意见内容，不能代替是否实施签核的属性。"
    "两份都支持双人签核的材料互相印证，不能配成冲突。政策说明出现在另一条记录中时，核查原文是否将政策推广到同类记录。"
    "同一机构、同一检查类型、同一流程的具体执行与通用规定相左时，可标注为材料间的流程口径冲突。"
    "仅日期不同不足以证明制度变更；原文明确写明不同生效时段、机构、适用条件或例外时，分别判断各自范围。"
    "先找出范围性否定说明，再逐份找与它不相容的证据；每份独立的相反证据单独提交一条。"
    "每条给出各方 source_id、原文取值和逐字 raw_text。据附录里的意见推知流程时，raw_text 应包含意见的来源说明，value 填意见正文。"
    "incompatibility 逐条说明哪方支持什么、哪方与之冲突，简述其共同适用范围。"
    "只返回无法准确归入现有字段的矛盾，owner_type 填最贴合的对象代码。可准确归字段的冲突交给另一轮。",
    "timelines": "在同一患者和同一关联项目或其他业务维度下，提取实际发生的状态变化。zjhm 使用患者在原文中已有的证件号。"
    "changeItem 填变化属性编码，relatedItem 填用于定位比较维度的属性编码，例如检查项目名称；"
    "relatedItemVal 与该字段业务含义一致：名称字段填名称，编号字段填实际编号。"
    "直接归属患者且无需比较维度时 relatedItem 与 relatedItemVal 均为空。"
    "values 按时间给出每次核实的状态值，包含之后重复的状态记录。ref 是原始材料 ID。"
    "比较检查结果时，value 只取本次报告明确写出的检查结果。独立的前后对照、变化程度或归档说明不拼入结果值。"
    "每条序列至少两个不同时间点。ref、zjhm、changeItem 均为非空字符串。"
    "使用报告或事件日期，不用就诊号推断日期。raw_time 填该来源中逐字的日期时间，raw_value 填该来源中逐字的状态短语。"
    "time 是同一日期时间的规范写法，保留原文精度。reason 无原文依据时写空字符串。",
    "pseudo_changes": "只检查给定的一对相邻报告，找对同一观察项的含义相同、措辞不同的两段原文。"
    "优先比较检查前后对照、进展或恢复程度等并列描述；这些原文短语可能独立于主状态摘要。"
    "按该材料中可证实的状态粒度判断等价。两份报告对照相同基线，主状态和其他范围描述一致，"
    "且没有证据表明状态进一步改变时，可把不同的定性措辞视为同一状态。此时不能仅根据用词差异判定恢复程度不同。"
    "例如设备两次检测均为可用，"
    "对初次故障的恢复分别描述为“功能基本恢复”和“主要功能已恢复”，并且两次均说明同一辅助功能待修。"
    "共享一部分文字、遗漏细节、增加新发现或状态前后不同都不足以判定等价。"
    "option 原样填相邻报告选项的编号。prior_quote 和 post_quote 分别逐字摘自选项的前、后报告，保留程度和相对时间词。"
    "截取表达差异的完整短语，保留紧邻的‘较前’等比较词，去掉外围相同的背景和后续描述。"
    "当两次主状态逐字相同时，不输出这两段相同的文字；继续查报告中前后对照的描述是否措辞不同。"
    "没有合适的对则返回空 items。",
}

REVIEW_SCOPE = {
    "conflicts": "只审核能归到字段的互斥断言。漏项必须给出至少两份材料及它们各自的相反说法。",
    "unmodeled_conflicts": "只审核无法准确归到现有字段的冲突。区分同类流程的材料口径不一致与单纯随时间变化。"
    "核对机构、检查类型和流程。审查的是材料中的口径是否一致，不能证明医院违规也可以有口径冲突。"
    "‘依据现行规定’没有给出生效时间。只有能在 evidence 里逐字引用实际生效时间或例外条件，才以此否决候选。"
    "每条意见的 evidence 填可回查的 ref 和原文 quote。"
    "漏项须指出相反说法、各自来源及共同适用范围。",
    "timelines": "只审核同一患者同一属性的状态序列。同一序列可包含不同日期重复的主状态。"
    "程序已核对每个状态的 raw_time、raw_value 来自该 ref，逐个按完整编号对应报告。"
    "可指出引用的日期或短语属于其他事件，但请给出同一报告里标明事件或结果的具体原文。"
    "某个状态有误时，candidate_index 填该状态所属的序列编号，具体日期和问题写进 reason。"
    "核对 value 是否对应所选字段的本次结果；单独写出的前后对照描述不要求加入结果值。"
    "整条序列缺失时须指出至少两个报告时间、来源和值。伪变化由后续阶段判断。",
    "pseudo_changes": "本轮仅审核指定报告对中的等价表述，可以使用主结果之外的原文。"
    "先核对两段是否描述同一观察项、同一基线、同等状态。遗漏、增减发现或明确改善/恶化属于语义差异，不得算作等价。"
    "真实变化由时序阶段负责；没有等价表述时，空列表应通过。漏项必须给出两段具体原文并说明其含义为何相同。",
}


DISPUTE_SCOPE = {
    "unmodeled_conflicts": (
        "判断两段材料对同一机构、同类事项的流程口径是否相左。具体做过的流程和同类事项无需该流程的通用声明可以相左，"
        "即使两份记录日期不同。此处标注文本差异，不判断历史合规。"
        "原文明确给出不同生效时段、适用范围或例外，且足以解释双方时，才依据该信息排除口径冲突。"
        "复核驳回依据若仅是猜测规则可能改变，按共有的机构、对象类型和流程重新判断。"
    ),
    "pseudo_changes": (
        "判断两段短语是否表达同一观察项的同等状态。分别核对观察项、比较基线、主结果及其余可见变化。"
        "若两份报告对同一基线给出相同的主结果和范围描述，只用两种定性词描写同方向的变化，且无额外证据表明程度继续变化，"
        "可以判为等价。‘明显’表示变化可见，‘多数’一类措辞描述范围，两者可同时成立。"
        "报告时间不同是本任务的前提。新增发现、仅一方有异常、明确进一步变化或基线不一致时，驳回该对。"
    ),
}


def event_dates(candidate: Candidate, schema: Schema) -> dict[str, datetime]:
    """只用审核过的日期字段；同一材料有多个日期时交给模型判断。"""
    dates: dict[str, set[datetime]] = {ref: set() for ref in candidate.refs}
    objects = schema.object_index()
    for row in candidate.records:
        if row.locator.scope != "event":
            continue
        properties = objects[row.locator.type_code].properties
        for fact in row.draft.facts:
            prop = properties.get(fact.field)
            if prop is None or not any(word in prop.title for word in ("日期", "时间")):
                continue
            if not isinstance(fact.value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:[T ][0-9]{2}:[0-9]{2}(?::[0-9]{2})?)?", fact.value):
                continue
            try:
                day = date_order(fact.value)
            except ValueError:
                continue  # 事实仍交给模型比较；无效日期不提供排序依据。
            for evidence in fact.evidence:
                if evidence.ref in dates:
                    dates[evidence.ref].add(day)
    return {ref: next(iter(days)) for ref, days in dates.items() if len(days) == 1}


def window_records(records: list[Record], refs: set[str]) -> list[Record]:
    """长组切窗后只发送能从本窗口回查的事实和关联。"""
    visible = {r.slot for r in records if refs.intersection(r.locator.refs)}

    def narrow(items):
        return [item.model_copy(update={"evidence": evidence}) for item in items
                if (evidence := [entry for entry in item.evidence if entry.ref in refs])]

    return [Record(row.slot, row.locator.model_copy(update={"refs": [ref for ref in row.locator.refs if ref in refs]}),
                   row.draft.model_copy(update={"facts": narrow(row.draft.facts),
                                                "connections": narrow([link for link in row.draft.connections
                                                                       if link.target_slot in visible]),
                                                "proposed_fields": narrow(row.draft.proposed_fields)}))
            for row in records if row.slot in visible]


def candidate_parts(candidate: Candidate, sources: Sources, schema: Schema, limit: int, *, use_tracks: bool) -> list[Candidate]:
    """长组按顺序切分，并把分散的关联证据补进单独窗口。"""
    groups = list(batches(sources.selected(candidate.refs), limit))
    if len(groups) == 1:
        return [Candidate(f"{candidate.key}_1", candidate.refs, candidate.records, candidate.patients)]

    identity_refs = {ref for record in candidate.records if record.locator.scope == "patient"
                     for ref in record.locator.refs if ref in candidate.refs}
    identity = next((ref for ref in candidate.refs if ref in identity_refs), None)
    tracks = candidate.tracks if use_tracks else ()
    result = []
    covered: dict[str, list[set[str]]] = {}

    def add(refs: list[str], *, supplemental: bool = False) -> None:
        # 每个窗口最多补一份患者材料。先检查实际比较材料是否已被完整覆盖。
        evidence = set(refs)
        if supplemental and any(evidence <= window for window in covered.get(refs[0], [])):
            return
        if identity and identity not in evidence:
            refs = [*refs, identity]
        if len(refs) < 2:
            return
        shared = set(refs)
        suffix = stable_code("evidence", *refs) if supplemental else str(len(result) + 1)
        result.append(Candidate(f"{candidate.key}_{suffix}", refs, window_records(candidate.records, shared),
                                candidate.patients))
        for ref in refs:
            covered.setdefault(ref, []).append(shared)

    def add_windows(parts: list[list[dict]], *, supplemental: bool = False) -> None:
        previous = None
        for group in parts:
            refs = [row["id"] for row in group]
            if previous is not None:
                refs.insert(0, previous)
            previous = refs[-1]
            add(refs, supplemental=supplemental)

    add_windows(groups)
    dates = event_dates(candidate, schema) if tracks and sources.task == "task2" else {}
    for track in tracks:
        refs = sorted(track, key=dates.__getitem__) if all(ref in dates for ref in track) else list(track)
        add_windows(list(batches(sources.selected(refs), limit)), supplemental=True)
    return result


def comparison_jobs(candidates: list[Candidate], sources: Sources, schema: Schema, limit: int, *, use_tracks: bool = True) -> list[Candidate]:
    """生成实际提交给模型的窗口，供执行和覆盖率检查共用。"""
    jobs = []
    for item in candidates:
        jobs.extend(candidate_parts(item, sources, schema, limit, use_tracks=use_tracks))
    return jobs


def _resolve_conflict_codes(items: list[Conflict], schema: Schema) -> None:
    """只转换结构中已有、能唯一匹配的中文名称；未知或歧义交给校验。"""
    objects = schema.object_index()
    for item in items:
        owner = objects.get(item.owner_type)
        if owner is None:
            matches = [obj for obj in objects.values() if obj.name == item.owner_type]
            if len(matches) != 1:
                continue
            owner = matches[0]
            item.owner_type = owner.typeCode
        if item.attribute_status == "已创建" and item.property not in owner.properties:
            codes = [code for code, prop in owner.properties.items() if prop.title == item.property]
            if len(codes) == 1:
                item.property = codes[0]


def compare(sources: Sources, schema: Schema, jobs: list[Candidate], client: ModelClient,
            workers: int, kind: Literal["conflicts", "timelines"], *, review_enabled: bool = True
            ) -> tuple[list[Conflict] | list[Timeline], list[dict]]:
    """分别判断两类冲突和两类时序结果，每个子步骤独立复核。"""
    structure = schema.model_dump(exclude_none=True)
    fields = {f"{obj.typeCode}.{code}": (obj.typeCode, code, f"{obj.name} · {prop.title}")
              for obj in schema.object_types for code, prop in obj.properties.items()}

    def judge(window: str, step: str, context: dict, known: dict[str, str],
              pair_options: PairOptions | None = None) -> tuple[list, dict]:
        contract = {"conflicts": ModeledConflicts, "unmodeled_conflicts": UnmodeledConflicts,
                    "timelines": Timelines, "pseudo_changes": StatusPairs}[step]
        rules = RULES[step]
        history: list[dict] = []
        feedback = []
        previous = None
        for attempt in range(2):
            try:
                if attempt and any(isinstance(item, dict) for item in feedback):
                    instructions = "根据结构与已核对的反馈修订待提交候选。用反馈指出的来源重新核对原文并补上遗漏。" + rules
                    payload = {k: v for k, v in context.items() if k != "已审核记录"}
                else:
                    instructions, payload = rules, context
                response = client.ask(f"{sources.task}.{step}.{window}.{attempt}", instructions,
                    {**payload, "上一版": previous, "反馈": feedback}, contract, max_tokens=3400)
                rows = response.items
                if isinstance(response, (ModeledConflicts, UnmodeledConflicts)):
                    # 选中字段后直接取得归属对象，两个值始终一致。
                    if isinstance(response, ModeledConflicts):
                        if any(row.field_key not in fields for row in response.items):
                            raise OutputError("冲突的 field_key 必须原样取自字段选项")
                        rows = [Conflict(owner_type=fields[row.field_key][0], property=fields[row.field_key][1],
                                         attribute_status="已创建", claims=row.claims) for row in response.items]
                    else:
                        rows = [Conflict(owner_type=row.owner_type, property="", attribute_status="未创建",
                                         claims=row.claims) for row in response.items]
                    _resolve_conflict_codes(rows, schema)
                    validate_conflicts(Task1(model=schema, conflicts=rows), known)
                elif isinstance(response, Timelines):
                    rows = resolve_timelines(response.items, known)
                    validate_timelines(rows, schema, known)
                    if any(not any(t.zjhm in text for text in known.values()) for t in rows):
                        raise ValueError("时序主体的证件号缺少本组原文证据")
                else:
                    assert isinstance(response, StatusPairs) and pair_options is not None
                    rows = resolve_status_pairs(response.items, pair_options, known)
                previous = response.model_dump()
            except (OutputError, ValueError) as exc:
                feedback = [str(exc)]
                history.append({"format_issues": feedback})
                continue
            if not review_enabled:
                return rows, {"step": step, "sources": list(known), "accepted": True, "history": history}

            try:
                # 校验审核反馈时可以重问审核者，保留已通过格式校验的候选。
                review_error = ""
                review_rows = dump(rows) if isinstance(response, StatusPairs) else previous["items"]
                for check in range(2):
                    try:
                        review = client.ask(
                            f"{sources.task}.{step}_review.{window}.{attempt}.{check}",
                            "审核待审核中的比较结果。核对对象是否相同、原文适用范围、时间和字段。"
                            "患者、就诊和检查等基础实例由另一阶段生成，待审核列表里无需列出它们。"
                            "只有存在本步骤的具体发现时才报告漏项；空列表可以通过。"
                            "source_ids 填支撑该意见的原始材料编号。漏项填 missing、candidate_index 为 null，说明缺少的具体发现。"
                            "已有候选错误填 incorrect，candidate_index 从待审核列表逐项标出的编号中选择，"
                            "说明错误在哪，与哪段原文不符。"
                            "表述差异需指出影响判定的具体含义。只报本步骤的问题，其他步骤负责的发现不计。"
                            "有问题时填 revise；无问题时填 accept、评分 90 到 100、issues 留空。\n" + REVIEW_SCOPE[step],
                            {**{k: v for k, v in context.items() if k != "已审核记录"},
                             "待审核": [{"candidate_index": i, **row} for i, row in enumerate(review_rows)],
                             "上轮反馈": feedback, "审核格式反馈": review_error},
                            UnmodeledReview if step == "unmodeled_conflicts" else ComparisonReview, max_tokens=2300,
                        )
                        validate_comparison_review(review, len(response.items), known)
                        break
                    except OutputError as exc:
                        review_error = str(exc)
                else:
                    raise OutputError(review_error)
                history.append({"draft": previous, "review": review.model_dump()})
                if review.verdict == "accept" and review.score >= 90 and not review.issues:
                    return rows, {"step": step, "sources": list(known), "accepted": True, "history": history}
                if attempt == 0 and step in DISPUTE_SCOPE and review.issues and all(
                    issue.kind == "incorrect" for issue in review.issues
                ):
                    decisions = adjudicate_disputed_rows(client, f"{sources.task}.{step}.{window}", step,
                                                        context, known, review_rows, review.issues)
                    history.append({"adjudication": decisions})
                    if all(item["decision"]["verdict"] == "supported" for item in decisions):
                        return rows, {"step": step, "sources": list(known), "accepted": True, "history": history}
                feedback = dump(review.issues) or ["请重新核查相关状态和证据"]
            except OutputError as exc:
                feedback = [str(exc)]
                history.append({"draft": previous, "format_issues": feedback})
                break
        return [], {"step": step, "sources": list(known), "accepted": False, "history": history}

    def analyze(candidate: Candidate) -> tuple[list, list[dict]]:
        known = {ref: sources.texts[ref] for ref in candidate.refs}
        context = {"结构": structure, **candidate.payload(sources)}
        if json_size(context) > client.config.context_chars:
            context["已审核记录"] = [
                {"定位": r.locator.model_dump(), "字段": [(f.field, f.value) for f in r.draft.facts]}
                for r in candidate.records
            ]
        if kind == "conflicts":
            context["字段选项"] = {key: option[2] for key, option in fields.items()}
        rows, report = judge(candidate.key, kind, context, known)
        if kind == "conflicts":
            others, second = judge(candidate.key, "unmodeled_conflicts", {"结构": structure, "材料": context["材料"]}, known)
            extra, checks = expand_unmodeled_conflicts(others, known, schema, client, candidate.key)
            return [*rows, *others, *extra], [report, second, *checks]
        reports = [report]
        # 每对报告单独判断和复核，一对未通过时保留同一序列的其他结论。
        for key, option in adjacent_report_options(rows).items():
            index, prior, post = option
            timeline = rows[index]
            refs = list(dict.fromkeys((prior[0], post[0])))
            pair_context = {"材料": sources.selected(refs), "变化字段": timeline.changeItem,
                            "比较项目": timeline.relatedItemVal,
                            "报告对": {"option": key, "前报告": {"ref": prior[0], "time": prior[1]},
                                       "后报告": {"ref": post[0], "time": post[1]}},
                            "已核实的结果": dump(v for v in timeline.values if (v.ref, v.time) in (prior, post))}
            pairs, pair_report = judge(f"{candidate.key}.{key}", "pseudo_changes", pair_context,
                                       {ref: known[ref] for ref in refs}, {key: option})
            reports.append(pair_report)
            timeline.statusPairs.extend(compact_status_pair(pair) for pair in pairs)
        return rows, reports

    reports = []
    conflicts: dict[tuple, Conflict] = {}
    timelines: dict[tuple, Timeline] = {}
    for result, window_reports in parallel_in_order(analyze, jobs, workers):
        reports.extend(window_reports)
        for item in result:
            if isinstance(item, Conflict):
                key = (item.owner_type, item.property, item.attribute_status,
                       tuple(sorted((claim.source_id, repr(claim.value)) for claim in item.claims)))
                conflicts.setdefault(key, item)
            else:
                key = (item.zjhm, item.changeItem, item.relatedItem, item.relatedItemVal)
                merged = timelines.get(key)
                if merged is None:
                    timelines[key] = item
                    continue
                samples = {(t.time, t.ref, repr(t.value)): t for t in (*merged.values, *item.values)}
                merged.values = sorted(samples.values(), key=lambda v: (date_order(v.time), v.time, v.ref, repr(v.value)))
                pairs = {(p.priorStatus.ref, p.postStatus.ref, p.priorStatus.time, p.postStatus.time,
                          repr(p.priorStatus.value), repr(p.postStatus.value)): p
                         for p in (*merged.statusPairs, *item.statusPairs)}
                merged.statusPairs = sorted(pairs.values(), key=lambda p: (date_order(p.priorStatus.time), date_order(p.postStatus.time)))
    if kind == "conflicts":
        return list(conflicts.values()), reports
    for item in timelines.values():
        for pair in prune_nonadjacent_pairs(item):
            reports.append({"step": "merge_pseudo_changes", "sources": [pair.priorStatus.ref, pair.postStatus.ref],
                            "accepted": False, "history": [{"format_issues": ["合并后两个报告时间不再相邻"]}]})
    return list(timelines.values()), reports


def expand_unmodeled_conflicts(items: list[Conflict], texts: dict[str, str], schema: Schema,
                               client: ModelClient, window: str) -> tuple[list[Conflict], list[dict]]:
    """已有互斥说法时，逐份材料查找可以独立支撑其中一方的证据。"""
    extra, reports = [], []
    seen: set[tuple[str, frozenset[str]]] = {
        (row.owner_type, frozenset(claim.source_id for claim in row.claims)) for row in items if len(row.claims) == 2
    }
    for index, item in enumerate(items):
        if len(item.claims) != 2:
            continue
        original = frozenset(claim.source_id for claim in item.claims)
        for n, (ref, text) in enumerate(texts.items()):
            if ref in original:
                continue
            step = f"task1.conflict_evidence.{window}.{index}.{n}"
            facts = {"已确认冲突": dump(item.claims), "待检查材料": {"id": ref, "content": text}}
            history: list[dict] = []
            try:
                result = client.ask(step,
                    "核对新材料是否提供了同一事项、同一适用范围内的独立证据，支持已确认冲突的一方。"
                    "只关注与原冲突相同的事实。补充说明中署名的意见可以支持该意见的出具或参与流程。"
                    "有证据时，claim 引用新材料的原始编号和逐字原文；value 只填支持判断的取值。"
                    "若证据是补充材料中的引述意见，value 填意见正文，raw_text 保留其出处和正文。"
                    "reason 说明为何有关；没有证据时 claim 为 null。",
                    facts, AdditionalClaim, max_tokens=1000)
                history.append({"draft": result.model_dump()})
                if result.claim is None:
                    reports.append({"step": step, "sources": [ref], "accepted": True, "history": history})
                    continue
                if result.claim.source_id != ref or not result.claim.raw_text.strip() or result.claim.raw_text not in text:
                    raise OutputError("补充证据需要引用当前材料的逐字原文")
                choice = client.ask(f"{step}.verify",
                    "独立核对新材料里的引文。请在原冲突双方中选择与新证据真正互斥的一方："
                    "claim 填被反驳一方的完整证据，原样复制；reason 解释两者为何不能同时成立。"
                    "若原冲突已确认是同类流程的口径冲突，判断新材料是否独立支持其中一方，并延用已核实的适用范围。"
                    "仅补充信息、支持同一方、适用范围不同或证据不足时，claim 为 null。",
                    {**facts, "新证据": result.claim.model_dump()}, AdditionalClaim, max_tokens=1000)
                history.append({"review": choice.model_dump()})
                if choice.claim is not None:
                    if choice.claim not in item.claims:
                        raise OutputError("补充证据复核只可从原冲突中选择反方")
                    key = (item.owner_type, frozenset((ref, choice.claim.source_id)))
                    if key not in seen:
                        new = item.model_copy(update={"claims": [result.claim, choice.claim]})
                        validate_conflicts(Task1(model=schema, conflicts=[new]), texts)
                        extra.append(new)
                        seen.add(key)
                reports.append({"step": step, "sources": [ref], "accepted": True, "history": history})
            except (OutputError, ValueError) as exc:
                history.append({"format_issues": [str(exc)]})
                reports.append({"step": step, "sources": [ref], "accepted": False, "history": history})
    return extra, reports


def validate_comparison_review(review: ComparisonReview, count: int, texts: dict[str, str]) -> None:
    """确保修订意见能定位到本轮的候选和原文。"""
    for issue in review.issues:
        index = issue.candidate_index
        valid_index = index is None if issue.kind == "missing" else index is not None and 0 <= index < count
        if not valid_index:
            raise OutputError(f"比较复核的 incorrect 必须使用待审核中标出的编号（本轮 {count} 项）；missing 填 null")
        if not set(issue.source_ids) <= texts.keys():
            raise OutputError("比较复核的 source_ids 必须来自本轮材料")
        if isinstance(issue, UnmodeledIssue) and any(
            piece.ref not in issue.source_ids or not piece.quote.strip() or piece.quote not in texts[piece.ref]
            for piece in issue.evidence
        ):
            raise OutputError("冲突复核的引文必须逐字出自意见列出的来源")


def adjudicate_disputed_rows(client: ModelClient, prefix: str, step: str, context: dict, texts: dict[str, str],
                             candidates: list[dict], issues: list[ComparisonIssue]) -> list[dict]:
    """独立核对首次复核驳回的条目，结论需要覆盖该候选的两侧来源。"""
    by_candidate: dict[int, list[ComparisonIssue]] = {}
    for issue in issues:
        assert issue.candidate_index is not None
        by_candidate.setdefault(issue.candidate_index, []).append(issue)
    reports = []
    for index, objections in by_candidate.items():
        row = candidates[index]
        refs = ({claim["source_id"] for claim in row["claims"]} if step == "unmodeled_conflicts"
                else {row["priorStatus"]["ref"], row["postStatus"]["ref"]})
        decision = client.ask(f"{prefix}.adjudicate.{index}",
            DISPUTE_SCOPE[step] + "按这些规则独立判断一个候选。supported 表示该发现成立，refuted 表示不成立，"
            "uncertain 表示原文无法判断。逐字引用候选两侧材料，在 reason 说明是否存在支持驳回理由的具体信息。",
            {"候选": row, "待核实的反对意见": dump(objections), "材料": context["材料"],
             **({"已核实的结果": context["已核实的结果"]} if step == "pseudo_changes" else {})},
            DisputedDecision, max_tokens=1400)
        if not refs <= {piece.ref for piece in decision.evidence} or any(
            not piece.quote.strip() or piece.quote not in texts.get(piece.ref, "") for piece in decision.evidence
        ):
            raise OutputError("争议复核需要候选两侧可回查的原文")
        reports.append({"candidate_index": index, "decision": decision.model_dump()})
    return reports


def resolve_timelines(items: list[EvidencedTimeline], texts: dict[str, str]) -> list[Timeline]:
    """原文证据留在审核记录，比赛输出只保留状态字段。"""
    result = []
    for item in items:
        for state in item.values:
            text = texts.get(state.ref, "")
            if any(not value.strip() or value not in text for value in (state.raw_time, state.raw_value)):
                raise ValueError("每个时序状态的日期和取值引文必须逐字来自该状态自己的材料")
            # 原文已经是 ISO 日期时直接比较，允许空格和 T 两种日期时间写法。
            if re.fullmatch(r"[0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2}(?:[T ][0-9]{2}:[0-9]{2}(?::[0-9]{2})?)?)?)?", state.raw_time):
                if len(state.raw_time) != len(state.time) or date_order(state.raw_time) != date_order(state.time):
                    raise ValueError("时序时间必须与该状态的日期引文一致并保留精度")
        result.append(Timeline(**item.model_dump(exclude={"values"}), statusPairs=[], values=[
            ChangeValue.model_validate(v.model_dump(exclude={"raw_time", "raw_value"})) for v in item.values
        ]))
    return result


def compact_status_pair(pair: IndexedStatusPair) -> StatusPair:
    """移除两段表述共有的末尾分句，保留各自的原文差异。"""
    before, after = pair.priorStatus, pair.postStatus
    a, b = str(before.value), str(after.value)
    tail = r"[，；,;]\s*([^，；,;]+)$"
    while (left := re.search(tail, a)) and (right := re.search(tail, b)):
        if left[1].strip() != right[1].strip() or a[:left.start()].strip() == b[:right.start()].strip():
            break
        a, b = a[:left.start()].rstrip(), b[:right.start()].rstrip()
    return StatusPair(priorStatus=before.model_copy(update={"value": a}), postStatus=after.model_copy(update={"value": b}))


def adjacency(timeline: Timeline) -> tuple[set[tuple[datetime, datetime]], set[tuple[str, datetime]]]:
    """索引同一序列的相邻时间和可用来源。"""
    occurrences = {(state.ref, date_order(state.time)) for state in timeline.values}
    dates = sorted({day for _, day in occurrences})
    return set(zip(dates, dates[1:])), occurrences


def pair_is_adjacent(pair: StatusPair, dates: set[tuple[datetime, datetime]], refs: set[tuple[str, datetime]]) -> bool:
    """检查两侧来源与日期是否都属于同一条相邻状态对。"""
    before, after = pair.priorStatus, pair.postStatus
    a, b = date_order(before.time), date_order(after.time)
    return (a, b) in dates and (before.ref, a) in refs and (after.ref, b) in refs


def prune_nonadjacent_pairs(timeline: Timeline) -> list[StatusPair]:
    """合并窗口可能补进一个新时间点；保留全部状态，移回失去相邻性的对供回查。"""
    dates, refs = adjacency(timeline)
    retained, removed = [], []
    for pair in timeline.statusPairs:
        (retained if pair_is_adjacent(pair, dates, refs) else removed).append(pair)
    timeline.statusPairs = retained
    return removed


def adjacent_report_options(timelines: list[Timeline]) -> PairOptions:
    """同一序列内按日期分组，只枚举相邻两组的报告。"""
    options: PairOptions = {}
    for index, timeline in enumerate(timelines):
        locations = dict.fromkeys((state.ref, state.time) for state in timeline.values)
        dated = sorted(locations, key=lambda item: date_order(item[1]))
        groups = [list(group) for _, group in groupby(dated, key=lambda item: date_order(item[1]))]
        for previous, current in pairwise(groups):
            for earlier, later in product(previous, current):
                options[f"P{len(options) + 1}"] = (index, earlier, later)
    return options


def resolve_status_pairs(choices: list[StatusPairChoice], options: PairOptions,
                         texts: dict[str, str]) -> list[IndexedStatusPair]:
    """核对两段逐字引文，使用选项里的报告编号与日期生成结果。"""
    result = []
    seen = set()
    for choice in choices:
        if choice.option not in options:
            raise ValueError("伪变化必须使用本轮的相邻报告选项编号")
        index, (prior_ref, prior_time), (post_ref, post_time) = options[choice.option]
        left, right = choice.prior_quote.strip(), choice.post_quote.strip()
        if not left or left not in texts[prior_ref] or not right or right not in texts[post_ref]:
            raise ValueError("伪变化两侧取值都需要逐字原文")
        if left == right:
            raise ValueError("伪变化需要两种不同的原文表述")
        key = (choice.option, left, right)
        if key in seen:
            raise ValueError("伪变化重复")
        seen.add(key)
        result.append(IndexedStatusPair(timeline_index=index,
            priorStatus=State(ref=prior_ref, time=prior_time, value=left),
            postStatus=State(ref=post_ref, time=post_time, value=right)))
    return result
