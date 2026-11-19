"""讨论材料规律，验证来源和分组建议，保留覆盖清单。"""

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Annotated

from pydantic import Field

from .contracts import Contract, Text
from .inputs import Sources
from .model import ModelClient, OutputError
from .stages import Evidence, batches, dump


class Signal(Contract):
    description: Text
    examples: Annotated[list[Evidence], Field(min_length=1, max_length=4)]
    limits: str


class Discussion(Contract):
    signals: list[Signal]


class IdPattern(Contract):
    delimiter: Annotated[str, Field(min_length=1, max_length=3)]
    segment: Annotated[int, Field(ge=0, le=8)]

    def group(self, ref: str) -> str | None:
        parts = ref.split(self.delimiter)
        return parts[self.segment] if len(parts) > 1 and len(parts) > self.segment and parts[self.segment] else None


class Findings(Contract):
    pattern: IdPattern | None
    signals: list[Signal]
    priority_refs: list[Text]


class GroupChoice(Contract):
    candidate: Text | None


@dataclass(frozen=True)
class Discovery:
    findings: Findings
    covered: list[str]
    groups: list[list[str]]

    def snapshot(self) -> dict:
        return {"findings": self.findings.model_dump(), "covered": self.covered, "groups": self.groups}


def _verified_signals(items: list[Signal], sources: Sources, allowed: set[str]) -> list[Signal]:
    return [signal for signal in items if all(e.check(sources, allowed) for e in signal.examples)]


MAX_GROUP_SIZE = 30


def partition_refs(refs: list[str], pattern: IdPattern) -> list[list[str]]:
    """按一个编号位置分组；无法切分的编号各自保留。"""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for ref in refs:
        part = pattern.group(ref)
        groups[("pattern", part) if part else ("single", ref)].append(ref)
    return list(groups.values())


def verify_group_pattern(sources: Sources, covered: list[str], findings: Findings, client: ModelClient) -> IdPattern | None:
    """让模型比较不同分段的真实分组样本，核对组内患者或业务相同。"""
    seen = set()
    choices: dict[str, IdPattern] = {}
    examples = {}
    priority = set(findings.priority_refs)
    priority.update(e.ref for signal in findings.signals for e in signal.examples)
    separators = set(re.findall(r"[_:./|-]", "".join(covered)))
    candidates = [IdPattern(delimiter=sep, segment=i) for sep in sorted(separators) for i in range(9)]
    if findings.pattern:
        candidates.insert(0, findings.pattern)
    for pattern in candidates:
        groups = partition_refs(covered, pattern)
        signature = tuple(map(tuple, groups))
        if signature in seen:
            continue
        seen.add(signature)
        # 常量片段和文档角色容易生成大桶；只考虑能形成多个可用组的分段。
        repeated = [refs for refs in groups if 2 <= len(refs) <= MAX_GROUP_SIZE]
        if len(repeated) < 2 and not (len(groups) == 1 and repeated and len(covered) <= 8):
            continue
        choice = str(len(choices) + 1)
        choices[choice] = pattern
        candidates_by_evidence = sorted(repeated, key=lambda refs: -len(priority.intersection(refs)))
        examples[choice] = {"分隔符": pattern.delimiter, "段号": pattern.segment, "组数": len(groups),
                            "可比较材料数": sum(map(len, repeated)),
                            "示例组": [[{"id": ref, "材料开头": sources.texts[ref][:240]}
                                      for ref in group[:4]] for group in candidates_by_evidence[:2]]}
    if not choices:
        return None
    checked = client.ask(f"{sources.task}.discover.verify_pattern",
        "选择用于一起阅读同一患者的材料编号分段。对照每个示例组的材料开头，核对患者姓名、证件号、就诊号或其它业务线索。"
        "把不同患者的同一文档角色（如都为首页或报告）归在一起的分段不可选。材料组号可以跨不同文档角色或不同次就诊。"
        "比较所有分段，选使同一患者相关材料聚在一起的一项。candidate 填对应选项的字符串键；没有原文依据时填 null。",
        {"待核对分段": examples}, GroupChoice, max_tokens=300)
    if checked.candidate is None:
        return None
    if checked.candidate not in choices:
        raise OutputError("规律复核选择了未提供的分段")
    return choices[checked.candidate]


def discover(sources: Sources, client: ModelClient) -> Discovery:
    """全量覆读：每批先找规律再查反例，最终把可执行分组约束在 ID 分段上。"""
    observations = []
    covered = []
    for number, rows in enumerate(batches(sources.selected(list(sources.texts)), client.config.context_chars - 5500), 1):
        ids = {row["id"] for row in rows}
        evidence = {"任务": sources.task, "材料": rows}
        proposed = client.ask(f"{sources.task}.discover.propose.{number}",
            "查看本批全部材料，找编号结构、同一次就诊的分组线索、文档类型、重复和疑似冲突或复查位置。"
            "举出可复制的短原文片段，说明每条线索可能失效的情形。最多 5 条线索。",
            evidence, Discussion, max_tokens=1800)
        proposed_items = _verified_signals(proposed.signals, sources, ids)
        checked = client.ask(f"{sources.task}.discover.challenge.{number}",
            "结合原文检查候选分组和语义线索；特别找反例、跨组重复、同名、时间或记录归属的歧义。"
            "把核对后的线索和限制输出，证据须为原文。最多 5 条。",
            {**evidence, "待核对": dump(proposed_items)}, Discussion, max_tokens=1800)
        observations.extend(_verified_signals(checked.signals, sources, ids))
        covered.extend(row["id"] for row in rows)

    # 汇总输入里保留全部原始编号和抽样后的假设；分批讨论已经覆盖全文。
    while len(observations) > 1 and len(str(dump(observations))) > client.config.context_chars // 2:
        compressed = []
        for index, group in enumerate(batches(dump(observations), client.config.context_chars // 2), 1):
            result = client.ask(f"{sources.task}.discover.combine.{index}",
                "合并重复线索，保留限制和原样证据。最多输出两项最有支撑的线索。", group, Discussion, max_tokens=1000)
            compressed.extend(_verified_signals(result.signals[:2], sources, set(covered)))
        if len(compressed) >= len(observations):
            observations = compressed[:8]
            break
        observations = compressed
    final = client.ask(f"{sources.task}.discover.conclude",
        "汇总讨论结果，选出有证据的分组提示。只有当材料 ID 确实含可用于非语义初分组的段时才填写 pattern；"
        "segment 从 0 开始，例如 AB_VISIT123_S2 的下划线分隔第 1 段。没有此规律时填 null。"
        "优先材料 ID 只用于改动处理顺序，不能当成答案。原文证据必须逐字复制。",
        {"任务": sources.task, "材料数量": len(covered), "编号样本": covered[:300], "线索": dump(observations)},
        Findings, max_tokens=2000)
    if covered != list(sources.texts):
        raise RuntimeError("规律发现没有覆盖全部输入")
    if set(final.priority_refs) - sources.texts.keys():
        raise OutputError("规律发现引用了不存在的材料 ID")
    final.signals = _verified_signals(final.signals, sources, set(covered))
    final.pattern = verify_group_pattern(sources, covered, final, client)
    groups = partition_refs(covered, final.pattern) if final.pattern else [[ref] for ref in covered]
    result = [group for items in groups for group in ([items] if len(items) <= MAX_GROUP_SIZE else [[ref] for ref in items])]
    priority = set(final.priority_refs)
    result.sort(key=lambda refs: not bool(priority.intersection(refs)))
    return Discovery(final, covered, result)
