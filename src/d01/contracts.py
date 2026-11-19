"""比赛 JSON 格式，以及表结构和实例之间的引用约束。"""

from collections import Counter, defaultdict
from datetime import datetime
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


Text = Annotated[str, StringConstraints(min_length=1)]
Scalar = str | bool | int | float
Reference = tuple[str, str | bool | int | float]
Date = Annotated[str, StringConstraints(pattern=r"^\d{4}(?:-\d{2}(?:-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?)?)?$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True, allow_inf_nan=False)


class Material(Contract):
    id: Text
    content: Text


class Property(Contract):
    type: Literal["string", "number", "boolean"]
    title: Text


class ObjectType(Contract):
    name: Text
    description: str
    typeCode: Text
    properties: dict[str, Property]
    primaryKey: Text
    foreignKeys: list[Text]
    displayKey: Text | None = None


class LinkType(Contract):
    name: Text
    sourceTypeCode: Text
    targetTypeCode: Text
    sourceProperty: Text
    targetProperty: Text
    cardinality: Literal["ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY"]


Join = tuple[str, str, str, str]


def instance_joins(link: LinkType, objects: dict[str, ObjectType]) -> set[Join]:
    """把表结构的关系方向转成实例方向：持有外键的一端指向目标。"""
    source, sp = link.sourceTypeCode, link.sourceProperty
    target, tp = link.targetTypeCode, link.targetProperty
    result = set()
    if link.cardinality != "ONE_TO_MANY" and sp in objects[source].foreignKeys:
        result.add((source, sp, target, tp))
    if link.cardinality != "MANY_TO_ONE" and tp in objects[target].foreignKeys:
        result.add((target, tp, source, sp))
    return result


class Schema(Contract):
    object_types: Annotated[list[ObjectType], Field(min_length=1)]
    link_types: list[LinkType]

    def object_index(self) -> dict[str, ObjectType]:
        return {item.typeCode: item for item in self.object_types}

    def joins(self) -> set[Join]:
        objects = self.object_index()
        return set().union(*(instance_joins(link, objects) for link in self.link_types))

    def issues(self) -> list[str]:
        """列出可反馈给模型的结构问题，避免打印患者数据。"""
        errors = []
        objects = self.object_index()
        for code, count in Counter(item.typeCode for item in self.object_types).items():
            if count > 1:
                errors.append(f"对象编码重复：{code}")
        if len({item.name for item in self.object_types}) != len(self.object_types):
            errors.append("对象中文名称重复")
        declared: set[tuple[str, str]] = set()
        connected: set[tuple[str, str]] = set()
        for code, obj in objects.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", code):
                errors.append(f"对象 {code} 需要字母开头的代码")
            for field in obj.properties:
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", field):
                    errors.append(f"{code}.{field} 需要字母开头的字段代码")
            keys = [obj.primaryKey, *obj.foreignKeys]
            if obj.displayKey is not None:
                keys.append(obj.displayKey)
            for field in set(keys) - obj.properties.keys():
                errors.append(f"{code}.{field} 是尚未定义的键")
            if len(set(obj.foreignKeys)) != len(obj.foreignKeys):
                errors.append(f"{code} 外键重复")
            declared.update((code, fk) for fk in obj.foreignKeys)
        seen: set[frozenset[Join]] = set()
        for index, link in enumerate(self.link_types, 1):
            source, target = objects.get(link.sourceTypeCode), objects.get(link.targetTypeCode)
            if source is None or target is None:
                errors.append(f"关系 {index} 引用了未知对象")
                continue
            prop, other = source.properties.get(link.sourceProperty), target.properties.get(link.targetProperty)
            if prop is None or other is None:
                errors.append(f"关系 {index} 引用了未知字段")
                continue
            if prop.type != other.type:
                errors.append(f"关系 {index} 两端字段类型不同")
            joins = frozenset(instance_joins(link, objects))
            if not joins:
                errors.append(f"关系 {index} 的外键位置与关系方向不一致")
            if joins in seen:
                errors.append(f"关系 {index} 重复")
            seen.add(joins)
            connected.update((kind, fk) for kind, fk, _, _ in joins)
        errors.extend(f"{kind}.{fk} 缺少关系定义" for kind, fk in sorted(declared - connected))
        return errors


class Claim(Contract):
    source_id: Text
    value: Scalar
    raw_text: Text


class Conflict(Contract):
    owner_type: Text
    property: str
    attribute_status: Literal["已创建", "未创建"]
    claims: Annotated[list[Claim], Field(min_length=2)]


class Task1(Contract):
    model: Schema
    conflicts: list[Conflict]


class InstanceLink(Contract):
    targetProperty: Text
    targetPropertyVal: Scalar
    targetTypeCode: Text


class Instance(Contract):
    typeCode: Text
    typeName: Text
    data: dict[str, Scalar]
    relations: dict[str, InstanceLink]


class State(Contract):
    value: Scalar
    ref: Text
    time: Date


class ChangeValue(State):
    reason: str


class StatusPair(Contract):
    priorStatus: State
    postStatus: State


class TimelineDraft(Contract):
    zjhm: Text
    relatedItem: str
    relatedItemVal: Scalar
    changeItem: Text
    values: Annotated[list[ChangeValue], Field(min_length=2)]


class Timeline(TimelineDraft):
    statusPairs: list[StatusPair]


def assert_valid(issues: list[str], stage: str) -> None:
    if issues:
        raise ValueError(f"{stage}：" + "；".join(issues[:12]) + (f"（共 {len(issues)} 项）" if len(issues) > 12 else ""))


def scalar_matches(value: Scalar, kind: str) -> bool:
    return type(value) in {"string": (str,), "number": (int, float), "boolean": (bool,)}[kind]


def date_order(value: str) -> datetime:
    """按时间排序；月份或年份的精度仍保留在原输出里。"""
    if len(value) == 4:
        value += "-01-01"
    elif len(value) == 7:
        value += "-01"
    return datetime.fromisoformat(value)


def instance_report_date(value: Scalar) -> Scalar:
    """把实例报告日期写为 ISO 日期时间；原始精度保留在抽取记录和时序中。"""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?", value):
        return value
    return datetime.fromisoformat(value).isoformat(timespec="seconds")


def validate_conflicts(answer: Task1, texts: dict[str, str]) -> None:
    assert_valid(answer.model.issues(), "任务一表结构")
    objects = answer.model.object_index()
    issues = []
    for i, item in enumerate(answer.conflicts, 1):
        obj = objects.get(item.owner_type)
        if obj is None:
            issues.append(f"冲突 {i} 的对象类型未定义")
        elif item.attribute_status == "已创建" and item.property not in obj.properties:
            issues.append(f"冲突 {i} 必须引用已有字段")
        elif item.attribute_status == "未创建" and item.property:
            issues.append(f"冲突 {i} 的字段编码应当留空")
        if len({claim.source_id for claim in item.claims}) < 2:
            issues.append(f"冲突 {i} 需要至少两个独立来源")
        for claim in item.claims:
            if not claim.raw_text.strip() or claim.raw_text not in texts.get(claim.source_id, ""):
                issues.append(f"冲突 {i} 的引文无法从任务一材料回查")
    assert_valid(issues, "任务一冲突")


def validate_instances(items: list[Instance], schema: Schema) -> None:
    objects = schema.object_index()
    joins = schema.joins()
    targets: dict[tuple, int] = defaultdict(int)
    primary: set[tuple] = set()
    issues = []
    for i, item in enumerate(items, 1):
        obj = objects.get(item.typeCode)
        if obj is None:
            issues.append(f"实例 {i} 类型未知")
            continue
        if item.typeName != obj.name:
            issues.append(f"实例 {i} 类型名称与表结构不符")
        for field, value in item.data.items():
            p = obj.properties.get(field)
            if p is None or not scalar_matches(value, p.type):
                issues.append(f"实例 {i} 字段 {field} 未定义或值类型错误")
            targets[item.typeCode, field, type(value), value] += 1
        key = item.data.get(obj.primaryKey)
        if key is None or key == "" or (item.typeCode, type(key), key) in primary:
            issues.append(f"实例 {i} 缺少主键或主键重复")
        primary.add((item.typeCode, type(key), key))
        if set(item.data).intersection(obj.foreignKeys) != set(item.relations):
            issues.append(f"实例 {i} 的外键与关系条目不一致")
    for i, item in enumerate(items, 1):
        for fk, link in item.relations.items():
            key = (item.typeCode, fk, link.targetTypeCode, link.targetProperty)
            val = link.targetPropertyVal
            if key not in joins or type(item.data.get(fk)) is not type(val) or item.data.get(fk) != val:
                issues.append(f"实例 {i} 的 {fk} 与关系定义或外键值不一致")
            elif targets[link.targetTypeCode, link.targetProperty, type(val), val] != 1:
                issues.append(f"实例 {i} 的 {fk} 无法唯一定位目标")
    assert_valid(issues, "任务二实例")


def validate_timelines(items: list[Timeline], schema: Schema, texts: dict[str, str]) -> None:
    fields = {field for obj in schema.object_types for field in obj.properties}
    issues = []
    seen = set()
    for i, item in enumerate(items, 1):
        key = (item.zjhm, item.changeItem, item.relatedItem, item.relatedItemVal)
        if key in seen:
            issues.append(f"时序 {i} 重复")
        seen.add(key)
        if item.changeItem not in fields or (item.relatedItem and item.relatedItem not in fields):
            issues.append(f"时序 {i} 引用了未定义的字段")
        if bool(item.relatedItem) != (item.relatedItemVal != ""):
            issues.append(f"时序 {i} 的比较维度和取值需要同时填写")
        states = [*item.values, *(s for p in item.statusPairs for s in (p.priorStatus, p.postStatus))]
        if any(state.ref not in texts for state in states):
            issues.append(f"时序 {i} 的来源不在任务二材料中")
        try:
            dates = [date_order(state.time) for state in item.values]
            if dates != sorted(dates) or len(set(dates)) < 2:
                issues.append(f"时序 {i} 需要至少两个已排序时间点")
            if any(date_order(p.priorStatus.time) >= date_order(p.postStatus.time) for p in item.statusPairs):
                issues.append(f"时序 {i} 的伪变化时间先后不合法")
        except ValueError:
            issues.append(f"时序 {i} 包含无效日期")
    assert_valid(issues, "任务二时序")
