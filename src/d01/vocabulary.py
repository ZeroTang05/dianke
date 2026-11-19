"""统一医疗术语的输出名称；对象和字段仍由模型根据原文发现。"""

from collections import Counter
from dataclasses import dataclass
from unicodedata import normalize

from .contracts import ObjectType, Schema, assert_valid


@dataclass(frozen=True)
class Term:
    code: str
    title: str
    aliases: tuple[str, ...] = ()

    def matches(self, name: str) -> bool:
        """判断名称是否与目录条目或其同义词一致。"""
        return _key(name) in {_key(value) for value in (self.title, *self.aliases)}


PATIENT_ID = Term("hzbh", "患者编号", ("患者标识", "患者ID", "病人编号"))
VISIT_ID = Term("jzbh", "就诊编号", ("就诊号", "就诊标识"))
EXAM_ID = Term("jcbh", "检查编号", ("检查项目编号", "项目编号"))
ORGANIZATION_ID = Term("yljgbh", "医疗机构编号", ("机构编号", "医院编号"))
DRUG_ID = Term("ypbh", "药品编号", ("药品编码",))

# 按对象限制同义词的范围。例如「编号」在检查项目与单次记录中含义不同。
CATALOG = (
    (Term("hz", "患者", ("患者信息", "患者主档", "患者档案", "病人")), (
        PATIENT_ID,
        Term("hzxm", "患者姓名", ("姓名", "病人姓名")),
        Term("zjhm", "证件号码", ("患者证件号码", "证件号", "身份证号", "身份证号码")),
        Term("xb", "性别", ("患者性别",)),
    )),
    (Term("jzjl", "就诊记录", ("就诊", "就诊信息", "就诊事件")), (
        VISIT_ID, PATIENT_ID, ORGANIZATION_ID,
        Term("yljg", "医疗机构", ("就诊机构", "就诊医疗机构", "医疗机构名称", "就诊医院")),
        Term("czfzbz", "初诊/复诊标志", ("就诊性质", "初复诊标志", "初诊复诊标志")),
    )),
    (Term("jc", "检查", ("检查项目", "检查项目目录", "检查目录")), (
        EXAM_ID,
        Term("jcxmmc", "检查项目名称", ("项目名称", "检查名称", "检查项目名")),
        Term("jclb", "检查类别", ("检查类型", "项目类别")),
    )),
    (Term("jcjl", "检查记录", ("检查执行记录", "检查报告记录", "检查项目记录")), (
        Term("jcjlbh", "检查记录编号", ("记录编号", "检查记录号")), VISIT_ID, EXAM_ID,
        Term("jcjg", "检查结果", ("检查所见", "检查结果描述")),
        Term("jcbw", "检查部位"),
        Term("bgrq", "报告日期", ("报告时间", "检查报告日期", "检查报告时间")),
    )),
    (Term("yljg", "医疗机构", ("就诊机构", "医疗机构档案")), (
        ORGANIZATION_ID,
        Term("yljgmc", "医疗机构名称", ("机构名称", "医院名称")),
        Term("yljglx", "医疗机构类型", ("机构类型",)),
    )),
    (Term("yp", "药品", ("药品信息", "药品目录")), (
        DRUG_ID,
        Term("ypmc", "药品名称", ("药名",)),
        Term("jx", "剂型"), Term("gg", "规格"), Term("yxqzt", "有效期状态"),
    )),
    (Term("yzyy", "医嘱用药", ("医嘱用药记录", "用药医嘱")), (
        Term("yzbh", "医嘱用药编号", ("用药医嘱编号",)), DRUG_ID, VISIT_ID,
        Term("dcjl", "单次剂量"), Term("dw", "剂量单位"), Term("yppl", "用药频次"),
        Term("tj", "给药途径"), Term("zxsj", "执行时间"), Term("zxts", "执行天数"),
    )),
)


def catalog_names() -> dict:
    """给模型提供简短的标准叫法，原文决定哪些条目适用。"""
    return {f"{obj.code} {obj.title}": {term.code: term.title for term in fields} for obj, fields in CATALOG}


def _key(value: str) -> str:
    return "".join(normalize("NFKC", value).casefold().split())


def _match(title: str, terms: tuple[Term, ...]) -> Term | None:
    return next((term for term in terms if term.matches(title)), None)


def catalog_field(object_name: str, field_name: str) -> Term | None:
    """按业务对象查找字段的标准名称和同义词。"""
    fields = next((fields for obj, fields in CATALOG if obj.matches(object_name)), ())
    return _match(field_name, fields)


def existing_field(obj: ObjectType, name: str, *, use_catalog: bool) -> str | None:
    """找到同一张表上定义过的字段，目录对照时可关闭同义词匹配。"""
    term = catalog_field(obj.name, name) if use_catalog else None
    return next((code for code, prop in obj.properties.items()
                 if _key(name) == _key(prop.title) or (term and term.matches(prop.title))), None)


def _renames(current: dict[str, str], choices: dict[str, Term]) -> dict[str, Term]:
    """保留会重名的候选，避免通过更名悄悄合并两条定义。"""
    selected = choices.copy()
    while selected:
        final = {code: (selected[code].code, selected[code].title) if code in selected else (code, title)
                 for code, title in current.items()}
        codes = Counter(code for code, _ in final.values())
        titles = Counter(_key(title) for _, title in final.values())
        safe = {code: term for code, term in selected.items()
                if codes[term.code] == titles[_key(term.title)] == 1}
        if safe == selected:
            break
        selected = safe
    return selected


def canonicalize_schema(schema: Schema) -> tuple[Schema, list[dict]]:
    """只改同义名称和编码，同时更新主键、外键及关系两端。"""
    assert_valid(schema.issues(), "术语归一输入")
    terms = tuple(term for term, _ in CATALOG)
    definitions = dict(CATALOG)
    matched = {obj.typeCode: term for obj in schema.object_types if (term := _match(obj.name, terms))}
    type_renames = _renames({obj.typeCode: obj.name for obj in schema.object_types}, matched)
    field_codes: dict[str, dict[str, str]] = {}
    type_codes: dict[str, str] = {}
    changes: list[dict] = []
    objects = []

    for obj in schema.object_types:
        matched_type = type_renames.get(obj.typeCode)
        type_code, title = (matched_type.code, matched_type.title) if matched_type else (obj.typeCode, obj.name)
        type_codes[obj.typeCode] = type_code
        if (type_code, title) != (obj.typeCode, obj.name):
            changes.append({"object": obj.typeCode, "from": [obj.typeCode, obj.name], "to": [type_code, title]})
        # 同义字段只在明确识别的对象内更名；含义有歧义的对象继续用原定义。
        aliases = definitions[matched_type] if matched_type else ()
        field_renames = _renames({code: prop.title for code, prop in obj.properties.items()}, {
            code: term for code, prop in obj.properties.items() if (term := _match(prop.title, aliases))
        })
        mapping = {code: field_renames[code].code if code in field_renames else code for code in obj.properties}
        field_codes[obj.typeCode] = mapping
        properties = {}
        for code, prop in obj.properties.items():
            name = field_renames[code].title if code in field_renames else prop.title
            if (mapping[code], name) != (code, prop.title):
                changes.append({"object": obj.typeCode, "from": [code, prop.title], "to": [mapping[code], name]})
            properties[mapping[code]] = prop.model_copy(update={"title": name})
        objects.append(obj.model_copy(update={"name": title, "typeCode": type_code, "properties": properties,
                                               "primaryKey": mapping[obj.primaryKey],
                                               "displayKey": mapping[obj.displayKey] if obj.displayKey else None,
                                               "foreignKeys": [mapping[code] for code in obj.foreignKeys]}))

    links = [link.model_copy(update={"sourceTypeCode": type_codes[link.sourceTypeCode],
                                     "targetTypeCode": type_codes[link.targetTypeCode],
                                     "sourceProperty": field_codes[link.sourceTypeCode][link.sourceProperty],
                                     "targetProperty": field_codes[link.targetTypeCode][link.targetProperty]})
             for link in schema.link_types]
    result = Schema(object_types=objects, link_types=links)
    assert_valid(result.issues(), "术语归一结果")
    return result, changes
