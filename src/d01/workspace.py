"""保存本次运行的阶段产物和计数；最终输出只包含官方文件。"""

from dataclasses import dataclass, field
from hashlib import sha256
import json
import logging
from pathlib import Path
import time
from uuid import uuid4

from .comparison import compare, comparison_jobs
from .contracts import Schema, Task1, validate_conflicts, validate_instances, validate_timelines
from .discovery import Discovery, discover
from .graph import Graph
from .inputs import Sources, material_digest, write_json
from .model import ModelClient
from .records import Extraction, extract
from .schema_stage import build_schema, review_field_gaps
from .stages import dump


LOG = logging.getLogger(__name__)


def schema_version(schema: Schema) -> str:
    """内容一致就得到相同版本，供证据和下游产物引用。"""
    return sha256(json.dumps(schema.model_dump(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def runtime_version() -> str:
    """只对推理源码计算版本；本地评分模块无需随比赛镜像运行。"""
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name != "evaluation.py":
            digest.update(path.name.encode("utf-8") + b"\0" + sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def review_issues(reviews: list[dict], artifact: str) -> list[dict]:
    """汇总最终未通过项；详细反馈与原文保存在完整审核文件中。"""
    issues = []
    for index, review in enumerate(reviews, 1):
        if review["accepted"]:
            continue
        latest = review["history"][-1] if review["history"] else {}
        reason = "格式或证据校验未通过" if "format_issues" in latest else "模型复核未通过"
        issues.append({"artifact": artifact, "entry": index, "reason": reason,
                       **{key: review[key] for key in ("slot", "type_code", "step", "sources") if key in review}})
    return issues


@dataclass(frozen=True)
class PipelineOptions:
    """记录每次本地对照更改的开关；默认全部使用参赛流程。"""

    candidate_mode: str = "indexed"
    comparison_window_mode: str = "tracked"
    record_group_mode: str = "discovered"
    record_review: bool = True
    comparison_review: bool = True
    use_catalog: bool = True
    use_example: bool = True


@dataclass
class Workspace:
    run_dir: Path
    started: float
    counts: dict = field(default_factory=dict)
    checkpoints: dict[str, str] = field(default_factory=dict)
    issues: dict[str, list[dict]] = field(default_factory=dict)
    phase: str = "准备运行"

    @classmethod
    def create(cls, root: Path) -> "Workspace":
        path = root / uuid4().hex[:12]
        path.mkdir(mode=0o700, parents=True)
        return cls(path, time.monotonic())

    def save(self, path: str, result: object) -> None:
        write_json(self.run_dir / path, result)

    def add_issues(self, task: str, category: str, issues: list[dict]) -> None:
        """按任务和步骤归档；按新版结构重抽后更新对应清单。"""
        self.issues[f"{task}.{category}"] = issues
        self.save("issues.json", self.issues)

    def schema(self, schema: Schema, review: dict) -> None:
        version = schema_version(schema)
        self.save("schema/" + version + ".json", {"model": schema.model_dump(exclude_none=True), **review})
        self.save("schema.json", {"version": version, "model": schema.model_dump(exclude_none=True)})

    def stage_output(self, relative_path: str, value: object) -> None:
        """保存已校验的阶段答案和指纹，方便复查中断的运行。"""
        checkpoint = f"validated/{relative_path}"
        self.save(checkpoint, value)
        self.checkpoints[relative_path] = sha256((self.run_dir / checkpoint).read_bytes()).hexdigest()

    def summary(self, client: ModelClient, *, complete: bool, failure: dict | None) -> None:
        self.save("issues.json", self.issues)
        self.save("summary.json", {"complete": complete, "elapsed_seconds": round(time.monotonic() - self.started, 2),
                                   "counts": self.counts, "validated_outputs": self.checkpoints,
                                   "issue_counts": {category: len(rows) for category, rows in self.issues.items()},
                                   "failure": failure, **client.summary()})


def run_discovery(sources: dict[str, Sources], client: ModelClient, work: Workspace) -> dict[str, Discovery]:
    found = {}
    for task, source in sources.items():
        work.phase = f"{task} 规律发现"
        LOG.info("%s：发现规律，材料 %d 条", task, len(source.texts))
        found[task] = discover(source, client)
        work.save(f"{task}/discovery.json", found[task].snapshot())
    return found


def run_schema(sources: Sources, discovery: Discovery, client: ModelClient, work: Workspace,
               options: PipelineOptions) -> Schema:
    work.phase = "task1 表结构归纳与审核"
    LOG.info("任务一：归纳表结构并审核")
    schema, review = build_schema(sources, discovery, client, use_catalog=options.use_catalog,
                                  use_example=options.use_example)
    work.schema(schema, review)
    LOG.info("表结构审核完成：%d 类对象，%d 类关系", len(schema.object_types), len(schema.link_types))
    return schema


def run_pipeline(root: Path, output: Path, client: ModelClient, work: Workspace, task: str,
                 workers: int, options: PipelineOptions) -> None:
    """各任务分别持有来源和事实；任务二使用同一次运行得到的任务一结构。"""
    if not 1 <= workers <= 32:
        raise ValueError("并发量应在 1–32 之间")
    work.phase = "task1 读取输入"
    sources = {"task1": Sources.load(root, "task1")}
    if task in {"all", "task2"}:
        work.phase = "task2 读取输入"
        sources["task2"] = Sources.load(root, "task2")
    work.save("inputs.json", {kind: material_digest(source.texts) for kind, source in sources.items()})
    discovery = run_discovery(sources, client, work)
    schema = run_schema(sources["task1"], discovery["task1"], client, work, options)
    wanted = list(sources) if task == "all" else [task]

    def extract_for(kind: str) -> Extraction:
        work.phase = f"{kind} 记录抽取与审核"
        LOG.info("%s：定位、逐条抽取并复核", kind)
        result = extract(sources[kind], discovery[kind], schema, client, workers,
                         group_mode=options.record_group_mode, review_enabled=options.record_review)
        version = schema_version(schema)
        artifact = f"{kind}/records-{version}.json"
        work.save(artifact,
                  {"schema_version": version, "records": [record.snapshot() for record in result.records],
                   "reviews": result.reviews, "missing_schema": dump(result.missing_schema)})
        work.add_issues(kind, "records", review_issues(result.reviews, artifact))
        return result

    # 所有模式先用任务一完成结构校验；任务二只读取最终版结构。
    task1_extraction = extract_for("task1")
    proposals = task1_extraction.missing_schema
    if proposals:
        work.phase = "task1 表结构漏项复核"
        LOG.info("任务一：核对抽取阶段提交的结构漏项")
        updated, review = review_field_gaps(schema, proposals, sources["task1"], client,
                                            use_catalog=options.use_catalog)
        before, after = schema_version(schema), schema_version(updated)
        work.save("schema-feedback.json", {"previous_version": before, "final_version": after, **review})
        if after != before:
            work.schema(updated, review)
            schema = updated
            if "task1" in wanted:
                task1_extraction = extract_for("task1")

    extractions = {"task1": task1_extraction} if "task1" in wanted else {}
    if "task2" in wanted:
        extractions["task2"] = extract_for("task2")

    counts = work.counts
    counts.update(object_types=len(schema.object_types), link_types=len(schema.link_types))
    for kind in wanted:
        source = sources[kind]
        extraction = extractions[kind]
        work.phase = f"{kind} 记录关联"
        graph = Graph(extraction.records, schema)
        counts[f"{kind}_unaccepted_records"] = sum(not review["accepted"] for review in extraction.reviews)
        work.add_issues(kind, "links", graph.unresolved)
        if kind == "task2":
            work.phase = "task2 实例组装与校验"
            items = graph.instances(source, client)
            work.add_issues(kind, "values", graph.unresolved_values)
            work.add_issues(kind, "links", graph.unresolved)
            validate_instances(items, schema)
            work.stage_output("task2/labels-t1.json", dump(items))
            counts["instances"] = len(items)
        work.save(f"{kind}/unresolved-links.json", graph.unresolved)
        counts[f"{kind}_unresolved_links"] = len(graph.unresolved)
        work.phase = f"{kind} 比较候选生成"
        candidates = comparison_jobs(graph.candidates(source, discovery[kind], options.candidate_mode),
                                     source, schema, client.config.context_chars // 3,
                                     use_tracks=options.comparison_window_mode == "tracked")
        work.save(f"{kind}/candidates.json", [{"key": c.key, "refs": c.refs, "patients": len(c.patients)} for c in candidates])
        LOG.info("%s：验证 %d 组候选", kind, len(candidates))
        counts[f"{kind}_comparison_windows"] = len(candidates)
        counts[f"{kind}_supplemental_windows"] = sum("_EVIDENCE_" in c.key for c in candidates)
        work.phase = f"{kind} 比较判断与复核"
        rows, reports = compare(source, schema, candidates, client, workers,
                                "conflicts" if kind == "task1" else "timelines", review_enabled=options.comparison_review)
        artifact = f"{kind}/comparison-review.json"
        work.save(artifact, reports)
        work.add_issues(kind, "comparisons", review_issues(reports, artifact))
        counts[f"{kind}_unaccepted_comparisons"] = sum(not review["accepted"] for review in reports)
        work.phase = f"{kind} 比较结果校验"
        if kind == "task1":
            answer = Task1(model=schema, conflicts=rows)
            validate_conflicts(answer, source.texts)
            work.stage_output("task1/labels.json", answer.model_dump(mode="json", exclude_none=True))
            counts["conflicts"] = len(rows)
        else:
            validate_timelines(rows, schema, source.texts)
            work.stage_output("task2/labels-t2.json", dump(rows))
            counts["timelines"] = len(rows)
    # 比赛输出目录只在各阶段完成后写入；中断时的阶段结果仍留在运行目录。
    work.phase = "导出比赛文件"
    for relative_path in work.checkpoints:
        value = json.loads((work.run_dir / "validated" / relative_path).read_bytes())
        write_json(output / relative_path, value)
    work.save("outputs.json", work.checkpoints)
