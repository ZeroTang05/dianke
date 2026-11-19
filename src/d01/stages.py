"""工作流共用的证据、审查结果和模型输入分批工具。"""

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
from typing import Annotated, Literal, TypeVar

from pydantic import Field

from .contracts import Contract, Text
from .inputs import Sources


T = TypeVar("T")
R = TypeVar("R")


def parallel_in_order(run: Callable[[T], R], jobs: Iterable[T], workers: int) -> list[R]:
    """按完成情况补充任务，最多保留 workers 个在途任务；返回顺序与输入一致。"""
    indexed = enumerate(jobs)
    results: dict[int, R] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Future[R], int] = {}

        def start_next() -> None:
            job = next(indexed, None)
            if job is not None:
                index, item = job
                pending[executor.submit(run, item)] = index

        try:
            for _ in range(workers):
                start_next()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=pending.__getitem__):
                    index = pending.pop(future)
                    results[index] = future.result()
                # 先读取已完成任务的异常，再启动下一批。
                for _ in done:
                    start_next()
        finally:
            for future in pending:
                future.cancel()
    return [results[index] for index in range(len(results))]


class Evidence(Contract):
    ref: Text
    quote: Text

    def check(self, sources: Sources, allowed: set[str] | None = None) -> bool:
        return (allowed is None or self.ref in allowed) and bool(self.quote.strip()) and self.quote in sources.texts.get(self.ref, "")


class FieldGap(Contract):
    """记录抽取时发现的候选字段及少量可回查的原文。"""

    type: Text
    field: Text
    sources: list[Text]
    evidence: Annotated[list[Evidence], Field(min_length=1, max_length=3)]


class Review(Contract):
    verdict: Literal["accept", "revise"]
    score: Annotated[int, Field(ge=0, le=100)]
    issues: list[Text]

    @property
    def passed(self) -> bool:
        return self.verdict == "accept" and self.score >= 90 and not self.issues


def json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def batches(values: Iterable[T], limit: int) -> Iterator[list[T]]:
    """按实际 JSON 长度分批，保留每个输入单位和原有顺序。"""
    group: list[T] = []
    size = 2
    for value in values:
        length = json_size(value) + 1
        if length > limit:
            raise ValueError(f"单条输入超过 {limit} 字的阶段预算，请提高 LLM_CONTEXT_CHARS")
        if group and size + length > limit:
            yield group
            group, size = [], 2
        group.append(value)
        size += length
    if group:
        yield group


def dump(items: Iterable[Contract]) -> list[dict]:
    return [item.model_dump(mode="json", exclude_none=True) for item in items]
