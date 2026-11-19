"""保持赛题的文本边界，通过任务编号和原始 ID 回查材料。"""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import TypeAdapter

from .contracts import Material


TaskName = Literal["task1", "task2"]
MATERIALS = TypeAdapter(list[Material])


@dataclass(frozen=True)
class Sources:
    task: TaskName
    texts: dict[str, str]

    @classmethod
    def load(cls, root: Path, task: TaskName) -> "Sources":
        path = root / task / "inputs.json"
        rows = MATERIALS.validate_json(path.read_bytes(), strict=True)
        duplicates = sum(count - 1 for count in Counter(row.id for row in rows).values())
        if not rows or duplicates or any(not row.id.strip() or not row.content.strip() for row in rows):
            raise ValueError(f"{path}：需要非空材料列表、唯一编号和非空文本；重复编号数 {duplicates}")
        return cls(task, {row.id: row.content for row in rows})

    def selected(self, ids: list[str]) -> list[dict[str, str]]:
        return [{"id": ref, "content": self.texts[ref]} for ref in ids]


def material_digest(texts: dict[str, str]) -> str:
    """包含材料顺序、编号和原文，便于核对两次运行使用的是同一份输入。"""
    return sha256(json.dumps(list(texts.items()), ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    """先写完临时文件再替换，读取方只看到完整 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
