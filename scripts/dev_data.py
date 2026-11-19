"""从官方验证集准备本地输入和单独存放的标准答案。"""

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d01.inputs import MATERIALS, write_json


PILOT = {"task1": {"M000589", "M000600", "M000703"}, "task2": {"M000003", "M000233", "M000235"}}
FOLDERS = {"task1": "任务一", "task2": "任务二"}
LABELS = {
    "task1": {"labels.json": "labels.json"},
    "task2": {"label-t1.json": "labels-t1.json", "label-t2.json": "labels-t2.json"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path, help="官方验证集目录，内含任务一、任务二")
    parser.add_argument("--out", type=Path, default=ROOT / ".runs/dev")
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.validation.resolve()):
        parser.error("准备目录必须位于官方验证集之外")
    result = {}
    for task, folder in FOLDERS.items():
        source = args.validation / folder
        materials = MATERIALS.validate_json((source / "inputs.json").read_bytes())
        pilot = [row for row in materials if PILOT[task].intersection(row.id.split("_"))]
        seen = {key for row in pilot for key in row.id.split("_")} & PILOT[task]
        if seen != PILOT[task]:
            raise ValueError(f"{task} 缺少试跑组：{sorted(PILOT[task] - seen)}")
        for dataset, rows in (("full", materials), ("pilot", pilot)):
            write_json(args.out / dataset / task / "inputs.json", [row.model_dump() for row in rows])
        target = args.out / "gold" / task
        target.mkdir(parents=True, exist_ok=True)
        for original, output in LABELS[task].items():
            shutil.copyfile(source / original, target / output)
        result[task] = {"pilot": len(pilot), "full": len(materials)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
