"""按允许清单打包参赛代码，排除数据、密钥、缓存和验证工具。"""

import argparse
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("Dockerfile", "run.sh", "requirements.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team")
    parser.add_argument("version")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[\w.-]+", part) or part in {".", ".."} for part in (args.team, args.version)):
        parser.error("队名和版本只可包含字母、数字、下划线、点和短横线")
    files = {name: ROOT / name for name in REQUIRED}
    # 提交包使用独立说明，避免把本地平台配置和开发命令带入评测。
    files["README.md"] = ROOT / "docs/submission-readme.md"
    # 评分脚本只在本地查看标准答案，容器只携带推理代码。
    sources = sorted(path for path in (ROOT / "src/d01").glob("*.py") if path.name != "evaluation.py")
    files.update((path.relative_to(ROOT).as_posix(), path) for path in sources)
    if not sources or any(not path.is_file() for path in files.values()):
        raise ValueError("提交文件不齐全")
    out = ROOT / "dist" / f"{args.team}_code_{args.version}.zip"
    out.parent.mkdir(exist_ok=True)
    with NamedTemporaryFile(dir=out.parent, delete=False) as temporary:
        draft = Path(temporary.name)
    try:
        with ZipFile(draft, "w", ZIP_DEFLATED) as archive:
            for name, path in files.items():
                archive.write(path, name)
        with ZipFile(draft) as archive:
            if archive.testzip() is not None:
                raise ValueError("提交包完整性校验失败")
        draft.replace(out)
    finally:
        draft.unlink(missing_ok=True)
    print(out)


if __name__ == "__main__":
    main()
