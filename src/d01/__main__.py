"""D01 命令行入口。"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import sys
from typing import Literal

from .contracts import Contract
from .inputs import Sources
from .model import ModelClient, ModelConfig, ModelError
from .workspace import PipelineOptions, Workspace, run_discovery, run_pipeline, run_schema, runtime_version


class Probe(Contract):
    status: Literal["ok"]


def main() -> int:
    parser = argparse.ArgumentParser(description="D01 医疗记录标注")
    parser.add_argument("--probe", action="store_true", help="用不含医疗数据的短请求验证模型接口")
    parser.add_argument("--in-dir", type=Path, default=Path(os.environ.get("DATA_ROOT", "/input")))
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "/output")))
    parser.add_argument("--task", choices=("all", "task1", "task2"), default="all")
    parser.add_argument("--work-dir", type=Path, default=Path(".runs/work"))
    parser.add_argument("--schema-only", action="store_true", help="执行规律发现和表结构生成")
    parser.add_argument("--cache-dir", type=Path, help="本地响应缓存目录，含材料衍生信息")
    parser.add_argument("--cache-only", action="store_true", help="仅重放真实响应缓存，不连接模型接口")
    parser.add_argument("--workers", type=int, default=8, help="模型请求并发数")
    parser.add_argument("--candidate-mode", choices=("indexed", "grouped"), default="indexed", help="本地对照：编号索引或材料初分组")
    parser.add_argument("--comparison-window-mode", choices=("tracked", "sequential"), default="tracked",
                        help="本地对照：长组补充关联证据窗口，或仅按材料顺序切分")
    parser.add_argument("--record-group-mode", choices=("discovered", "single"), default="discovered", help="本地对照：按规律分组或逐份材料定位")
    parser.add_argument("--skip-record-review", action="store_true", help="本地对照：省略实例抽取复核")
    parser.add_argument("--skip-comparison-review", action="store_true", help="本地对照：省略冲突和时序复核")
    parser.add_argument("--omit-medical-catalog", action="store_true", help="本地对照：建模时省略医疗候选目录")
    parser.add_argument("--omit-schema-example", action="store_true", help="本地对照：建模时省略输入输出示例")
    args = parser.parse_args()
    options = PipelineOptions(candidate_mode=args.candidate_mode, comparison_window_mode=args.comparison_window_mode,
                              record_group_mode=args.record_group_mode,
                              record_review=not args.skip_record_review,
                              comparison_review=not args.skip_comparison_review,
                              use_catalog=not args.omit_medical_catalog, use_example=not args.omit_schema_example)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.cache_only and (args.probe or args.cache_dir is None):
            parser.error("--cache-only 需要 --cache-dir，且不能用于接口连通性检查")
        client = ModelClient(ModelConfig.from_environment(require_key=not args.cache_only), cache=args.cache_dir,
                             cache_only=args.cache_only)
        if args.probe:
            result = client.ask("probe", '连通性检查，请输出 {"status":"ok"}。', {}, Probe, max_tokens=128)
            print(json.dumps({"status": result.status, **client.summary()}, ensure_ascii=False))
            return 0
        if args.out_dir.resolve().is_relative_to(args.in_dir.resolve()) or args.work_dir.resolve().is_relative_to(args.in_dir.resolve()):
            parser.error("输出目录和工作目录应与只读输入分开")
        if args.cache_dir and args.cache_dir.resolve().is_relative_to(args.in_dir.resolve()):
            parser.error("缓存目录应与只读输入分开")
        work = Workspace.create(args.work_dir)
        model = client.config
        work.save("settings.json", {
            "task": args.task, "schema_only": args.schema_only, "workers": args.workers,
            "runtime_version": runtime_version(), "options": asdict(options),
            "model": {"endpoint_digest": sha256(model.base_url.encode("utf-8")).hexdigest(),
                      "name": model.model, "json_schema": model.json_schema,
                      "thinking": model.thinking, "rpm": model.rpm, "timeout": model.timeout,
                      "context_chars": model.context_chars, "cache_enabled": client.cache is not None},
        })
        complete = False
        failure = None
        try:
            if args.schema_only:
                work.phase = "task1 读取输入"
                task1 = Sources.load(args.in_dir, "task1")
                discovery = run_discovery({"task1": task1}, client, work)
                schema = run_schema(task1, discovery["task1"], client, work, options)
                work.counts.update(object_types=len(schema.object_types), link_types=len(schema.link_types))
            else:
                run_pipeline(args.in_dir, args.out_dir, client, work, args.task, args.workers, options)
            complete = True
        except ModelError as exc:
            failure = {"phase": work.phase, "kind": type(exc).__name__, "message": str(exc)}
            logging.error("%s：%s", work.phase, exc)
            return 1
        except (ValueError, OSError) as exc:
            failure = {"phase": work.phase, "kind": type(exc).__name__}
            # 详细校验错误可能带模型生成的文本，保留在本地工作目录。
            work.save("failure-details.json", {**failure, "detail": str(exc)})
            logging.error("%s失败（%s），详情见 failure-details.json", work.phase, failure["kind"])
            return 1
        except KeyboardInterrupt:
            failure = {"phase": work.phase, "kind": "KeyboardInterrupt"}
            return 130
        finally:
            work.summary(client, complete=complete, failure=failure)
            if not complete:
                logging.info("运行记录保存在 %s", work.run_dir)
        print(json.dumps({"work_dir": str(work.run_dir), "counts": work.counts, **client.summary()}, ensure_ascii=False))
    except (ValueError, OSError, ModelError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
