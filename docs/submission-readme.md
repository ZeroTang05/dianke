# D01 医疗记录自动标注

程序从任务一材料归纳表结构和关系，按结构整理记录，再分别判断任务一的属性冲突和任务二的时序变化。

## 构建和运行

在解压后的目录构建 Ubuntu 22.04、x86-64 容器：

```sh
docker build --platform linux/amd64 -t d01-solution .
```

评测服务通过 `/app/program/solution/run.sh` 启动程序，把输入目录只读挂载到 `/input`，并通过 `--out-dir` 指定输出目录（默认 `/output`）。输入文件为 `task1/inputs.json` 和 `task2/inputs.json`。

评测服务注入 `LLM_BASE_URL`（模型网关地址）、`LLM_MODEL`（模型名）和 `LLM_API_KEY`（调用令牌）。程序也读取 `LLM_USE_JSON_SCHEMA`（按指定 JSON 结构返回）和 `LLM_ENABLE_THINKING`（思考模式）配置。所有阶段共用每分钟最多 120 次请求的限速。

## 输出文件

| `--task` | 需要的输入 | 输出（相对 `--out-dir`） |
| --- | --- | --- |
| `all`（默认） | 两项任务的输入 | 下列三份文件 |
| `task1` | 任务一输入 | `task1/labels.json` |
| `task2` | 两项任务的输入，任务一用来归纳表结构 | `task2/labels-t1.json`、`task2/labels-t2.json` |

程序校验本次所需的输出后写入结果目录。运行摘要和阶段审核记录存在独立的工作目录，用于定位问题。
