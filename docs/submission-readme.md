# D01 医疗记录自动标注

程序从任务一材料归纳表结构和关系，按结构整理记录，再分别判断任务一的属性冲突和任务二的时序变化。

## 构建和运行

在解压后的目录构建 Ubuntu 22.04、x86-64 容器：

```sh
docker build --platform linux/amd64 -t d01-solution .
```

评测服务通过 `/app/program/solution/run.sh` 启动程序。程序从 `DATA_ROOT` 指向的只读目录读取 `task1/inputs.json` 和 `task2/inputs.json`，默认目录为 `/input`。结果写入 `--out-dir` 指定的目录。未传此参数时使用 `OUTPUT_DIR`，默认目录为 `/output`。

| 主办方注入的变量 | 程序如何使用 |
| --- | --- |
| `LLM_BASE_URL` | OpenAI 兼容网关地址，程序在末尾加上 `/chat/completions` |
| `LLM_API_KEY` | 作为请求的调用令牌 |
| `LLM_MODEL` | 作为请求中的模型名 |
| `LLM_USE_JSON_SCHEMA` | `1` 时请求网关按指定 JSON 结构返回，`0` 时由程序校验模型返回的 JSON |
| `LLM_ENABLE_THINKING` | `1` 开启思考模式，`0` 关闭 |

所有阶段合计每分钟最多发送 120 次请求。单次请求默认超时 120 秒。

## 输出文件

| `--task` | 需要的输入 | 输出（相对 `--out-dir`） |
| --- | --- | --- |
| `all`（默认） | 两项任务的输入 | 下列三份文件 |
| `task1` | 任务一输入 | `task1/labels.json` |
| `task2` | 两项任务的输入，任务一用来归纳表结构 | `task2/labels-t1.json`、`task2/labels-t2.json` |

程序校验本次所需的输出后写入结果目录。容器将运行摘要和阶段审核记录存到 `/tmp/.runs/work/`，用于定位问题。
