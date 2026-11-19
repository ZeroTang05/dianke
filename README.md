# D01 初赛标注程序

程序从任务一的原文归纳表结构，再按这套结构整理两个任务的记录。每条事实保留原文出处。记录按患者和业务编号关联，模型分开判断冲突和时序变化。

## 本地运行

用 `uv` 安装 Python 3.10 环境和锁定的依赖：

```sh
uv venv --python 3.10
uv pip sync requirements.txt
source .venv/bin/activate
```

本地验证入口 `scripts/run_siliconflow.sh` 使用硅基流动中国站和 `Qwen/Qwen3.5-27B`，默认关闭思考模式。通过进程环境提供 `LLM_API_KEY`（密钥），可用终端隐式输入或密钥管理器注入。用以下命令验证接口，测试内容不含医疗信息：

```sh
./scripts/run_siliconflow.sh --probe
```

通用入口 `./run.sh` 从 `LLM_BASE_URL`（模型接口地址）和 `LLM_MODEL`（模型名）读取配置，正式评测由主办方提供这两项配置和密钥。`LLM_USE_JSON_SCHEMA=1` 需要网关支持按指定结构返回 JSON；`LLM_ENABLE_THINKING` 按网关要求设置。

以下是仓库开发步骤。`dev_data.py` 接收含「任务一」「任务二」的官方验证集目录，将首轮 20 条材料、完整输入和标准答案分别放在 `.runs/dev/pilot`、`.runs/dev/full` 和 `.runs/dev/gold`：

```zsh
python scripts/dev_data.py *1016*/验证集
python scripts/check_data_invariants.py .runs/dev
```

确认数据获准发往配置中的模型平台后，启动首轮试跑：

```sh
./scripts/run_siliconflow.sh --in-dir .runs/dev/pilot --out-dir output/pilot --cache-dir .runs/cache
python scripts/check_answers.py --in-dir .runs/dev/pilot --answers output/pilot --gold .runs/dev/gold --pilot --report .runs/pilot-report.json
```

首轮报告用 `pilot_groups` 列出六组结果，便于逐组核对。通过首轮验收后，将输入和输出路径分别改为 `.runs/dev/full` 和 `output/full`，评估时将 `--pilot` 换成 `--holdout-from .runs/dev/pilot`，分开查看首轮组和其余组。

报告同时给出原始匹配分和编号对齐后的实例分。只有能凭原文业务信息唯一对应的记录才对齐，输出文件保持原样。`instance_differences` 按对象和字段统计匹配、漏项、多报和取值差异。`comparison_evidence` 单独核对冲突的来源组合、时序和伪变化的来源与取值，并补充按天核对的结果。严格分数和原始时间精度保持不变。评估时加 `--work-run` 指向生成这些结果的运行目录，可以同时核对文件、候选覆盖、新请求数和缓存命中数。

验证集的参考标注可能有遗漏。医疗机构或药品等目录内的对象和字段，有原文支持并通过审核时会正常导出。评估报告的 `catalog_extension_candidates`（目录扩展候选）列出参考标注未覆盖的对象、字段和相关关联，同时提供实例和关联数量。回查原文和模型审核记录后，判断扩展是否合理。严格匹配分数用于和参考标注对照。

如果接口在记录抽取完成后中断，可用 `scripts/check_extracted_instances.py --help` 查看离线检查参数。该脚本读取运行目录、原始输入、标准答案和真实响应缓存，只核对表结构、实例和关联，不访问接口。恢复接口后，用相同的运行参数和缓存目录重跑，程序会重用未改动步骤的响应。缓存按模型区分，切换模型后会单独请求并记录结果。

`--workers` 控制并发，默认 8；`LLM_RPM` 控制所有阶段每分钟合计请求数，最高 120。程序随任务完成补充下一项，保留输入顺序。模型接口返回 `401`、`402` 或 `403` 时，停止发起新的请求，已发出的请求仍会收尾。

### 本地对照开关

每次使用一个开关，在相同输入上和默认流程比较。

| 参数 | 这一轮改变什么 |
| --- | --- |
| `--record-group-mode single` | 按单份材料定位记录 |
| `--candidate-mode grouped` | 比较时使用初始分组，省略患者编号合组 |
| `--comparison-window-mode sequential` | 长组只按材料顺序切分，省略关联证据窗口 |
| `--skip-record-review` | 省略逐条记录的模型复核 |
| `--skip-comparison-review` | 省略冲突和时序结果的模型复核 |
| `--omit-medical-catalog` | 建模时省略医疗候选目录及术语命名表 |
| `--omit-schema-example` | 建模时省略独立示例 |

分别用 `--gold` 和 `--work-run` 为基准轮、试验轮生成评分报告，再运行：

```sh
python scripts/compare_experiments.py .runs/base-report.json .runs/trial-report.json --report .runs/experiment.json
```

对照命令核对材料、标准答案、代码和模型配置，只接受改动一个开关的两轮。报告分别列出新请求、缓存命中和实际耗时；比较执行耗时时，先确认两轮的缓存使用情况相当。

每次运行在 `.runs/work/<运行编号>/` 保存运行设置、输入输出指纹、请求统计、逐条复核和候选材料。`validated/` 保存已通过校验的阶段答案，`summary.json` 记录完成状态、中断位置和各类问题数量。`issues.json` 按任务列出未通过审核的记录和比较、未解决的关联及未能确认的实例取值。清单列出材料编号和审核条目序号，便于回查。运行遇到输入或结果校验错误时，详细内容保存在 `failure-details.json`。

`--cache-only` 配合 `--cache-dir` 使用，只读取相同模型与请求的真实缓存，不需要密钥；缺少某条响应时停止。工作目录与缓存可能含医疗内容，仅用于本地排查。最终输出只包含比赛 JSON。

## Docker 与提交

```sh
docker build --platform linux/amd64 -t d01-solution .
python scripts/package_solution.py TEAM 0.1.0
```

容器入口为 `/app/program/solution/run.sh`，默认从只读 `/input` 读取材料，写入 `/output`。`--task all` 生成三份结果；`task1` 生成任务一结果；`task2` 也读取任务一输入，完成建模和结构漏项复核，再生成任务二的两份结果。`--in-dir`、`--out-dir` 可覆盖对应路径。正式评测由主办方注入模型配置。

| 输出 | 内容 |
| --- | --- |
| `task1/labels.json` | 对象、字段、关系和冲突 |
| `task2/labels-t1.json` | 实例记录和关联 |
| `task2/labels-t2.json` | 时序状态和伪变化 |

提交包仅包含运行需要的文件，并使用独立的评测运行说明 `docs/submission-readme.md` 作为包内的 `README.md`。技术方案单独提交。比赛成绩以官方评测为准。
