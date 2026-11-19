#!/bin/sh
# 跑完整验证集并保存评分。每次使用独立目录，模型响应可以跨轮复用。
set -eu
umask 077
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
run_dir=${1:?请提供一个新的运行目录}
task=${2:-all}
case "$task" in
    all|task2) ;;
    *) printf '验证范围只支持 all 或 task2\n' >&2; exit 2 ;;
esac
mkdir "$run_dir"
exec >>"$run_dir/run.log" 2>&1

write_status() {
    printf '%s\n' "$1" >"$run_dir/status.tmp"
    mv "$run_dir/status.tmp" "$run_dir/status"
}

finish() {
    code=$?
    if [ "$code" -eq 0 ]; then
        write_status complete
        printf '验证完成，评分报告保存在 %s/report.json\n' "$run_dir"
    else
        write_status "failed ($code)"
        printf '验证退出（退出码 %s），详见 %s/run.log\n' "$code" "$run_dir" >&2
    fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_status running
PATH="$root/.venv/bin:$PATH"
export PATH
validation_key=${LLM_API_KEY:-${SILICONFLOW_API_KEY:-}}
: "${validation_key:?请通过环境提供模型密钥}"
rpm=${LLM_RPM:-45}
workers=${D01_VALIDATION_WORKERS:-4}
printf '开始全量验证（%s，每分钟最多 %s 次请求，并发 %s）\n' "$task" "$rpm" "$workers"
LLM_API_KEY="$validation_key" LLM_RPM="$rpm" LLM_TIMEOUT=120 \
    ./scripts/run_siliconflow.sh --task "$task" --workers "$workers" \
    --in-dir .runs/dev/full --out-dir "$run_dir/answers" \
    --work-dir "$run_dir/work" --cache-dir .runs/qwen35/pilot/cache

printf '比赛文件已保存，正在校验并评分\n'
set -- "$run_dir"/work/*/summary.json
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    printf '找不到唯一的运行摘要\n' >&2
    exit 1
fi
"$root/.venv/bin/python" scripts/check_answers.py \
    --task "$task" --in-dir .runs/dev/full --answers "$run_dir/answers" --gold .runs/dev/gold \
    --holdout-from .runs/dev/pilot --work-run "${1%/*}" \
    --report "$run_dir/report.json" >"$run_dir/scoring.log" 2>&1
