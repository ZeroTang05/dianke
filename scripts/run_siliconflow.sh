#!/bin/sh
# 本地验证入口；密钥从启动进程的环境读取。
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export LLM_BASE_URL=https://api.siliconflow.cn/v1
export LLM_MODEL=Qwen/Qwen3.5-27B
export LLM_ENABLE_THINKING="${LLM_ENABLE_THINKING:-0}"
exec "$root/run.sh" "$@"
