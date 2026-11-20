#!/bin/sh
# 本地验证入口；地址、模型和密钥使用与评测相同的环境变量。
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export LLM_ENABLE_THINKING="${LLM_ENABLE_THINKING:-0}"
exec "$root/run.sh" "$@"
