#!/usr/bin/env bash
# Docker 复现入口：仅准备挂载目录，绝不下载数据、模型或依赖。
set -euo pipefail

mkdir -p /app/output /app/temp
echo '初始化完成：将使用镜像内依赖与模型，运行期间不联网。'
