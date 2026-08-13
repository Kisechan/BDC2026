FROM python:3.12-slim-bookworm

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib C library
# Source: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# 仅打包本次提交的复现闭包：运行代码、最终 v1.22 工件、其冻结 v1.20.1
# Transformer/Scaler 来源，以及对应的五年赛事行情和行业历史快照。历史试验、日志和
# 开发用数据均不进入镜像，避免无关文件影响镜像体积或人工复核。
COPY code ./code
COPY data_5y ./data_5y
COPY model/60_158+39_reduced25_relmarket12_risk15_indresid12_v1.22_submission_2026-07-31 \
     ./model/60_158+39_reduced25_relmarket12_risk15_indresid12_v1.22_submission_2026-07-31
COPY model/60_158+39_reduced25_relmarket12_risk15_indresid12_threefold_recent504_rankxendcg_officialraw_v20_1_candidate \
     ./model/60_158+39_reduced25_relmarket12_risk15_indresid12_threefold_recent504_rankxendcg_officialraw_v20_1_candidate
COPY init.sh train.sh test.sh README.md ./

RUN chmod 755 /app/init.sh /app/train.sh /app/test.sh && \
    mkdir -p /app/data /app/output /app/temp

# Set environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# Keep container running idle; execute train/predict manually via docker exec.
CMD ["sleep", "infinity"]
