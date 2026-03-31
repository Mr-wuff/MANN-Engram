# 使用包含 PyTorch 环境的 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装基础系统依赖 (处理 PDF/DOCX 等文档时部分 Python 库需要 C 编译器)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 第一步：复制项目根目录下的 requirements.txt 和 setup.py 并安装 SDK 依赖
# 这样可以利用 Docker 的层缓存，加快构建速度
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 第二步：复制微服务专属的依赖并安装
COPY ./api_service/requirements.txt /app/api_service/
RUN pip install --no-cache-dir -r /app/api_service/requirements.txt

# 第三步：复制完整的 SDK 源码和微服务源码
COPY ./mann_engram_en /app/mann_engram_en
COPY ./setup.py /app/
COPY ./api_service /app/api_service

# 将 SDK 自身作为可编辑包安装到 Docker 环境中
RUN pip install -e .

# 暴露 FastAPI 运行的 8000 端口
EXPOSE 8000

# 启动 Uvicorn 服务器
CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]