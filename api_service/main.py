import os
import tempfile
import shutil
import traceback
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List

# 导入我们在上一阶段编写的核心 SDK
from mann_engram_en.router import MANNEngramRouter

# ==========================================
# 1. API 基础配置与实例化
# ==========================================
app = FastAPI(
    title="MANN-Engram API",
    description="Edge-Cloud Multimodal Semantic Router Microservice. Filters noise from clinical data.",
    version="0.1.0"
)

# 全局路由引擎占位符
router_engine = None

@app.on_event("startup")
async def startup_event():
    """在服务启动时，预热并加载本地张量模型和意图模型"""
    global router_engine
    print("[API] Starting MANN-Engram Engines...")
    try:
        # 注意：这里假设您的权重文件挂载在项目根目录的 weights/ 文件夹下
        router_engine = MANNEngramRouter(
            ckpt_path="./weights/skew_model_v4full_en.pt",
            enable_local_intent=True, # 微服务默认开启纯本地化意图提取，确保数据安全
            local_intent_model="Qwen/Qwen2.5-0.5B-Instruct"
        )
        print("[API] Engines warmed up successfully!")
    except Exception as e:
        print(f"[API Error] Failed to load engines: {e}")

# ==========================================
# 2. API 路由端点
# ==========================================
@app.get("/health")
def health_check():
    """健康检查端点，用于 K8s/Docker 探针"""
    if router_engine is None:
        raise HTTPException(status_code=503, detail="Routing engine is not initialized.")
    return {"status": "ok", "message": "MANN-Engram Microservice is running and ready."}

@app.post("/api/v1/purify")
async def purify_data(
    query: str = Form(..., description="用户的原始脏输入（包含抱怨、背景和真实问题）"),
    top_p: float = Form(0.85, description="Top-P 截断阈值，越小过滤越严格"),
    files: List[UploadFile] = File(default=[], description="上传的文档附件 (PDF, DOCX, TXT 等)"),
    images: List[UploadFile] = File(default=[], description="上传的图像附件 (JPG, PNG 等)")
):
    """
    核心端点：接收多模态大杂烩，返回提纯后的 Core Query, 净化文本和有效图片 Base64。
    """
    if router_engine is None:
        raise HTTPException(status_code=503, detail="Service is starting up, please try again later.")

    # 创建一个临时目录，用于存放通过 HTTP 上传的文件
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    image_paths = []

    try:
        # 1. 保存普通文档到临时目录
        for f in files:
            if f.filename:
                path = os.path.join(temp_dir, f.filename)
                with open(path, "wb") as buffer:
                    shutil.copyfileobj(f.file, buffer)
                file_paths.append(path)

        # 2. 保存图片到临时目录
        for img in images:
            if img.filename:
                path = os.path.join(temp_dir, img.filename)
                with open(path, "wb") as buffer:
                    shutil.copyfileobj(img.file, buffer)
                image_paths.append(path)

        # 3. 呼叫 SDK 核心逻辑进行处理
        results = router_engine.process_session(
            raw_chat_input=query,
            file_paths=file_paths,
            image_paths=image_paths,
            openai_client=None, # 微服务强制走本地意图模型
            top_p=top_p
        )

        # 4. 为了 JSON 序列化，我们需要将返回结果中的 PIL Image 对象剔除
        # (因为 SDK 返回了 Base64 已经足够 API 消费者使用了)
        if "purified_images_pil" in results:
            del results["purified_images_pil"]

        return JSONResponse(content={"status": "success", "data": results})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # 5. 无论成功与否，务必清理临时文件夹，防止服务器磁盘撑爆
        shutil.rmtree(temp_dir, ignore_errors=True)