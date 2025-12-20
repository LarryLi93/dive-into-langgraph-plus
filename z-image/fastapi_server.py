import os
import time
import torch
import io
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles  # 导入静态文件挂载模块
from pydantic import BaseModel, Field
from diffusers import ZImagePipeline

# --- 1. 初始化配置 ---
app = FastAPI(title="Z-Image-Turbo API", description="通义 Z-Image 图像生成服务")

SAVE_DIR = "zimage-gem"
os.makedirs(SAVE_DIR, exist_ok=True)

# 【核心修改】：挂载静态资源目录
# 这样访问 http://IP:8888/images/xxx.png 就能直接看到图片
app.mount("/images", StaticFiles(directory=SAVE_DIR), name="images")

# 模型全局变量
pipe = None

# --- 2. 定义请求参数模型 ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="生成提示词")
    width: int = Field(1440, ge=256, le=2048, description="宽度")
    height: int = Field(1920, ge=256, le=2048, description="高度")
    steps: int = Field(9, ge=1, le=50, description="生成步数/精度")
    seed: int = Field(42, description="随机种子")

# --- 3. 生命周期管理 (启动时加载模型) ---
@app.on_event("startup")
def load_model():
    global pipe
    model_path = "/root/.cache/modelscope/hub/models/Tongyi-MAI/Z-Image-Turbo"
    print(f"正在加载模型: {model_path}...")
    try:
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")

# --- 4. 核心 API 接口 ---
@app.post("/generate")
async def generate_image(request: GenerateRequest, http_request: Request):
    """
    生成图片并返回图片的访问 URL 和保存路径
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型未加载完成")

    try:
        # 执行推理
        # Z-Image-Turbo 特性：guidance_scale 固定为 0.0
        image = pipe(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(request.seed),
        ).images[0]

        # 保存到本地 zimage-gem 目录
        timestamp = int(time.time())
        file_name = f"api_{timestamp}.png"
        file_path = os.path.join(SAVE_DIR, file_name)
        image.save(file_path)

        # 动态构建图片的访问 URL
        base_url = str(http_request.base_url)  # 例如 http://127.0.0.1:8888/
        image_url = f"{base_url}images/{file_name}"

        # 返回 JSON 信息，包含图片链接
        return {
            "status": "success",
            "file_name": file_name,
            "url": image_url,
            "params": {
                "prompt": request.prompt,
                "size": f"{request.width}x{request.height}",
                "steps": request.steps,
                "seed": request.seed
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

# --- 5. 启动入口 ---
if __name__ == "__main__":
    import uvicorn
    # 监听 0.0.0.0 支持远程访问，端口 8888
    uvicorn.run(app, host="0.0.0.0", port=8888)