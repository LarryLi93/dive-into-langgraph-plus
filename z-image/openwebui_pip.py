"""
title: Z-Image Standalone Model (Async)
author: open-webui
author_url: https://github.com/open-webui
dec: Z-Image 模型的异步管道实现，用于在 OpenWebUI 中调用 Z-Image FastAPI 生成图片。
version: 0.6
"""

import requests
import time
import asyncio
from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator, Callable, Awaitable


class Pipe:
    class Valves(BaseModel):
        api_url: str = Field(
            default="http://139.196.198.169:8888/generate",
            description="Z-Image FastAPI 的生成接口地址",
        )
        width: int = Field(default=1440, description="图片宽度")
        height: int = Field(default=1920, description="图片高度")
        steps: int = Field(default=9, description="生成步数")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        return [{"id": "z_image_turbo", "name": "Z-Image Turbo (免费生图)"}]

    # 注意：这里改成了 async def
    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        # 1. 获取用户提示词
        messages = body.get("messages", [])
        if not messages:
            return "没有收到提示词。"

        prompt = messages[-1]["content"].strip()
        if not prompt:
            return "请输入绘图提示词。"

        # 2. 推送状态通知（使用 await 代替 asyncio.run）
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"正在生成图片: {prompt[:20]}...",
                        "done": False,
                    },
                }
            )

        try:
            # 3. 构造请求参数
            payload = {
                "prompt": prompt,
                "width": self.valves.width,
                "height": self.valves.height,
                "steps": self.valves.steps,
                "seed": int(time.time()),
            }

            # 4. 执行 API 调用
            # 虽然 requests 是阻塞的，但在 async def 中简单使用通常没问题
            # 如果想更“专业”，可以使用 httpx.AsyncClient()
            response = requests.post(self.valves.api_url, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                image_url = result.get("url")

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "绘图成功！", "done": True},
                        }
                    )

                # 5. 返回 Markdown 结果
                return f"![Generated Image]({image_url})\n\n**提示词:** {prompt}\n**分辨率:** {self.valves.width}x{self.valves.height}"
            else:
                return f"❌ 接口失败: {response.status_code}\n{response.text}"

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"错误: {str(e)}", "done": True},
                    }
                )
            return f"⚠️ 发生错误: {str(e)}"
