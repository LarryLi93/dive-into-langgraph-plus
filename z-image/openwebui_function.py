"""
title: Z_Image_Generator_Direct
author: larry li
des: 调用 Z-Image FastAPI 生成图片,基于Filter过滤器
version: 0.3
"""

import requests
import time
from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable


class Filter:
    class Valves(BaseModel):
        api_url: str = Field(
            default="http://139.196.198.169:8888/generate",
            description="Z-Image FastAPI 的生成接口地址",
        )
        trigger_word: str = Field(default="画图", description="触发词")

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        messages = body.get("messages", [])
        if not messages:
            return body

        last_message_content = messages[-1]["content"].strip()

        if last_message_content.startswith(self.valves.trigger_word):
            prompt = last_message_content.replace(
                self.valves.trigger_word, "", 1
            ).strip()

            if not prompt:
                return body

            # 1. 发送“正在生成”的状态到 UI
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"正在为您绘图: {prompt}", "done": False},
                }
            )

            try:
                # 2. 调用 API
                response = requests.post(
                    self.valves.api_url,
                    json={
                        "prompt": prompt,
                        "width": 1440,
                        "height": 1920,
                        "steps": 9,
                        "seed": int(time.time()),
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    image_url = response.json().get("url")

                    # 3. 【核心操作】直接把图片渲染到 UI 界面，不经过 AI 处理
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"🎨 **绘图完成！**\n\n![Generated Image]({image_url})\n\n"
                            },
                        }
                    )

                    # 4. 告诉 UI 状态已完成
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "绘图成功", "done": True},
                        }
                    )

                    # 5. 修改给 AI 的指令，让 AI 针对这张图说句赞美的话，而不是重复生成
                    messages[-1][
                        "content"
                    ] = f"我已经生成了这张图片：{prompt}。请你用很简短的一句话赞美一下这张画，不要再尝试生成或回复 Markdown 链接。"

                else:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "生成失败", "done": True},
                        }
                    )

            except Exception as e:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"错误: {e}", "done": True},
                    }
                )

        return body
