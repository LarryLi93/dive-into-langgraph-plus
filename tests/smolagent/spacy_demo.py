import os
from smolagents import CodeAgent, OpenAIServerModel, VisitWebpageTool

# 1. 配置模型 (硅基流动)
model = OpenAIServerModel(
    model_id="deepseek-ai/DeepSeek-V3.1-Terminus",
    api_base="https://api.siliconflow.cn/v1",
    api_key="sk-kodzewuwqkxlypmgegdjdgvhwntqfegmcamipvcoylribmss", # 记得替换你的Key
    max_tokens=256000
)

# 2. 工具准备
web_tool = VisitWebpageTool()

# 3. 创建 Agent
# ⚠️ 注意：这里我把 max_steps 设为了 10，给它足够的操作空间去“跳跃”网页
agent = CodeAgent(
    tools=[web_tool],
    model=model,
    add_base_tools=False,
    max_steps=6
)

# 4. 定义一个“多跳”任务
# 假设我们访问一个国内能打开的技术博客或者新闻站
target_url = "https://www.textileworld.com/category/textile-world/breaking-news/" # 36氪科技频道

task = f"""
请完成以下连贯的操作：
1. 使用工具访问 '{target_url}'。
2. 在页面正文中找到最近10天内容的文章的标题和链接。
3. **再次使用工具**访问这些文章的链接（进入下一层页面）。
4. 读取全部的正文内容，并用 500-600 个字以内汇总总结文章在讲什么。
"""

print("🤖 Agent 正在执行多步浏览任务...")

try:
    result = agent.run(task)
    print(f"\n✅ 最终总结:\n{result}")
except Exception as e:
    print(f"❌ 执行失败: {e}")
