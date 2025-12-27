from smolagents import CodeAgent, OpenAIServerModel

# ==========================================
# 1. 配置模型：使用硅基流动 (SiliconFlow)
# ==========================================
model = OpenAIServerModel(
    # 你提供的模型 ID
    model_id="deepseek-ai/DeepSeek-V3.1-Terminus",

    # 硅基流动的 API 地址
    api_base="https://api.siliconflow.cn/v1",

    # 你提供的 API Key
    api_key="sk-kodzewuwqkxlypmgegdjdgvhwntqfegmcamipvcoylribmss",

    # 稍微调高 max_tokens 以防输出截断
    max_tokens=256000
)

# ==========================================
# 2. 创建 Agent
# ==========================================
# add_base_tools=False 意味着不加载联网搜索工具，
# 这样在国内网络环境下运行最稳定，不会报连接超时错误。
agent = CodeAgent(
    tools=[],
    model=model,
    add_base_tools=True
)

# ==========================================
# 3. 定义任务
# ==========================================
# 这是一个逻辑题，测试模型写代码的能力
task = """
这周人民日报最重要热点是什么 ？
"""

print(f"🤖 正在调用模型 [{model.model_id}] 编写代码...\n")

try:
    # 运行 Agent
    result = agent.run(task)
    print(f"\n✅ 最终答案: {result}")

except Exception as e:
    # 如果模型 ID 写错了或者服务器报错，这里会捕获
    print(f"\n❌ 发生错误: {e}")
    print(
        "提示：如果报错 'model not found'，请检查 model_id 是否拼写正确，或者尝试换成 'Qwen/Qwen2.5-Coder-32B-Instruct' 试试。")
