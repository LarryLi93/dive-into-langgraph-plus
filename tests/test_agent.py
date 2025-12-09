# 创建一个简单的Agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

# 加载模型配置
_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
)

agent = create_agent(model=llm)

response = agent.invoke({"messages": "你好"})
print(response["messages"][-1].content)

# 生成 ASCII 图（终端可直接看）
print(agent.get_graph().draw_ascii())

# 导出成图片文件
png_bytes = agent.get_graph().draw_mermaid_png()
with open("./img/test_agent.png", "wb") as f:
    f.write(png_bytes)
print("已输出 ./img/test_agent.png")