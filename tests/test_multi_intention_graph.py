import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from typing import Literal

_ = load_dotenv()

# 配置大模型服务
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
    temperature=0.7,
)

# 定义工具函数
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息"""
    return f"{city}的天气是晴天，温度25°C，适合出行！"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式的结果"""
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# 创建工具节点
weather_tools = [get_weather]
math_tools = [calculate]
weather_tool_node = ToolNode(weather_tools)
math_tool_node = ToolNode(math_tools)

# 分类节点：使用LLM对问题进行意图分类
def classify_intention(state: MessagesState, config: RunnableConfig):
    """使用LLM对用户问题进行意图分类"""
    system_prompt = """你是一个意图分类助手。请根据用户的问题，判断其意图类型。

    可选的意图类型包括：
    1. weather - 天气相关的问题（如查询天气、温度等）
    2. math - 数学计算相关的问题（如计算、数学运算等）
    3. chat - 通用聊天对话（其他所有问题）

    请只返回一个单词：weather、math 或 chat，不要返回其他内容。"""

    last_message = state["messages"][-1]
    classification_prompt = f"""用户问题：{last_message.content}

    请判断意图类型（只返回一个单词：weather、math 或 chat）："""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=classification_prompt)
    ]
    
    response = llm.invoke(messages)
    intention = response.content.strip().lower()
    
    # 确保返回的是有效的意图类型
    if intention not in ["weather", "math", "chat"]:
        intention = "chat"  # 默认为聊天
    
    # 将分类结果添加到消息中
    classification_msg = HumanMessage(
        content=f"[系统分类结果：{intention}]"
    )
    
    return {"messages": state["messages"] + [classification_msg]}

# 天气处理节点
def weather_handler(state: MessagesState, config: RunnableConfig):
    """处理天气相关的问题"""
    system_prompt = "你是一个天气助手，可以帮助用户查询天气信息。"
    all_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    model = llm.bind_tools(weather_tools)
    return {"messages": [model.invoke(all_messages)]}

# 数学处理节点
def math_handler(state: MessagesState, config: RunnableConfig):
    """处理数学计算相关的问题"""
    system_prompt = "你是一个数学计算助手，可以帮助用户进行数学计算。"
    all_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    model = llm.bind_tools(math_tools)
    return {"messages": [model.invoke(all_messages)]}

# 通用聊天节点
def chat_handler(state: MessagesState, config: RunnableConfig):
    """处理通用聊天对话"""
    system_prompt = "你是一个友好的助手，可以回答各种问题并进行对话。"
    all_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    return {"messages": [llm.invoke(all_messages)]}

# 路由函数：根据分类结果决定路由
def route_by_intention(state: MessagesState, config: RunnableConfig) -> Literal["weather", "math", "chat"]:
    """根据分类结果路由到不同的处理节点"""
    # 从最后一条消息中提取分类结果
    last_message = state["messages"][-1].content
    
    if "[系统分类结果：weather]" in last_message:
        return "weather"
    elif "[系统分类结果：math]" in last_message:
        return "math"
    else:
        return "chat"

# 判断是否需要调用工具（天气）
def should_use_weather_tool(state: MessagesState, config: RunnableConfig):
    """判断天气处理是否需要调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "use_tool"
    return "end"

# 判断是否需要调用工具（数学）
def should_use_math_tool(state: MessagesState, config: RunnableConfig):
    """判断数学处理是否需要调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "use_tool"
    return "end"

# 构建多意图状态图
def build_multi_intention_graph():
    """构建多意图分类处理图"""
    builder = StateGraph(MessagesState)
    
    # 添加节点
    builder.add_node("classify", classify_intention)  # 分类节点
    builder.add_node("weather_handler", weather_handler)  # 天气处理节点
    builder.add_node("math_handler", math_handler)  # 数学处理节点
    builder.add_node("chat_handler", chat_handler)  # 聊天处理节点
    builder.add_node("weather_tool", weather_tool_node)  # 天气工具节点
    builder.add_node("math_tool", math_tool_node)  # 数学工具节点
    
    # 添加边：从START到分类节点
    builder.add_edge(START, "classify")
    
    # 添加条件边：根据分类结果路由
    builder.add_conditional_edges(
        "classify",
        route_by_intention,
        {
            "weather": "weather_handler",
            "math": "math_handler",
            "chat": "chat_handler",
        },
    )
    
    # 天气处理流程：判断是否需要调用工具
    builder.add_conditional_edges(
        "weather_handler",
        should_use_weather_tool,
        {
            "use_tool": "weather_tool",
            "end": END,
        },
    )
    builder.add_edge("weather_tool", "weather_handler")
    
    # 数学处理流程：判断是否需要调用工具
    builder.add_conditional_edges(
        "math_handler",
        should_use_math_tool,
        {
            "use_tool": "math_tool",
            "end": END,
        },
    )
    builder.add_edge("math_tool", "math_handler")
    
    # 聊天处理直接结束
    builder.add_edge("chat_handler", END)
    
    return builder.compile(name="multi-intention-graph")

# 运行多意图图
def demo(query: str = "北京今天天气怎么样？"):
    """运行多意图分类处理示例"""
    graph = build_multi_intention_graph()
    
    # 生成 ASCII 图（终端可直接看）
    print(graph.get_graph().draw_ascii())

    # 导出成图片文件
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("./img/test_multi_intention_graph.png", "wb") as f:
        f.write(png_bytes)
    print("已输出 ./img/test_multi_intention_graph.png")
    
    # 运行状态图
    print(f"\n用户问题：{query}\n")
    response = graph.invoke({"messages": [HumanMessage(content=query)]})
    
    print("处理结果：")
    print("-" * 60)
    for msg in response["messages"]:
        if hasattr(msg, 'content') and msg.content:
            print(f"[{msg.__class__.__name__}] {msg.content}")
    print("-" * 60)
    
    return response["messages"]


if __name__ == "__main__":
    # 测试不同的问题类型
    test_queries = [
        # "北京今天天气怎么样？",
        "帮我计算一下 123 + 456 等于多少？",
        # "你好，介绍一下你自己",
    ]
    
    for query in test_queries:
        print("\n" + "=" * 60)
        demo(query)
        print("\n")

