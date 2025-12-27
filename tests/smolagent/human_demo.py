import os
import requests
import random
from urllib.parse import quote
from smolagents import CodeAgent, OpenAIServerModel, tool

# ==========================================
# 0. 全局配置
# ==========================================
SILICON_API_KEY = "sk-kodzewuwqkxlypmgegdjdgvhwntqfegmcamipvcoylribmss"
SILICON_BASE_URL = "https://api.siliconflow.cn/v1"

model = OpenAIServerModel(
    model_id="Pro/zai-org/GLM-4.7",
    api_base=SILICON_BASE_URL,
    api_key=SILICON_API_KEY,
    max_tokens=200000,
)


# ==========================================
# 1. 定义工具 (修复了 docstring 格式)
# ==========================================

@tool
def get_weather(location: str) -> str:
    """
    查询指定城市的天气状况。

    Args:
        location: 要查询天气的城市名称（例如 'Beijing', 'Shanghai'）。
    """
    try:
        url = f"https://wttr.in/{quote(location)}?format=3"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return f"查询失败: {r.status_code}"
        return r.text.strip()
    except Exception as e:
        return f"查询异常: {e}"


@tool
def book_hotel(city: str, check_in_date: str, nights: int = 1, budget: str = "标准") -> str:
    """
    预订酒店工具。
    调用此工具后，系统会暂停并请求用户在终端确认。如果用户拒绝，工具会返回错误信息，你需要根据用户的反馈重新调整参数。

    Args:
        city: 目标城市名称。
        check_in_date: 入住日期（如 '明天', '2024-05-01'）。
        nights: 入住天数，默认为 1。
        budget: 预算等级（'经济型', '标准', '豪华型'），默认为 '标准'。
    """
    # 1. 打印核对单
    print("\n" + "=" * 40)
    print("📋 [系统后台] 收到预订请求，请人工核对：")
    print(f"   📍 城市: {city}")
    print(f"   📅 日期: {check_in_date}")
    print(f"   🌙 晚数: {nights}")
    print(f"   💰 预算: {budget}")
    print("=" * 40)

    # 2. 阻塞程序，等待人类输入
    # 这里的 prompt 提示语要清楚，告诉用户可以输入 'y' 或者修改意见
    user_audit = input(">>> (y/n) 信息正确输入 'y'，错误直接输入修改意见: ")

    # 3. 根据人类反馈处理逻辑
    if user_audit.lower() in ['y', 'yes', '是', 'ok', '1']:
        order_id = f"HT-{random.randint(10000, 99999)}"
        print(f"✅ [系统] 订单已提交，单号 {order_id}")
        return f"预订成功！订单号: {order_id}。详情: {city}, {check_in_date}, {nights}晚。"
    else:
        # 确认失败，返回错误信息给 Agent
        print(f"❌ [系统] 用户驳回了请求。意见: {user_audit}")
        # 返回值非常关键，必须把用户的意见传回去，Agent 才能看到
        return f"预订失败。用户驳回了操作，并给出了修改意见：'{user_audit}'。请严格按照此意见更新参数并立即重试。"


# ==========================================
# 2. 初始化 Agent
# ==========================================

agent = CodeAgent(
    tools=[get_weather, book_hotel],
    model=model,
    add_base_tools=True
)

# ==========================================
# 3. 交互主程序
# ==========================================

if __name__ == "__main__":
    print("🤖 交互式预订助手 ")
    print("--------------------------------------------------")

    while True:
        try:
            user_input = input("\n👤 请输入指令 (q退出): ").strip()
            if user_input.lower() in ['q', 'exit']:
                break
            if not user_input:
                continue

            agent.run(user_input)

        except Exception as e:
            # 打印完整的错误栈以便调试
            import traceback

            traceback.print_exc()
