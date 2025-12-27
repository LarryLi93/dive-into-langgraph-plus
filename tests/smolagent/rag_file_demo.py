import os
from smolagents import CodeAgent, OpenAIServerModel, tool
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 移除报错的 import
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. 硅基流动 API 配置
# ==========================================
SILICON_API_KEY = "sk-kodzewuwqkxlypmgegdjdgvhwntqfegmcamipvcoylribmss"
SILICON_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL_ID = "Pro/zai-org/GLM-4.7"

model = OpenAIServerModel(
    model_id=LLM_MODEL_ID,
    api_base=SILICON_BASE_URL,
    api_key=SILICON_API_KEY,
    max_tokens=200000,
)

# ==========================================
# 2. 读取本地文件并切分 (纯 Python 实现)
# ==========================================
FILE_PATH = "1.md"

print(f"正在读取文件: {FILE_PATH} ...")

try:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()
except FileNotFoundError:
    print(f"错误: 找不到文件 {FILE_PATH}，请检查路径。")
    exit(1)


# --- 替代方案：手写一个简单的切分函数 ---
def simple_chunk_text(text, chunk_size=300, overlap=50):
    """简单的文本切分函数，替代 LangChain 的复杂依赖"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)

        # 如果已经到了末尾，停止
        if end == text_len:
            break

        # 下一段的起始位置 = 当前结束位置 - 重叠量
        start = end - overlap

    return chunks


# 使用新函数切分
chunks = simple_chunk_text(full_text, chunk_size=300, overlap=50)
print(f"文件已切分为 {len(chunks)} 个片段，正在构建 BM25 索引...")

# 转换为 LangChain Document 并构建检索器
documents = [Document(page_content=chunk) for chunk in chunks]
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 2


# ==========================================
# 3. 定义检索工具
# ==========================================
@tool
def search_local_file(query: str) -> str:
    """
    这是一个本地文件搜索工具。
    当用户询问关于本地文档的具体内容时，使用此工具进行检索。

    Args:
        query: 搜索关键词。
    """
    print(f"\n>>> [工具调用] 正在检索: {query}")
    results = bm25_retriever.invoke(query)

    if not results:
        return "本地文件中未找到相关信息。"

    return "\n---\n".join([doc.page_content for doc in results])


# ==========================================
# 4. 初始化 Agent
# ==========================================
agent = CodeAgent(
    tools=[search_local_file],
    model=model,
    add_base_tools=True
)

# ==========================================
# 5. 运行
# ==========================================
if __name__ == "__main__":
    question = "请查阅本地文件，总结一下里面的主要内容。"
    print(f"User: {question}")
    print("-" * 50)
    try:
        agent.run(question)
    except Exception as e:
        print(f"运行出错: {e}")
