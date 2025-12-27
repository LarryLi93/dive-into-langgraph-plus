import os
from smolagents import CodeAgent, OpenAIServerModel, tool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ==========================================
# 1. 配置信息
# ==========================================
SILICON_API_KEY = "sk-kodzewuwqkxlypmgegdjdgvhwntqfegmcamipvcoylribmss"
SILICON_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL_ID = "Pro/zai-org/GLM-4.7"
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
VECTOR_SIZE = 4096  # BGE-M3 维度

# Qdrant 配置
QDRANT_HOST = "120.24.168.78"
QDRANT_PORT = 7021
COLLECTION_NAME = "demo_1227"  # 集合名称

# 初始化模型对象
model = OpenAIServerModel(
    model_id=LLM_MODEL_ID,
    api_base=SILICON_BASE_URL,
    api_key=SILICON_API_KEY,
    max_tokens=4096,
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_ID,
    openai_api_key=SILICON_API_KEY,
    openai_api_base=SILICON_BASE_URL,
    check_embedding_ctx_length=False
)

# ==========================================
# 2. 智能数据库连接 (核心优化逻辑)
# ==========================================
print(f"正在连接 Qdrant ({QDRANT_HOST}:{QDRANT_PORT})...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# 检查集合是否存在
if client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"✅ 发现已存在的集合 '{COLLECTION_NAME}'。")
    print("   -> 跳过文件读取和向量化步骤，直接使用现有数据。")

    # 直接初始化 Store，不做 add_documents 操作
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

else:
    print(f"❌ 集合 '{COLLECTION_NAME}' 不存在，准备初始化...")
    print("   -> 正在读取文件并构建向量索引（仅首次运行需要）...")

    # 1. 创建集合
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # 2. 初始化 Store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # 3. 读取文件 & 切分
    FILE_PATH = "company_qa.txt"
    try:
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            full_text = f.read()


        # 简单切分函数
        def simple_chunk_text(text, chunk_size=300, overlap=50):
            chunks = []
            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunks.append(text[start:end])
                if end == text_len: break
                start = end - overlap
            return chunks


        text_chunks = simple_chunk_text(full_text)
        documents = [Document(page_content=chunk) for chunk in text_chunks]

        # 4. 上传数据 (这一步会消耗 Embedding Token)
        if documents:
            print(f"   -> 正在上传 {len(documents)} 条数据到远程数据库...")
            vector_store.add_documents(documents)
            print("   -> ✅ 数据构建完成！")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {FILE_PATH}，无法构建知识库。")
        exit(1)


# ==========================================
# 3. 定义检索工具
# ==========================================
@tool
def search_vector_db(query: str) -> str:
    """
    这是一个基于向量语义的知识库搜索工具。

    Args:
        query: 用户的搜索问题。
    """
    print(f"\n>>> [工具调用] 检索中: {query}")

    # 搜索 Top 3
    results = vector_store.similarity_search(query, k=3)

    if not results:
        return "知识库中未找到相关信息。"

    return "\n---\n".join([f"片段 {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)])


# ==========================================
# 4. 运行 Agent
# ==========================================
agent = CodeAgent(
    tools=[search_vector_db],
    model=model,
    add_base_tools=True
)

if __name__ == "__main__":
    # 你可以修改这里的问题，此时不会再触发文件读取，速度会非常快
    question = "出差补贴补贴有什么"

    print(f"User: {question}")
    print("-" * 50)
    try:
        agent.run(question)
    except Exception as e:
        print(f"运行出错: {e}")
