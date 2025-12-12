"""
本示例演示如何将 `files` 目录下的 txt 文件按空行分段，写入内存向量库，
并通过 ReAct Agent 检索回答。

注意：若是Excel，需额外写python脚本，将列表头转为以下格式（空行分割），问答效果更准。
------
问题：
答案：
------

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool


# 加载模型配置
_ = load_dotenv()


# 配置大模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
    temperature=0,
)

# 创建 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)


class DashScopeEmbeddings(Embeddings):
    """DashScope 兼容的 Embeddings 封装。"""

    def __init__(self, model: str = "text-embedding-v4", dimensions: int = 1024):
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 10):
            chunk = texts[i : i + 10]
            response = client.embeddings.create(
                model=self.model,
                input=chunk,
                dimensions=self.dimensions,
            )
            vectors.extend([item.embedding for item in response.data])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=self.dimensions,
        )
        return response.data[0].embedding


def load_txt_documents(data_dir: Path) -> list[Document]:
    """读取目录下的 txt 文件并按空行分割为 Document。"""

    def split_on_blank(text: str) -> Iterable[str]:
        for block in re.split(r"\n\s*\n", text):
            cleaned = block.strip()
            if cleaned:
                yield cleaned

    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        for idx, part in enumerate(split_on_blank(content)):
            documents.append(
                Document(
                    page_content=part,
                    metadata={"source": path.name, "chunk_id": idx},
                )
            )
    if not documents:
        raise ValueError(f"目录 {data_dir} 下未找到 txt 文档")
    return documents


def build_vector_store(data_dir: Path | None = None) -> InMemoryVectorStore:
    """读取 txt 文件并构建内存向量库。"""
    # 默认指向仓库根目录下的 files，而非 tests/files
    target_dir = data_dir or (Path(__file__).parent.parent / "files")
    documents = load_txt_documents(target_dir)

    print(f"成功加载 {len(documents)} 个文档到向量库")

    embeddings = DashScopeEmbeddings()
    vector_store = InMemoryVectorStore(embedding=embeddings)
    _ = vector_store.add_documents(documents)
    
    return vector_store


def create_react_agent(vector_store: InMemoryVectorStore):
    """基于给定向量库创建带检索工具的 ReAct Agent。"""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """基于向量库检索与问题最相关的文本片段。"""
        retrieved = vector_store.similarity_search(query, k=3)
        serialized = "\n\n".join(
            f"[{doc.metadata['source']}#{doc.metadata['chunk_id']}] {doc.page_content}"
            for doc in retrieved
        )
        return serialized, retrieved

    return create_agent(
        llm,
        tools=[retrieve_context],
        system_prompt=(
            "你可以使用检索工具获得参考资料。回答时结合检索到的内容，"
            "如有必要可以在答案中简单引用来源标识。"
        ),
    )


def run_demo():
    """简单演示：针对 txt 知识库发起提问。"""
    query = "考勤缺卡怎么处理？"

    # 嵌入向量数据库
    vector_store = build_vector_store()

    print('嵌入完成' + '\n')

    # 检索向量数据库
    agent = create_react_agent(vector_store)
    for event in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    run_demo()

