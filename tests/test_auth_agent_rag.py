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
from typing import Iterable, Optional

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
    """读取目录下的 txt 文件，按空行分割，并提取'权限'字段到 metadata。"""

    def split_on_blank(text: str) -> Iterable[str]:
        for block in re.split(r"\n\s*\n", text):
            cleaned = block.strip()
            if cleaned:
                yield cleaned

    def parse_metadata(text_block: str) -> tuple[str, dict]:
        """从文本块中解析权限等元数据"""
        lines = text_block.split('\n')
        content_lines = []
        metadata = {}
        
        for line in lines:
            line = line.strip()
            # 提取权限字段
            if line.startswith("权限:"):
                metadata["permission"] = line.replace("权限:", "").strip()
            # 提取关键词字段（可选，放入metadata方便后续使用）
            elif line.startswith("关键词:"):
                metadata["keywords"] = line.replace("关键词:", "").strip()
            else:
                content_lines.append(line)
        
        # 重新组合剩余的文本作为正文内容
        return "\n".join(content_lines), metadata

    documents: list[Document] = []

    for path in sorted(data_dir.glob("*.txt")):

        print(f"正在读取文件: {path.absolute()}") 

        content = path.read_text(encoding="utf-8")
        for idx, part in enumerate(split_on_blank(content)):
            # 关键修改：调用解析函数分离内容和元数据
            clean_content, extracted_meta = parse_metadata(part)
            
            # 合并基础元数据
            final_metadata = {
                "source": path.name, 
                "chunk_id": idx,
                **extracted_meta # 将 permission 加入 metadata
            }
            
            documents.append(
                Document(
                    page_content=clean_content,
                    metadata=final_metadata,
                )
            )
            
    if not documents:
        raise ValueError(f"目录 {data_dir} 下未找到 txt 文档")
    return documents


def build_vector_store(data_dir: Path | None = None) -> InMemoryVectorStore:

    """读取 txt 文件并构建内存向量库。"""
    target_dir = data_dir or Path(__file__).parent.parent / "files"
    
    documents = load_txt_documents(target_dir)

    print(f"成功加载 {len(documents)} 个文档到向量库")
    # 打印一下看看 metadata 长什么样
    for doc in documents:
        print(f"  - 文档片段ID: {doc.metadata['chunk_id']}, 权限: {doc.metadata.get('permission', '无')}")

    embeddings = DashScopeEmbeddings()
    vector_store = InMemoryVectorStore(embedding=embeddings)
    _ = vector_store.add_documents(documents)
    
    return vector_store


def create_react_agent(vector_store: InMemoryVectorStore, user_permission: str):
    """
    基于给定向量库创建 ReAct Agent。
    新增参数: user_permission (当前用户的权限)
    """

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """基于向量库检索与问题最相关的文本片段。"""
        
        # 关键修改：定义过滤函数
        # InMemoryVectorStore 的 filter 参数接受一个函数 (doc) -> bool
        def filter_func(doc: Document) -> bool:
            doc_perm = doc.metadata.get("permission")
            # 逻辑：如果文档没写权限，视为公开；如果写了权限，必须匹配
            # 你也可以改为：如果没写权限，也不可见
            if not doc_perm: 
                return True # 公开文档
            return doc_perm == user_permission

        print(f"\n[检索中] 用户权限: {user_permission}, 查询: {query}")
        
        # 将 filter 传入 search 方法
        retrieved = vector_store.similarity_search(
            query, 
            k=3, 
            filter=filter_func
        )
        
        serialized = "\n\n".join(
            f"[{doc.metadata['source']}#{doc.metadata['chunk_id']}] {doc.page_content}"
            for doc in retrieved
        )
        return serialized, retrieved

    return create_agent(
        llm,
        tools=[retrieve_context],
        system_prompt=(
            "你可以使用检索工具获得参考资料。然后进行精简的回答。"
        ),
    )


def run_demo():
    """演示：模拟不同权限的用户进行提问。"""
    
    # 1. 构建向量库
    vector_store = build_vector_store()
    print('\n=== 嵌入完成 ===\n')

    query = "考勤缺卡怎么处理？" # 或者 "考勤方式是什么？"

    # 场景 A: 
    print("--- 场景测试: 用户拥有 [IT组] 权限 ---")
    agent_authorized = create_react_agent(vector_store, user_permission="IT组")
    
    for event in agent_authorized.stream(
        {"messages": [{"role": "user", "content": "公司的考勤方式是什么？"}]}, 
        stream_mode="values"
    ):
        event["messages"][-1].pretty_print()

    print("\n" + "-"*30 + "\n")

    # 场景 B: 
    print("--- 场景测试: 用户拥有 [运营组] 权限 ---")
    agent_unauthorized = create_react_agent(vector_store, user_permission="运营组")
    
    for event in agent_unauthorized.stream(
        {"messages": [{"role": "user", "content": "公司的考勤方式是什么？"}]}, 
        stream_mode="values"
    ):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    run_demo()