import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.chains import RetrievalQA

# 1. 大模型与 Embedding 配置
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="text-embedding-v4"          # 通义 Embedding 模型
)

# 2. 按空行切分 txt
def load_txt_by_empty_line(path: str):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # 两个及以上换行符视为段落分隔
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

docs = load_txt_by_empty_line("../files/question.txt")

# 3. 写入 Milvus
VECTOR_DB_HOST = "120.24.168.78"
VECTOR_DB_PORT = "7042"

vector_store = Milvus.from_texts(
    texts=docs,
    embedding=embeddings,
    collection_name="paragraphs",
    connection_args={"host": VECTOR_DB_HOST, "port": VECTOR_DB_PORT},
    drop_old=True,   # 每次运行重建集合，测试用
)

# 4. 检索 + QA
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vector_store.as_retriever(search_kwargs={"k": 3})
# )

# query = "请用中文总结全文要点"
# answer = qa.run(query)
# print("---- 回答 ----")
# print(answer)