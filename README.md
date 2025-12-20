<div align="center">
    <img src="./img/social-preview.jpg" width="100%">
    <h1>Dive Into LangGraph Plus</h1>
    <h3>Larry Li</h3>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/LarryLi93/dive-into-langgraph-plus?style=flat&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/badge/language-Chinese-brightgreen?style=flat" alt="Language"/>
  <a href="https://github.com/LarryLi93/dive-into-langgraph-plus"><img src="https://img.shields.io/badge/GitHub-Project-blue?style=flat&logo=github" alt="GitHub Project"></a>
  <!-- <a href="https://github.com/luochang212/langgraph-tutorial/actions/workflows/deploy-book.yml"><img src="https://github.com/luochang212/langgraph-tutorial/actions/workflows/deploy-book.yml/badge.svg?branch=main" alt="deploy-book"/></a> -->
</div>

<div align="center">

中文 | [English](./docs/README-en.md)

</div>

<div align="center">
  <p><a href="https://larryli93.github.io/dive-into-langgraph-plus/">📚 在线阅读地址</a></p>
  <h3>📖《LangGraph 1.0 完全指南 Plus版》</h3>
  <p><em>从零开始，动手实现强大的智能体</em></p>
  <p><em>本教程基于Dive Langgraph和官网，进行了更全面的梳理/整合，并加入多个开发案例(Cases)、以及商业解决方案(Solutions)，可作为入门+进阶教程使用。</em></p>
</div>

---

## 一、项目介绍

> 2025 年 10 月中旬，LangGraph 发布 1.0 版本。开发团队承诺这是一个稳定版本，预计未来接口不会大改，因此现在正是学习它的好时机。

这是一个开源教程项目，旨在帮助 Agent 开发者快速掌握 LangGraph 框架。[LangGraph](https://github.com/langchain-ai/langgraph) 是由 LangChain 团队开发的开源智能体框架。它功能强大，你要的记忆、MCP、护栏、状态管理、多智能体它全都有。LangGraph 通常与 [LangChain](https://github.com/langchain-ai/langchain) 一起使用：LangChain 提供基础组件和工具，LangGraph 负责工作流和状态管理。因此，两个库都需要学习。为了让大家快速入门，本教程将两个库的主要功能提取出来，分成 14 个章节进行介绍。


> \[!NOTE\]
> 
> **前提**：本项目默认你已完成AI核心知识的学习，若没有。
> 请移步到项目[《AI Core Knowledge》](https://github.com/LarryLi93/AI-Core-Knowledge)进行学习。


## 二、安装依赖

```bash
pip install -r requirements.txt
```

<details>
  <summary>依赖包列表</summary>

  以下为 `requirements.txt` 中的依赖包清单：

  ```text
  pydantic
  python-dotenv
  langchain[openai]
  langchain-community
  langchain-mcp-adapters
  langchain-text-splitters
  langgraph
  langgraph-cli[inmem]
  langgraph-supervisor
  langgraph-checkpoint-sqlite
  langmem
  ipynbname
  fastmcp
  bs4
  ```
</details>

## 三、章节目录

本教程的内容速览：

| 序号 | 章节 | 主要内容 |
| -- | -- | -- |
| 0 | [AI框架详解](./0.introduce.ipynb) | 介绍当下主流的AI框架及其特点 |
| 1 | [快速入门](./1.quickstart.ipynb) | 创建你的第一个 ReAct Agent |
| 2 | [状态图](./2.stategraph.ipynb) | 使用 StateGraph 创建工作流 |
| 3 | [中间件](./3.middleware.ipynb) | 使用自定义中间件实现四个功能：预算控制、消息截断、敏感词过滤、PII 检测 |
| 4 | [人机交互](./4.human_in_the_loop.ipynb) | 使用内置的 HITL 中间件实现人机交互 |
| 5 | [记忆](./5.memory.ipynb) | 创建短期记忆、长期记忆 |
| 6 | [上下文工程](./6.context.ipynb) | 使用 State、Store、Runtime 管理上下文 |
| 7 | [MCP Server](./7.mcp_server.ipynb) | 创建 MCP Server 并接入 LangGraph |
| 8 | [监督者模式](./8.supervisor.ipynb) | 两种方法实现监督者模式：tool-calling、langgraph-supervisor |
| 9 | [并行](./9.parallel.ipynb) | 如何实现并行：节点并行、Map-reduce |
| 10 | [Deep Agents](./10.deep_agents.ipynb) | 简单介绍 Deep Agents |
| 11 | [调试页面](./11.langgraph_cli.ipynb) | 介绍 langgraph-cli 提供的调试页面 |
| 12 | [开发案例](./12.development_case.ipynb) | 介绍 常见的核心开发案例 |
| 13 | [商业案例](./13.business_case.ipynb) | 介绍 常见的商业案例 |

## 四、开发案例

| 开发案例 | 功能简介 | 核心内容 |
| :--- | :--- | :--- |
| [多意图智能体 (Multi Intention Agent)](./12.1.multi_intention_graph.ipynb) | 根据大模型对用户问题进行意图识别与分类，自动选择对应的处理流程并生成相应输出。 | 意图路由 (route_by_intention) |

| [知识智能体(内存版) (Agent RAG)](./12.2.agent_rag.ipynb) | 将知识文档嵌入内存向量数据库，构建RAG系统，并通过智能智能体实现检索与问答。 | 百练嵌入模型(DashScopeEmbeddings) <br />内存向量数据库(InMemoryVectorStore)<br />相似向量检索(similarity_search) |

| [知识智能体(权限过滤版) (Auth Agent RAG)](./12.3.auth_agent_rag.ipynb) | 将知识文档嵌入内存向量数据库，将对应权限写入元数据，构建RAG系统，并通过智能智能体实现检索与权限过滤的问答。 | 自定义过滤(filter_func)<br />元数据(metadata) |
| [知识智能体(向量数据库版) (DB Agent RAG)](./12.4.agent_rag_db.ipynb) | 基于12.3实现的将知识文档嵌入远程向量数据库，构建RAG系统，并通过智能智能体实现检索与权限过滤的问答。 | 向量数据库(Qdrant) |

| [知识智能体(重排版) (DB Agent RAG Rerank)](./12.5.agent_rag_db_rerank.ipynb) | 实现的将知识文档嵌入远程向量数据库，构建RAG智能体，并通过智能智能体实现检索重排。 | 重排模型 (BAAI/bge-reranker-v2-m3) |

| [检索增强知识库智能体 (Enhance Agent RAG)](./12.6.enhance_agent_rag.ipynb) | 集成常用检索优化策略的高阶RAG智能体，通过多路径检索、阈值过滤、重排序等技术，提升回答的准确性与可靠性。 | 检索优化策略 (enhance) |
| [高级知识库智能体 (Senior Agentic RAG)](./12.7.senior_agent_rag.ipynb) | 智能体引入自主决策、编排、推理，实现了更强大和灵活的检索。 | Agentic RAG (route_by_intention) |

| [多模态文档预处理 (Multi Modal Preprocessing)](./12.8.multi_modal_preprocessing.ipynb) | 支持对多种格式文档（如PDF、Markdown、TXT、Excel、Image、Video）的预处理，包括文本提取、图像识别、视频分析等。 | 多模态文档处理 |
| [知识图谱智能体(Neo4j版) (Graph RAG Agent)](./12.9.graph_agent_rag.ipynb) | 将知识文档嵌入知识图谱，构建基于图结构的RAG系统，并通过智能智能体实现检索与问答。 | 知识图谱 (rag) |

| [知识图谱智能体(Graphiti+Neo4j版) (AI Graph RAG Agent)](./12.10.ai_graph_agent_rag.ipynb) | 将知识文档嵌入知识图谱，构建基于图结构的RAG系统，并通过智能智能体实现检索与问答。 | 知识图谱 (rag) |

| [深度研究智能体 (DeepAgents)](./12.11.deep_agent.ipynb) | 将知识文档嵌入知识图谱，构建基于图结构的RAG系统，并通过智能智能体实现深度研究与问答。 | 多智能体协作<br />子智能体分工<br />文件系统工具集成<br />网络搜索<br />研究计划自动生成<br />结构化研究报告输出 |

| [Text2Image-MCP (Z-Image文生图)](./12.12.text2image.ipynb) | 基于Z-Image模型，将文本描述转换为图像，封装为MCP工具（给Agent调用）。 | 文本到图像 (text2image) |

| [代码执行器-MCP (Code Execution)](./12.13.code_execution.ipynb) | 基于MCP工具，执行用户输入的Python代码，并返回运算结果。 | 代码执行 (code_execution) |

未出现在上述章节但比较重要的代码，我放在仓库的 tests 目录下：

|代码|说明|
| -- | -- |
| [/tests/test_rag.py](./tests/test_rag.py) | 使用 `RAG` 将本地文档片段注入智能体 |
| [/tests/test_langmem.py](./tests/test_langmem.py) | 使用 `LangMem` 管理智能体长期记忆 |
| [/tests/test_store.py](./tests/test_store.py) | 使用 `RedisStore` 快速读写长期记忆 |
| [/tests/test_router.py](./tests/test_router.py) | 实现一个简单的智能体路由 |


> \[!NOTE\]
> 
> **承诺**：本教程完全基于 LangGraph v1 编写，不含任何 v0.6 的历史残留。

## 四、调试页面

`langgraph-cli` 提供了一个可快速启动的调试页面。

```bash
langgraph dev
```

详见 [第11章](./11.langgraph_cli.ipynb)

## 五、延伸阅读

**官方文档：**

- [LangChain](https://docs.langchain.com/oss/python/langchain/overview)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
- [LangMem](https://langchain-ai.github.io/langmem/)

**官方教程：**

- [langgraph-101](https://github.com/langchain-ai/langgraph-101)
- [langchain-academy](https://github.com/langchain-ai/langchain-academy)

## 六、如何贡献

我们欢迎任何形式的贡献！

- 🐛 报告 Bug - 发现问题请提交 Issue
- 💡 功能建议 - 有好想法就告诉我们
- 📝 内容完善 - 帮助改进教程内容
- 🔧 代码优化 - 提交 Pull Request

## 七、开源协议

本作品采用 [知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-sa/4.0/) 进行许可。
