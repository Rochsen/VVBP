---
title: AI框架设计与选型
date: 2026-03-21
categories: [教程, 知乎]
tags: [AI, Langchain]
---

<!-- more -->

## 目录

1. [AI Agent核心问题概述](#1-ai-agent核心问题概述)
2. [LLM统一接口层](#2-llm统一接口层)
3. [工具注册与调度](#3-工具注册与调度)
4. [Context管理机制](#4-context管理机制)
5. [控制流编排](#5-控制流编排)
6. [主流框架详解](#6-主流框架详解)
7. [框架对比与选型](#7-框架对比与选型)
8. [实战案例：多文件智能问答](#8-实战案例多文件智能问答)

---

## 1. AI Agent核心问题概述

### 1.1 什么是AI Agent？

**AI Agent（智能体）** 是一种能够自主理解目标、规划任务、执行操作并根据反馈调整行为的AI系统。与简单的问答不同，Agent能够：

- 分解复杂任务为多个步骤
- 调用外部工具完成特定操作
- 保持对话上下文记忆
- 自我反思和修正错误

### 1.2 构建AI Agent的四个核心问题

构建一个AI Agent框架，需要解决以下四个核心问题：

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Agent 核心架构                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                               │
│  │ 🤖 LLM大脑   │  1. LLM统一接口 - 适配不同模型                  │
│  │ (统一接口层) │                                               │
│  └──────┬──────┘                                               │
│         │                                                       │
│  ┌──────┴──────┐     ┌─────────────┐     ┌─────────────┐      │
│  │ 🔧 双手      │     │ 🧠 记忆      │     │ 🎯 中枢      │      │
│  │ 工具注册调度 │     │ Context管理  │     │ 控制流编排   │      │
│  └─────────────┘     └─────────────┘     └─────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 四大核心问题详解

| 核心组件         | 解决的问题                                   | 类比理解        |
| ---------------- | -------------------------------------------- | --------------- |
| **LLM统一接口**  | 适配不同语言模型（OpenAI、DeepSeek、Qwen等） | 大脑 - 思考能力 |
| **工具注册调度** | 让LLM能够调用外部函数执行实际操作            | 双手 - 执行能力 |
| **Context管理**  | 管理对话历史和长期记忆                       | 记忆 - 经验存储 |
| **控制流编排**   | 协调各组件完成复杂任务流程                   | 中枢 - 协调指挥 |

---

## 2. LLM统一接口层

### 2.1 为什么需要统一接口层？

不同的LLM服务商（OpenAI、DeepSeek、阿里Qwen等）有不同的API格式和参数命名。如果直接调用，需要编写大量适配代码。**统一接口层**通过适配器模式，抹平这些差异：

> [!tip]
>
> 想象你买了一个万能充电器，不管什么手机都能充
>
> 统一接口就是AI框架的"万能充电器"

### 2.2 三大框架的LLM配置方式

#### LangChain 方式

```python
from langchain_community.chat_models import ChatTongyi

llm = ChatTongyi(
    model_name="deepseek-v3",           # 模型名称
    dashscope_api_key="your-api-key"    # API密钥
)
```

#### Qwen-Agent 方式

```python
llm_cfg = {
    'model': 'deepseek-v3',                              # 模型名称
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',  # 模型服务器地址
    'api_key': 'your-api-key',                           # API密钥
    'generate_cfg': {'top_p': 0.8}                        # 生成参数
}
```

#### LlamaIndex 方式

```python
from llama_index.llms.dashscope import DashScope

llm = DashScope(
    model="deepseek-v3",       # 模型名称
    api_key="your-api-key",   # API密钥
    temperature=0.7,           # 温度参数
)
```

### 2.3 统一接口的核心价值

```
┌────────────────────────────────────────────────────────┐
│                  统一接口层的三大价值                     │
├────────────────────────────────────────────────────────┤
│  1. 📋 统一调用方式                                      │
│     不管调用哪个模型，都用同样的方法：llm.invoke("你好")   │
│                                                        │
│  2. ⚙️ 统一参数配置                                      │
│     temperature、top_p等参数统一管理                      │
│                                                        │
│  3. 📦 统一输出格式                                      │
│     输出统一转为Message对象，方便后续处理                 │
└────────────────────────────────────────────────────────┘
```

### 2.4 Prompt管理

**System Message（系统消息）** 用于定义AI的角色和行为：

```python
# LangChain 方式
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的AI助手。"),
    MessagesPlaceholder(variable_name="history"),  # 对话历史占位
    ("human", "{input}")                           # 用户输入占位
])

# Qwen-Agent 方式
system_instruction = """你是一个乐于助人的AI助手。
在收到用户的请求后，你应该：
- 首先思考问题的关键点
- 然后调用合适的工具解决问题
你总是用中文回复用户。"""
```

**人设与任务分离**的好处：

- 便于复用：一个人设可以用于多个场景
- 易于维护：修改人设不影响业务流程
- 职责清晰：角色定义和具体指令分开管理

---

## 3. 工具注册与调度

### 3.1 为什么需要工具系统？

LLM本身只能输出文本，无法执行实际操作（如查询网络、操作文件、运行代码）。**工具系统**让LLM能够"看到"并"调用"外部函数：

```
用户：帮我检查 www.example.com 的网络连通性
                │
                ▼
         ┌─────────────┐
         │   LLM       │ 思考：我需要调用"ping_tool"
         └──────┬──────┘
                │ 调用工具
                ▼
         ┌─────────────┐
         │ ping_tool   │ 执行：检查网络连通性
         └──────┬──────┘
                │
                ▼
           "Ping www.example.com 成功：延迟 20ms"
```

### 3.2 三大框架的工具注册方式对比

#### LangChain: @tool 装饰器（最简洁）

```python
from langchain_core.tools import tool

@tool
def ping_tool(target: str) -> str:
    """检查本机到指定主机名或IP地址的网络连通性。

    参数:
        target: 目标主机名或IP地址
    返回:
        模拟的ping结果
    """
    if "unreachable" in target:
        return f"Ping {target} 失败"
    return f"Ping {target} 成功"

@tool
def dns_tool(hostname: str) -> str:
    """解析给定的主机名，获取其对应的IP地址。

    参数:
        hostname: 要解析的主机名
    返回:
        DNS解析结果
    """
    if hostname == "www.example.com":
        return f"DNS解析 {hostname} 成功：IP是93.184.216.34"
    return f"DNS解析 {hostname} 失败：找不到主机"
```

**@tool装饰器的优势：**

- 自动从docstring解析工具描述
- 自动识别参数类型
- 一行装饰器，零配置即可使用

#### Qwen-Agent: @register_tool + 类（显式定义）

```python
from qwen_agent.tools.base import BaseTool, register_tool
import json5
import urllib.parse

@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    # description 告诉LLM这个工具的功能
    description = 'AI绘画服务，输入文本描述，返回图像URL'

    # parameters 显式定义输入参数
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '期望的图像内容的详细描述',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        # params 是LLM生成的JSON字符串
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False
        )
```

#### LlamaIndex: FunctionTool类（强类型约束）

```python
from llama_index.core.tools import FunctionTool

def retrieve_documents(query: str) -> str:
    """从文档中检索相关信息"""
    response = query_engine.query(query)
    return str(response)

# 封装为FunctionTool
retrieve_tool = FunctionTool.from_defaults(fn=retrieve_documents)
```

### 3.3 工具注册原理

LLM是如何"看到"工具的？

```
┌─────────────────────────────────────────────────────────────┐
│                    工具注册到LLM的流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Python函数                                                 │
│  ┌─────────────────────┐                                   │
│  │ def ping_tool(      │                                   │
│  │   target: str       │  1. 提取函数名                      │
│  │ ) -> str:           │  ───────────────────────────────>   │
│  │   """检查网络..."""  │  "ping_tool"                       │
│  └─────────────────────┘                                   │
│         │                                                  │
│         ▼  2. 提取docstring                                 │
│  ┌─────────────────────┐                                   │
│  │ "检查本机到指定主机   │                                   │
│  │ 名或IP地址的网络..."  │  ───────────────────────────────>   │
│  └─────────────────────┘                                   │
│         │                                                  │
│         ▼  3. 提取类型注解                                   │
│  ┌─────────────────────┐                                   │
│  │ target: str         │  ───────────────────────────────>   │
│  └─────────────────────┘                                   │
│                                                             │
│         ▼                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │     转换为 JSON Schema                   │                │
│  │     {                                   │                │
│  │       "name": "ping_tool",              │                │
│  │       "description": "检查网络...",      │                │
│  │       "parameters": {                   │                │
│  │         "target": {"type": "string"}    │                │
│  │       }                                 │                │
│  │     }                                   │                │
│  └─────────────────────────────────────────┘                │
│         │                                                  │
│         ▼                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │     发送给 LLM                           │                │
│  │     LLM现在知道有这个工具可用              │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Code Interpreter（代码解释器）

Qwen-Agent的杀手锏功能：内置代码执行沙箱。

```python
# 配置工具列表，包含代码解释器
tools = ['my_image_gen', 'code_interpreter']

# code_interpreter 可以：
# - 下载文件 (requests.get)
# - 处理图像 (PIL)
# - 数据分析 (pandas)
# - 绑图展示 (matplotlib)
# - 执行失败时自动修正重试
```

| 能力     | 说明                   |
| -------- | ---------------------- |
| 代码生成 | LLM自动生成Python代码  |
| 沙箱执行 | 安全隔离环境运行代码   |
| 结果获取 | 捕获输出、图像、文件   |
| 错误修复 | 执行失败时自动修正重试 |

---

## 4. Context管理机制

### 4.1 为什么需要记忆管理？

```
┌────────────────────────────────────────────────────────────┐
│                    LLM的"失忆症"                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  第1轮对话：                                                │
│  用户：我的公司叫"ABC科技"                                  │
│  AI：好的，ABC科技，有什么可以帮您？                          │
│                                                            │
│  第2轮对话：                                                │
│  用户：我们公司想购买保险                                    │
│  AI：好的，您想了解什么类型的保险？                           │
│  ❓ AI已经忘记公司名叫"ABC科技"                              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

LLM是无状态的，每次调用都是独立的。为了让AI记住对话内容，需要**Context管理**。

### 4.2 记忆的分类

```
┌─────────────────────────────────────────────────────────────┐
│                    记忆系统架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  短期记忆 (Session Memory)                            │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │   │
│  │  │对话记录1 │→│对话记录2 │→│对话记录3 │→...           │   │
│  │  └─────────┘  └─────────┘  └─────────┘              │   │
│  │                                                      │   │
│  │  策略：滑动窗口 - 只保留最近N轮对话                    │   │
│  │  原因：Context Window有Token限制，不能无限增长          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  长期记忆 (Long-term Memory)                         │   │
│  │  ┌─────────────────────────────────────────┐        │   │
│  │  │ 📄 文档片段1  ──向量1──→  [数据库]       │        │   │
│  │  │ 📄 文档片段2  ──向量2──→  [数据库]       │        │   │
│  │  │ 📄 文档片段3  ──向量3──→  [数据库]       │        │   │
│  │  └─────────────────────────────────────────┘        │   │
│  │                                                      │   │
│  │  检索方式：相似度搜索 - 找与问题最相关的文档           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 三大框架的记忆管理

#### LangChain: RunnableWithMessageHistory

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 创建会话存储
store = {}

def get_session_history(session_id: str):
    """获取指定会话ID的历史记录"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建带记忆的对话链
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 使用时指定session_id（支持多用户并发）
config = {"configurable": {"session_id": "user_123"}}
output = conversation.invoke({"input": "Hi!"}, config=config)
```

#### Qwen-Agent: messages列表手动管理

```python
# 对话历史
messages = []

# 添加用户消息
messages.append({'role': 'user', 'content': query})

# 运行助手
for response in bot.run(messages=messages):
    pass

# 添加助手回复到历史
messages.extend(response)
```

#### LlamaIndex: Agent内置记忆

```python
# LlamaIndex的Agent内部自动管理对话历史
agent = ReActAgent.from_tools(
    tools=[retrieve_tool],
    llm=llm,
    verbose=True
)

# 直接对话，内部自动管理历史
response = await agent.run(query)
```

### 4.4 Context管理的核心策略

```
┌────────────────────────────────────────────────────────────┐
│                 有限注意力的管理策略                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. 滑动窗口策略 (Sliding Window)                          │
│     ┌──────────────────────────────────────────────────┐  │
│     │ [对话1] → [对话2] → [对话3] → [对话4] → [对话5] │  │
│     └──────────────────────────────────────────────────┘  │
│                        │                                    │
│                   保留最近3轮                                │
│                        ▼                                    │
│     ┌──────────────────────────────────────────────────┐  │
│     │ [对话3] → [对话4] → [对话5]                      │  │
│     └──────────────────────────────────────────────────┘  │
│                                                            │
│  2. Token限制控制                                           │
│     Context Window 是昂贵的资源，需要合理分配                │
│     系统提示词 + 对话历史 + 检索上下文 ≤ 最大Token数          │
│                                                            │
│  3. 重要性排序 (Relevance Scoring)                         │
│     优先保留与当前任务相关的内容                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 5. 控制流编排

### 5.1 为什么需要控制流编排？

简单的任务可以靠LLM一口气完成，但复杂任务需要拆解成多个步骤：

```
用户：帮我分析这份销售报告，找出问题和改进建议

❌ 简单方式（效果差）：
   直接让LLM分析，但它可能：
   - 遗漏重要数据
   - 分析不够深入
   - 结论缺乏依据

✅ 复杂方式（效果好）：
   1. 检索相关文档 → 了解业务背景
   2. 分析销售数据 → 找出异常波动
   3. 对比历史数据 → 识别趋势变化
   4. 生成分析报告 → 综合以上信息给出建议
```

### 5.2 四种控制流模式

#### 管道模式 (Pipeline) - 线性处理

```
输入 → 处理1 → 处理2 → 处理3 → 输出
```

适用场景：输入确定、输出确定、顺序固定的场景

#### ReAct循环模式 (Single Agent) - 思考-行动-观察

```
      ┌─────────────────┐
      │                 │
      ▼                 │
   ┌──────┐             │
   │思考   │◄───────────┘
   └──────┘
      │
      ▼
   ┌──────┐
   │行动   │──调用工具──→ 观察结果
   └──────┘
      │
      ▼
   继续思考？
      │
     是├─────────────┐
      │             │
      ▼             │
   ┌──────┐         │
   │完成   │         │
   └──────┘         │
                    │
      否 ◄──────────┘
```

适用场景：单Agent需要多工具协作完成的复杂任务

#### DAG有向无环图模式 - 接力赛

```
节点A ──┬──→ 节点C ──→ 最终结果
        └──→ 节点B ───↗
```

适用场景：有明确前后依赖的流程化任务

#### GroupChat多人对话模式 - 圆桌会议

```
Agent1 ◄──────► Agent2
    ▲               ▲
    │               │
    └────► Agent3 ◄─┘
```

适用场景：需要多角色协作讨论的开放式任务

### 5.3 LangChain的LCEL管道语法

**LCEL (LangChain Expression Language)** 是LangChain的核心创新：

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import StrOutputParser

# 创建组件
prompt = PromptTemplate(
    input_variables=['product'],
    template='What is a good name for a company that makes {product}?'
)
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key="your-key")

# 管道语法组合（用 | 符号）
chain = prompt | llm | StrOutputParser()

# 调用
result = chain.invoke({"product": "colorful socks"})
```

```
┌─────────────────────────────────────────────────────────────┐
│                  LCEL 管道执行流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户输入                                                   │
│   ┌───────────┐                                             │
│   │"colorful  │                                            │
│   │ socks"    │                                            │
│   └─────┬─────┘                                             │
│         │                                                   │
│         ▼ prompt |                                          │
│   ┌───────────┐                                             │
│   │Prompt     │ 格式化为：                                   │
│   │Template   │ "What is a good name for a company          │
│   └─────┬─────┘  that makes colorful socks?"               │
│         │                                                   │
│         ▼ llm |                                             │
│   ┌───────────┐                                             │
│   │ChatModel  │ 调用API，返回：                              │
│   │(LLM)      │ "Socktastic"                               │
│   └─────┬─────┘                                             │
│         │                                                   │
│         ▼ StrOutputParser |                                 │
│   ┌───────────┐                                             │
│   │Output     │ 解析为字符串：                               │
│   │Parser     │ "Socktastic"                               │
│   └─────┬─────┘                                             │
│         │                                                   │
│         ▼                                                   │
│   ┌───────────┐                                             │
│   │ 最终结果   │                                             │
│   │"Socktastic│                                            │
│   └───────────┘                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 主流框架详解

### 6.1 LangChain - 全能型LLM应用框架

**定位：** 全能型框架，生态丰富，适合各种复杂场景

**GitHub:** github.com/langchain-ai/langchain

**核心特性：**

| 特性         | 说明                        |
| ------------ | --------------------------- |
| LCEL管道语法 | 直观易懂的链式调用          |
| 丰富的生态   | 100+模型、50+向量数据库支持 |
| @tool装饰器  | 最简洁的工具注册方式        |
| 完善记忆管理 | session_id支持多用户并发    |

**适用场景：**

- 工具调用型Agent（网络诊断、API调用等）
- 多轮对话系统（客服机器人）
- 复杂流程编排
- 快速原型开发

**代码示例：**

```python
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent

# 加载模型
llm = ChatTongyi(model_name="deepseek-v3", dashscope_api_key="your-key")

# 定义工具
tools = [ping_tool, dns_tool, calculator]

# 创建Agent
agent = create_agent(llm, tools)

# 调用
result = agent.invoke({
    "messages": [("user", "检查 www.example.com 的连通性")]
})
print(result["messages"][-1].content)
```

### 6.2 LlamaIndex - 数据驱动的RAG专家

**定位：** 为LLM装上私有数据的最强接口

**GitHub:** github.com/run-llama/llama_index

**核心哲学：Index-First**

- 不同于LangChain关注流程，LlamaIndex关注数据结构
- 核心问题：如何让LLM高效索引私有数据

```
┌─────────────────────────────────────────────────────────────┐
│              LlamaIndex RAG 完整流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              预处理阶段 (Ingestion)                  │   │
│  │                                                     │   │
│  │  📄文档文件 → 📖加载 → 📑分块 → 🔢向量化 → 💾索引   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                               │
│                            ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              查询阶段 (Query)                        │   │
│  │                                                     │   │
│  │  ❓用户问题 → 🔍向量检索 → 📋上下文 → 🤖LLM回答     │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**适用场景：**

- 企业知识库问答
- 合同审查助手
- 学术论文分析
- 客服机器人

**代码示例：**

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# 读取文档
reader = SimpleDirectoryReader('./docs')
documents = reader.load_data()

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 创建检索工具
query_engine = index.as_query_engine()
retrieve_tool = FunctionTool.from_defaults(fn=lambda q: str(query_engine.query(q)))

# 创建Agent
agent = ReActAgent.from_tools(
    tools=[retrieve_tool],
    llm=llm,
    system_prompt="你是一个乐于助人的AI助手"
)

# 对话
response = agent.chat("介绍下雇主责任险")
```

### 6.3 Qwen-Agent - 轻量级的全能选手

**定位：** 阿里生态亲儿子，轻量、灵活

**GitHub:** github.com/QwenLM/Qwen-Agent

**核心优势：**

| 能力             | 说明                                 |
| ---------------- | ------------------------------------ |
| Tool Use优化     | 专门为Qwen模型的Tool Calling能力优化 |
| Code Interpreter | 内置代码执行沙箱，自我修正错误       |
| 内置WebUI        | 一行代码启动Web界面                  |
| 长文本优势       | 支持超长Context（1M Token）          |

**适用场景：**

- 数据分析（Code Interpreter绑图）
- 复杂工具调用链
- 图像处理（生成、编辑、分析）
- 长文档问答

**代码示例：**

```python
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

# 定义工具
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    description = 'AI绘画服务'
    parameters = [...]

    def call(self, params, **kwargs):
        ...

# 创建Assistant
bot = Assistant(
    llm=llm_cfg,
    system_message="你是一个乐于助人的AI助手",
    function_list=['my_image_gen', 'code_interpreter'],
    files=['./docs/file1.txt', './docs/file2.txt']
)

# 对话
response = []
for response in bot.run(messages):
    print(response[0]['content'], end='')
```

### 6.4 AutoGen - 多智能体框架

**定位：** 微软开源的多智能体对话框架

**核心理念：** Agent之间通过自然语言对话协作，而非硬编码的函数调用

```
┌─────────────────────────────────────────────────────────────┐
│              AutoGen 多智能体协作架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      ┌─────────┐                           │
│                      │  用户   │                           │
│                      └────┬────┘                           │
│                           │                                 │
│                           ▼                                 │
│                      ┌─────────┐                           │
│                      │GroupChat│                           │
│                      │ Manager │                           │
│                      └────┬────┘                           │
│           ┌───────────────┼───────────────┐               │
│           │               │               │               │
│           ▼               ▼               ▼               │
│      ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│      │Assistant│◄──►│ User    │◄──►│ Critic  │          │
│      │ Agent   │    │Proxy    │    │ Agent   │          │
│      └────┬────┘    └────┬────┘    └────┬────┘          │
│           │              │              │               │
│           ▼              ▼              ▼               │
│        写代码         执行代码         评审结果           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> **注意：** 2025年10月起，AutoGen进入维护模式，新特性都迁移到Agent Framework

---

## 7. 框架对比与选型

### 7.1 核心维度对比

| 维度           | LangChain                  | Qwen-Agent           | LlamaIndex     | AutoGen        |
| -------------- | -------------------------- | -------------------- | -------------- | -------------- |
| **核心定位**   | 全能型框架                 | 轻量工具调用         | RAG数据接口    | 多Agent协作    |
| **工具注册**   | @tool装饰器                | @register_tool       | FunctionTool   | @register      |
| **RAG支持**    | 需集成VectorStore          | 基础文件读取         | 专业级向量索引 | 需自行集成     |
| **多Agent**    | LangGraph支持              | 基础支持             | 需自行编排     | 原生GroupChat  |
| **代码执行**   | 需集成                     | 内置code_interpreter | 需集成         | UserProxyAgent |
| **记忆管理**   | RunnableWithMessageHistory | messages列表         | Agent内置      | GroupChat自动  |
| **学习曲线**   | 中等                       | 简单                 | 中等           | 中等           |
| **生态完整度** | 最丰富                     | 阿里生态             | RAG社区        | 微软生态       |

### 7.2 选型指南

```
┌─────────────────────────────────────────────────────────────┐
│                      框架选型指南                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🎯 选 LangChain 如果：                                      │
│     • 开发通用的AI应用                                       │
│     • 需要灵活控制流程                                       │
│     • 需要切换多种模型                                       │
│     • 需要丰富的预置组件                                      │
│                                                             │
│  🎯 选 LlamaIndex 如果：                                     │
│     • 主要做RAG（检索增强生成）                               │
│     • 有大量PDF/Word/Excel要处理                             │
│     • 构建企业知识库                                         │
│     • 需要专业级的向量检索                                    │
│                                                             │
│  🎯 选 Qwen-Agent 如果：                                     │
│     • 主要用Qwen模型                                         │
│     • 需要数据分析（Code Interpreter）                       │
│     • 处理超长文档（1M Context）                              │
│     • 快速搭建工具调用型Agent                                 │
│                                                             │
│  🎯 选 AutoGen 如果：                                        │
│     • 任务太复杂，单Agent干不完                               │
│     • 需要多角色协作讨论                                     │
│     • 需要Agent之间自然对话                                   │
│     • 微软技术栈                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 实战案例：多文件智能问答

### 8.1 项目背景

构建一个**保险产品智能问答Agent**，帮助用户快速了解各类保险产品的详细信息。

**加载的文档：**

- 雇主责任险
- 平安商业综合责任保险
- 企业团体综合意外险
- 财产一切险
- 施工保、装修保等

### 8.2 RAG核心流程

```
用户问题 → 向量检索 → 召回相关文档 → LLM生成回答
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│                    为什么需要RAG？                           │
├─────────────────────────────────────────────────────────────┤
│  1. LLM没有私有数据的知识                                    │
│  2. 避免模型幻觉（编造信息）                                  │
│  3. 回答可追溯到具体文档来源                                  │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 LangChain实现

```python
#!/usr/bin/env python
# coding: utf-8
"""
基于 LangChain 的多文件 RAG 应用
支持加载 docs 文件夹下的多种格式文件进行问答
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 获取 API Key
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


# 步骤 1：加载文档并创建索引
def load_documents_and_create_index(file_dir: str = './docs', persist_dir: str = './langchain_storage'):
    """加载文档文件夹中的所有文件并创建向量索引"""
    
    # 创建嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 检查索引是否已存在
    if os.path.exists(persist_dir):
        try:
            # 从存储中加载索引
            vector_store = FAISS.load_local(
                persist_dir, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("从存储加载索引成功")
            return vector_store
        except Exception as e:
            print(f"加载索引失败: {e}，将重新创建索引")
    
    # 如果索引不存在，创建新索引
    if not os.path.exists(file_dir):
        print(f"文档目录 {file_dir} 不存在")
        return None
    
    # 加载目录下的所有 txt 文件
    loader = DirectoryLoader(
        file_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档")
    
    if not documents:
        print("没有找到任何文档")
        return None
    
    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文本被分割成 {len(chunks)} 个块")
    
    # 创建向量索引
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 保存索引
    os.makedirs(persist_dir, exist_ok=True)
    vector_store.save_local(persist_dir)
    print(f"索引已保存到 {persist_dir}")
    
    return vector_store


# 步骤 2：创建问答链
def create_qa_chain(llm):
    """创建 QA 问答链 (LangChain 1.x LCEL 写法)"""
    
    # QA Prompt 模板
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个乐于助人的AI助手。
根据以下上下文内容回答用户的问题。如果上下文中没有相关信息，请如实说明。
你总是用中文回复用户。

上下文内容:
{context}"""),
        ("human", "{question}")
    ])
    
    # 创建问答链 (LCEL 管道语法)
    qa_chain = qa_prompt | llm | StrOutputParser()
    
    return qa_chain


# 步骤 3：主函数
def main():
    """主函数"""
    # 配置 LLM
    llm = ChatTongyi(
        model_name="deepseek-v3",
        dashscope_api_key=DASHSCOPE_API_KEY
    )
    
    # 加载文档并创建索引
    vector_store = load_documents_and_create_index()
    if vector_store is None:
        print("无法创建索引，程序退出")
        return
    
    # 创建问答链
    qa_chain = create_qa_chain(llm)
    
    # 执行查询
    query = "介绍下雇主责任险"
    print(f"\n用户查询: {query}\n")
    
    # 相似度搜索，找到相关文档
    docs = vector_store.similarity_search(query, k=5)
    
    # 显示召回的文档内容
    print("===== 召回的文档内容 =====")
    if docs:
        for i, doc in enumerate(docs):
            print(f"\n文档片段 {i+1}:")
            print(f"内容: {doc.page_content[:200]}...")
            print(f"来源: {doc.metadata.get('source', '未知')}")
    else:
        print("没有召回任何文档内容")
    print("===========================\n")
    
    # 格式化上下文
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # 执行问答链
    print("===== AI 回复 =====")
    response = qa_chain.invoke({"context": context, "question": query})
    print(response)
    print("===================\n")


if __name__ == "__main__":
    main()

```

### 8.4 LlamaIndex实现

```python
#!/usr/bin/env python
# coding: utf-8

import os
import asyncio
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)


# 步骤 1：配置 LLM 和 Embedding
def setup_llm_and_embedding():
    """配置 LLM 和 Embedding，使用 DashScope"""
    api_key = os.getenv('DASHSCOPE_API_KEY')
    
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    
    # 使用 DashScope LLM
    llm = DashScope(
        model="deepseek-v3",
        api_key=api_key,
        temperature=0.7,
        top_p=0.8,
    )
    
    # 使用 DashScope Embedding（自动从环境变量读取 API key）
    embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    )
    
    return llm, embed_model


# 步骤 2：加载文档并创建索引
def load_documents_and_create_index(file_dir: str = './docs'):
    """加载文档文件夹中的所有文件并创建向量索引"""
    # 检查索引是否已存在
    persist_dir = "./storage"
    
    if os.path.exists(persist_dir):
        try:
            # 从存储中加载索引
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print("从存储加载索引成功")
            return index
        except Exception as e:
            print(f"加载索引失败: {e}，将重新创建索引")
    
    # 如果索引不存在，创建新索引
    if not os.path.exists(file_dir):
        print(f"文档目录 {file_dir} 不存在")
        return None
    
    # 读取文档
    reader = SimpleDirectoryReader(file_dir)
    documents = reader.load_data()
    
    if not documents:
        print("没有找到任何文档")
        return None
    
    print(f"加载了 {len(documents)} 个文档")
    
    # 创建向量索引
    index = VectorStoreIndex.from_documents(documents)
    
    # 保存索引
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"索引已保存到 {persist_dir}")
    
    return index


# 步骤 3：创建智能体
def create_agent(index, llm):
    """创建 ReAct 智能体"""
    # 创建检索器
    retriever = index.as_retriever(similarity_top_k=5)
    
    # 创建查询引擎（用于检索工具）
    query_engine = index.as_query_engine(similarity_top_k=5)
    
    # 定义系统提示词
    system_instruction = '''你是一个乐于助人的AI助手。
你可以从给定的文档中检索相关信息来回答用户的问题。
你总是用中文回复用户。'''
    
    # 创建检索工具（用于查询文档）
    def retrieve_documents(query: str) -> str:
        """从文档中检索相关信息"""
        response = query_engine.query(query)
        return str(response)
    
    retrieve_tool = FunctionTool.from_defaults(fn=retrieve_documents)
    
    # 创建智能体（新版 API）
    agent = ReActAgent(
        tools=[retrieve_tool],
        llm=llm,
        system_prompt=system_instruction,
    )
    
    return agent, retriever


# 步骤 4：主函数
async def main():
    """主函数"""
    # 配置 LLM 和 Embedding
    llm, embed_model = setup_llm_and_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 加载文档并创建索引
    index = load_documents_and_create_index()
    if index is None:
        print("无法创建索引，程序退出")
        return
    
    # 创建智能体
    agent, retriever = create_agent(index, llm)
    
    # 执行查询
    query = "介绍下雇主责任险"
    print(f"\n用户查询: {query}\n")
    
    # 显示召回的文档内容
    print("\n===== 召回的文档内容 =====")
    retrieved_nodes = retriever.retrieve(query)
    if retrieved_nodes:
        for i, node in enumerate(retrieved_nodes):
            print(f"\n文档片段 {i+1}:")
            # 处理特殊字符，避免 Windows 控制台编码问题
            text_preview = node.text[:200].encode('gbk', errors='replace').decode('gbk')
            print(f"内容: {text_preview}...")  # 只显示前200个字符
            print(f"元数据: {node.metadata}")
            if hasattr(node, 'score'):
                print(f"相似度分数: {node.score}")
    else:
        print("没有召回任何文档内容")
    print("===========================\n")
    
    # 使用智能体回答问题（新版 API 使用 run 方法，是异步的）
    print("\n===== 智能体回复 =====")
    response = await agent.run(query)
    # 处理特殊字符，避免 Windows 控制台编码问题
    response_str = str(response).encode('gbk', errors='replace').decode('gbk')
    print(response_str)
    print("======================\n")


if __name__ == "__main__":
    asyncio.run(main())
```

### 8.5 Qwen-Agent实现

```python
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI
import os

# 步骤 1：添加一个名为 `my_image_gen` 的自定义工具。
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    # `description` 用于告诉智能体该工具的功能。
    description = 'AI 绘画（图像生成）服务，输入文本描述，返回基于文本信息绘制的图像 URL。'
    # `parameters` 告诉智能体该工具有哪些输入参数。
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '期望的图像内容的详细描述',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        # `params` 是由 LLM 智能体生成的参数。
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)


# 步骤 2：配置您所使用的 LLM。
llm_cfg = {
    # 使用 DashScope 提供的模型服务：
    'model': 'deepseek-v3',
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),  # 从环境变量获取API Key
    'generate_cfg': {
        'top_p': 0.8
    }
}

# 步骤 3：定义系统提示词和工具列表
system_instruction = '''你是一个乐于助人的AI助手。
在收到用户的请求后，你应该：
- 首先绘制一幅图像，得到图像的url，
- 然后运行代码`requests.get`以下载该图像的url，
- 最后从给定的文档中选择一个图像操作进行图像处理。
用 `plt.show()` 展示图像。
你总是用中文回复用户。'''
tools = ['my_image_gen', 'code_interpreter']  # `code_interpreter` 是框架自带的工具，用于执行代码。

# 获取文件夹下所有文件
def get_doc_files():
    """获取 docs 文件夹下的所有文件"""
    file_dir = os.path.join('./', 'docs')
    files = []
    if os.path.exists(file_dir):
        # 遍历目录下的所有文件
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):  # 确保是文件而不是目录
                files.append(file_path)
    print('加载的文件:', files)
    return files


# ====== 初始化智能体服务 ======
def init_agent_service():
    """初始化智能体服务"""
    try:
        # 获取文档文件列表
        files = get_doc_files()
        
        bot = Assistant(
            llm=llm_cfg,
            system_message=system_instruction,
            function_list=tools,
            files=files
        )
        print("智能体初始化成功！")
        return bot
    except Exception as e:
        print(f"智能体初始化失败: {str(e)}")
        raise


def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('\n用户问题: ')
                
                # 输入验证
                if not query:
                    print('用户问题不能为空！')
                    continue
                    
                # 构建消息
                messages.append({'role': 'user', 'content': query})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                current_index = 0
                for response in bot.run(messages=messages):
                    if current_index == 0:
                        # 尝试获取并打印召回的文档内容
                        if hasattr(bot, 'retriever') and bot.retriever:
                            print("\n===== 召回的文档内容 =====")
                            retrieved_docs = bot.retriever.retrieve(query)
                            if retrieved_docs:
                                for i, doc in enumerate(retrieved_docs):
                                    print(f"\n文档片段 {i+1}:")
                                    print(f"内容: {doc.page_content[:200]}...")
                                    print(f"元数据: {doc.metadata}")
                            else:
                                print("没有召回任何文档内容")
                            print("===========================\n")
                    
                    current_response = response[0]['content'][current_index:]
                    current_index = len(response[0]['content'])
                    print(current_response, end='')
                
                # 将机器人的回应添加到聊天历史
                messages.extend(response)
                print("\n")
            except KeyboardInterrupt:
                print("\n\n退出程序")
                break
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        
        # 配置聊天界面，列举一些典型问题
        chatbot_config = {
            'prompt.suggestions': [
                '介绍下雇主责任险',
                '帮我生成一幅关于春天的图像',
                '分析一下文档中的关键信息',
            ]
        }
        
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）
    # app_tui()        # 终端交互模式（可选）


```

## requirement

```python
json5==0.9.14
langchain_community==0.4.1
langchain_core==1.2.8
langchain_text_splitters==1.1.0
llama_index==0.14.13
qwen_agent==0.0.25
```

