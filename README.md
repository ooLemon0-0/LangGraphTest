# Local LangGraph Tool-Use Service

这是一个本地运行的 LangGraph 工具调用示例项目。它把一个完整的 Agent 执行链拆成三个独立服务：

- `LLM Server`：提供本地 OpenAI-compatible 大模型接口
- `MCP Server`：提供工具清单和工具调用入口
- `Gateway`：承接外部请求，驱动 LangGraph 完成意图识别、工具规划、工具执行和结果整理

这个仓库目前更偏向“可运行骨架”而不是完整业务系统：核心链路已经打通，房源和经纪人工具也已经抽象出来，但大部分工具返回的还是 mock 数据，方便后续替换成真实数据库、CRM 或内部服务。

## 项目目标

这个项目主要用来验证一条本地 Agent 工作流：

1. 用户发送自然语言请求
2. LangGraph 判断这是不是查工具、读数据、改数据，还是普通问答
3. 如果需要工具，先从 MCP 工具清单里选择合适工具并组织参数
4. 调用 MCP Server 执行工具
5. 对结果做简单审查
6. 由 LLM 生成最终回答

从设计上看，它适合做这些事情：

- 本地验证 LangGraph + Tool Use 的整体链路
- 演示“模型决策”和“工具执行”解耦的架构
- 给后续真实业务 Agent 提供最小可扩展脚手架
- 快速替换 mock 工具为真实业务实现

## 整体架构

```text
User / Client
    |
    v
Gateway (FastAPI)
    |
    v
LangGraph Workflow
    |                    \
    |                     \ 
    v                      v
Local LLM Server        MCP Server
                              |
                              v
                         Tool Registry
                              |
                              v
                     Mock tools / Real business tools
```

默认端口：

- `Gateway`: `http://127.0.0.1:8000`
- `LLM Server`: `http://127.0.0.1:8001`
- `MCP Server`: `http://127.0.0.1:8002`

## 项目结构

```text
.
├─ app/
│  ├─ common/        # 公共配置、日志、跨服务 schema
│  ├─ gateway/       # 对外 API，负责调用 LangGraph
│  ├─ graph/         # LangGraph 状态、节点、路由、提示词
│  ├─ llm_client/    # 对 OpenAI-compatible LLM 的客户端封装
│  ├─ llm_server/    # 本地模型服务
│  └─ mcp_server/    # 工具服务、工具注册表、工具实现
├─ config/
│  ├─ config.yaml    # 项目主配置
│  └─ tools.yaml     # MCP 工具清单
├─ logs/             # 运行日志
├─ models/           # 本地模型缓存
├─ scripts/          # 分服务启动脚本
├─ init.py           # 安装依赖并下载模型
├─ start.py          # 启动 LLM / MCP / Gateway 三个服务
├─ bootstrap.*       # 按平台初始化运行环境
└─ start.*           # 按平台启动项目
```

## 核心模块说明

### 1. Gateway

`app/gateway/` 是项目的外部入口。

主要职责：

- 提供 `/v1/chat` 接口
- 构造 LangGraph 初始状态
- 执行整个图
- 保存 trace，便于调试和回放
- 代理 `/v1/tools` 到 MCP Server

关键文件：

- [app/gateway/main.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/app/gateway/main.py)
- [app/gateway/api.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/app/gateway/api.py)

### 2. LLM Server

`app/llm_server/` 提供本地 OpenAI-compatible 接口，默认实现基于 `transformers` 加载本地 Hugging Face 模型。

主要职责：

- 启动本地模型
- 提供 `/v1/chat/completions`
- 给 LangGraph 节点中的分类、规划、总结阶段提供统一推理能力

如果配置中把 `server_backend` 设为 `vllm`，Linux 下可以切到 vLLM；Windows 和 macOS 会自动回退到 `transformers`。

关键文件：

- [app/llm_server/main.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/app/llm_server/main.py)

### 3. MCP Server

`app/mcp_server/` 是工具层，负责把“工具定义”和“工具实现”集中管理。

主要职责：

- 暴露 `/tools` 返回工具清单
- 暴露 `/invoke` 执行指定工具
- 从 `config/tools.yaml` 加载工具元数据
- 通过 `ToolRegistry` 把工具名绑定到 Python 函数

当前工具分成三类：

- 元工具：`list_tools`、`get_tool_detail`
- 经纪人工具：`get_agent_id_by_name`、`get_houses_by_agent_id`、`get_agent_by_house_id`
- 房源工具：`get_house_detail`、`update_house_name`、`update_house_price`

当前这些工具大多返回 mock 数据，适合作为接入真实业务前的联调层。

关键文件：

- [app/mcp_server/main.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/app/mcp_server/main.py)
- [app/mcp_server/registry.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/app/mcp_server/registry.py)
- [config/tools.yaml](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/config/tools.yaml)

## LangGraph 设计

LangGraph 实现在 `app/graph/` 下，核心入口是：

- [app/graph/build_graph.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/app/graph/build_graph.py)

### 图中的状态

图状态定义在 [app/graph/state.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/app/graph/state.py)，主要字段有：

- `messages`：原始会话消息
- `normalized_user_input`：归一化后的用户输入
- `intent`：意图分类结果，支持 `tool_lookup` / `read` / `write` / `general`
- `tool_plan`：模型规划出的工具调用列表
- `tool_results`：工具执行结果
- `review_notes`：结果审查备注
- `final_answer`：最终返回给用户的答案
- `trace_id`：一次调用的追踪标识

### 图中的节点

当前工作流节点如下：

```text
START
  -> normalize_input
  -> classify_intent
  -> plan_tool_calls
  -> [if tool_plan exists] execute_tools
  -> review_results
  -> finalize
  -> END
```

各节点职责如下：

#### `normalize_input`

- 提取最后一条用户消息
- 去掉多余空白
- 生成稳定输入，减少后续提示词波动

#### `classify_intent`

- 用 LLM 把请求归类到四种意图
- 同时输出简单 `rationale`
- 如果模型调用失败，会退回到关键词启发式判断

#### `plan_tool_calls`

- 先向 MCP Server 拉取 `/tools`
- 把工具清单和用户请求一起交给 LLM
- 生成 `tool_calls`
- 如果模型规划失败，会进入简单 heuristic fallback

这个设计有一个明显优点：LangGraph 不直接写死工具知识，而是运行时读取工具清单，因此工具层可以独立演进。

#### `execute_tools`

- 遍历 `tool_plan`
- 对每个工具向 MCP `/invoke` 发请求
- 收集成功和失败结果

#### `review_results`

- 对工具执行结果做轻量审查
- 当前主要标注两类信息：
  - 是否返回 mock 数据
  - 是否有工具执行失败

这一步现在比较轻，但它给后续扩展留了位置，比如：

- 风险审查
- 写操作确认
- 权限校验
- 多工具结果一致性检查

#### `finalize`

- 把用户请求、工具结果、审查备注交给 LLM
- 生成最终用户回答
- 如果 LLM 不可用，会退回到简单兜底文本

### 条件路由

条件路由定义在 [app/graph/router.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/app/graph/router.py)：

- 如果 `tool_plan` 非空，走 `execute_tools`
- 如果 `tool_plan` 为空，直接跳到 `review_results`

所以这个图既支持“需要工具的请求”，也支持“不需要工具的普通回答”。

### 提示词设计

提示词定义在 [app/graph/prompts.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/app/graph/prompts.py)，分成三类：

- `intent_classification_prompt`
- `tool_planning_prompt`
- `final_answer_prompt`

这种拆法的优点是：

- 每个节点只关心当前任务
- 便于单独优化分类、规划、总结的 prompt
- 出问题时更容易定位是“理解错了”还是“工具选错了”

## 当前请求流

一次 `/v1/chat` 调用的大致过程如下：

1. 客户端向 Gateway 发送消息
2. Gateway 创建初始图状态并执行 LangGraph
3. 图先做输入归一化和意图分类
4. 图读取 MCP 工具清单并做工具规划
5. 如果有工具计划，就调用 MCP Server 执行
6. 图整理工具结果并生成最终答复
7. Gateway 返回答案、工具调用记录、审查备注和原始状态

## API 概览

### Gateway

- `GET /health`
- `POST /v1/chat`
- `GET /v1/tools`
- `GET /v1/traces/{trace_id}`

### MCP Server

- `GET /health`
- `GET /tools`
- `GET /tools/{tool_name}`
- `POST /invoke`

### LLM Server

- `GET /health`
- `POST /v1/chat/completions`

## 快速启动

### Windows

初始化环境：

```bat
bootstrap.bat
```

启动服务：

```bat
start.bat
```

### Linux

初始化环境：

```bash
chmod +x bootstrap.sh start.sh
./bootstrap.sh
```

启动服务：

```bash
./start.sh
```

## `bootstrap` 和 `start` 做了什么

### `bootstrap`

`bootstrap` 会在目标 conda 环境里执行 [init.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/init.py)，主要做这些事：

- 创建 `logs/` 和 `models/`
- 安装 `requirements.txt`
- 安装合适的 `torch` 运行时
- 按配置下载模型到本地
- Linux + NVIDIA + conda 环境下优先安装 CUDA 版 PyTorch
- 模型下载时优先走 ModelScope，失败后回退到 Hugging Face

### `start`

`start` 会执行 [start.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/start.py)，主要做这些事：

- 同时拉起 LLM / MCP / Gateway 三个服务
- 为每个服务写入独立日志
- 轮询健康检查直到服务可用
- 主进程退出时统一关闭子进程

## 关键配置

主配置文件：

- [config/config.yaml](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/config/config.yaml)

示例：

```yaml
project:
  environment_name: lg_local_tool_service
  python_version: "3.11"

llm:
  model_name: Qwen/Qwen3-1.7B
  base_url: http://127.0.0.1:8001/v1
  service:
    server_backend: transformers
    model_source: Qwen/Qwen3-1.7B
    model_cache_dir: models

mcp:
  manifest_path: config/tools.yaml

gateway:
  service:
    host: 127.0.0.1
    port: 8000
```

你通常会改这些项：

- `project.environment_name`：conda 环境名
- `llm.model_name`：对外展示的模型名
- `llm.base_url`：LangGraph 访问 LLM 的地址
- `llm.service.server_backend`：`transformers` 或 `vllm`
- `llm.service.model_source`：模型来源
- `llm.service.model_cache_dir`：模型缓存目录
- `mcp.manifest_path`：工具清单路径
- `gateway.service.port`：网关端口

## 日志

运行后日志默认写到 `logs/`：

- `logs/llm.log`
- `logs/mcp.log`
- `logs/gateway.log`

## 适合如何扩展

如果要把这个项目从 demo 骨架升级成真实业务版本，推荐按下面顺序扩展：

1. 把 `app/mcp_server/tools/` 中的 mock 实现替换成真实数据源
2. 在 `config/tools.yaml` 中补充更准确的工具描述、字段和风险等级
3. 增强 `review_results` 节点，加入写操作审批、权限和校验逻辑
4. 为 `trace` 增加持久化存储，替换当前内存版 `TraceStore`
5. 给 `plan_tool_calls` 增加更严格的参数约束和重试策略

## 目前的边界

当前项目已经具备可运行的最小闭环，但也有明确边界：

- 工具返回值以 mock 数据为主
- 没有真实持久化层
- 写操作没有审批或权限模型
- trace 仅保存在内存中
- prompt 和 fallback 逻辑偏演示性质

如果你的目标是“先把 LangGraph + 工具调用链跑起来”，这个仓库已经足够；如果目标是直接上生产，还需要继续补业务接入、鉴权、审计和持久化。
