# Local LangGraph Tool-Use Service

这个项目现在采用“`conda` 启动器 + Python 服务”的方式运行。

## 入口方式

不要直接先手动创建 Python 虚拟环境，推荐使用根目录启动器。

### Windows

初始化环境与模型：

```bat
bootstrap.bat
```

启动整个项目：

```bat
start.bat
```

### Linux

初始化环境与模型：

```bash
chmod +x bootstrap.sh start.sh
./bootstrap.sh
```

启动整个项目：

```bash
./start.sh
```

## 启动器职责

`bootstrap` 启动器会先读取 [config/config.yaml](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/config/config.yaml)，然后：

- 检查本机是否存在 `conda`
- 检查配置里的 `project.environment_name`
- 如果 conda 环境不存在，则自动创建
- 使用该 conda 环境执行 [init.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/init.py)

`start` 启动器会：

- 检查 conda 环境是否存在
- 使用 `conda run -n <env>` 执行 [start.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/start.py)
- 按顺序启动 LLM、MCP、Gateway

## init.py 会做什么

[init.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/init.py) 现在不负责创建环境，它只负责当前 conda 环境内部的初始化：

- 升级 `pip`
- 安装 [requirements.txt](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/requirements.txt)
- 按配置下载模型到 `models/`
- 在 Windows / macOS 上自动把 `vllm` 回退为 `transformers`

## start.py 会做什么

[start.py](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/start.py) 会：

- 启动本地 LLM 服务
- 启动 MCP 服务
- 启动 Gateway 服务
- 逐个检查 `/health`
- 将日志写入 `logs/`

## 默认 conda 配置

当前 [config/config.yaml](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/config/config.yaml) 中的默认环境配置是：

```yaml
project:
  environment_name: lg_local_tool_service
  python_version: "3.11"
```

## 默认模型配置

```yaml
llm:
  model_name: Qwen/Qwen3-1.7B
  service:
    server_backend: transformers
    model_source: Qwen/Qwen3-1.7B
    model_cache_dir: models
```

## 单独服务脚本

如果你已经进入目标 conda 环境，也可以继续使用 `scripts/` 下的服务脚本：

- [scripts/start_llm.sh](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/scripts/start_llm.sh)
- [scripts/start_mcp.sh](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/scripts/start_mcp.sh)
- [scripts/start_gateway.sh](/E:/something%20I%20can%20turn%20to/HouGarden/Tool-Use/LangGraphTest/scripts/start_gateway.sh)

## 日志

- `logs/llm.log`
- `logs/mcp.log`
- `logs/gateway.log`
