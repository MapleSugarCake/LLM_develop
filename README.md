# LLM-Develop: 异步文本智能分析工具

基于 Ollama 本地大模型的异步命令行文本分析工具，支持文本摘要生成、情感极性判断、关键词提取及多文档对比分析。

## ✨ 特性

- 🚀 **异步并发处理**：基于 `asyncio` 实现高并发 API 调用，大幅提升处理效率
- 📝 **智能文本分析**：自动完成摘要生成、情感分析、关键词提取三大核心任务
- 📊 **多文档对比**：支持跨文档语义合成，识别模式、冲突与共识
- 🔧 **超长文本处理**：内置分段滚动策略（Map-Reduce），支持长文本自动切片与聚合
- 🎯 **声明式签名系统**：模块化 Prompt 工程，支持 Chain-of-Thought (CoT) 推理与 Few-Shot 示例注入
- 💾 **自动报告生成**：分析结果自动保存为 Markdown 格式报告

## 📋 系统要求

- Python >= 3.13
- Ollama 服务（本地或远程）
- 推荐显存：30B 模型需 24GB+ 显存

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 2. 配置 Ollama 服务

编辑 `text_cli_Ollama_async.py` 中的配置区域：

```python
OLLAMA_API_URL = "http://open-webui-ollama.open-webui:11434"
MODEL_NAME = "qwen3-coder:30b"  # 或 "gpt-oss:20b"
```

### 3. 运行程序

```bash
python text_cli_Ollama_async.py
```

## 📖 使用说明

### 新建报告流程

1. 输入报告名称
2. 选择输入源：
   - `1`：直接输入纯文本
   - `2`：从文件读取文本
   - `3`：结束输入，开始分析
3. 系统自动执行以下任务：
   - 单文档分析 → 生成独立报告
   - 多文档对比（≥2 份资料）→ 生成对比分析报告

### 输出示例

```markdown
# 报告名称的智能分析报告
**生成时间**: 2025-04-12 15:30:00

---

### 📑 文本摘要
[AI 生成的核心摘要]

### 🎭 情感倾向
中性：肯定技术进步，指出资源约束。

### 🔑 核心关键词
Qwen3,编码能力,显存,参数规模,推理效率

---
```

## ⚙️ 高级配置

### 提示工程配置

```python
ENABLE_COT = True          # 启用 Chain-of-Thought 推理
FEW_SHOT_ENABLED = True    # 启用 Few-Shot 示例注入
```

### 分块策略配置

```python
MAX_CTX = 32768            # 全局上下文窗口
CHUNK_MAX_TOKENS = 8192    # 单次切片上限
CHUNK_OVERLAP = 300        # 片段重叠 token 数
```

### 并发控制

```python
MAX_API_CONCURRENCY = 3    # 最大并发请求数（根据显存调整）
```

## 🏗️ 项目结构

```
.
├── text_cli_Ollama_async.py  # 主程序入口
├── tool.py                    # 工具函数
├── pyproject.toml             # 项目配置与依赖
├── README.md                  # 项目说明文档
├── LICENSE                    # 开源许可证
├── Prompt/                    # Prompt 设计文档
│   ├── Prompt 第一版.md
│   ├── Prompt 第二版.md
│   ├── Prompt 第三版.md
│   └── Prompt 第四版.md
└── reports/                   # 生成的报告存储目录（自动创建）
```

## 🔬 核心技术

### 1. 异步架构
- 使用 `httpx.AsyncClient` 进行非阻塞 API 调用
- 信号量控制并发度，避免显存溢出
- 超时控制与指数退避重试机制

### 2. 声明式签名系统
```python
Signature(
    inputs={"text": Field("待分析的文本片段")},
    outputs={
        "summary": Field("精炼的核心摘要"),
        "sentiment": Field("情感极性", type_="enum[正面，负面，中性]"),
        "keywords": Field("核心术语列表", type_="list")
    },
    instruction="文本语义理解与结构化抽取..."
)
```

### 3. Map-Reduce 长文本处理
- **Map 阶段**：并发处理各文本片段
- **Reduce 阶段**：聚合片段结果生成全局摘要

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题，请通过 GitHub Issues 联系。
