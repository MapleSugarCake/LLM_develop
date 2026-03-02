import os
import time
import logging
import jieba
import asyncio
from typing import List, Dict
from pathlib import Path
import functools
import httpx
from ollama import AsyncClient, ResponseError

# ================================== 配置区域 ==================================
OLLAMA_API_URL = "http://open-webui-ollama.open-webui:11434"
MODEL_NAME = "qwen3-coder:30b"

# Chunking (分段策略) 配置
MAX_CTX = 32768
# 单次切片最大上限为 2048 Token
CHUNK_MAX_TOKENS = 2048
CHUNK_OVERLAP = 100

# 全局 API 最大并发请求限制
# 根据显存大小和模型并发能力设置，30b 模型建议设置 2~5
MAX_API_CONCURRENCY = 2
_api_semaphore = None    # 信号量对象（必须在事件循环启动后初始化）

# 禁用 jieba 的默认日志输出，保持 CLI 清洁
jieba.setLogLevel(logging.INFO)

# 初始化报告存储目录
BASE_DIR = Path("./reports")
BASE_DIR.mkdir(parents=True, exist_ok=True)


# ================================== 调试代码 (兼容同步与异步) ==================================
def timetest(func):
    """
    智能装饰器，自动判断函数是同步还是异步，并计算耗时
    """
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            print(f"函数 '{func.__name__}' 耗时：{end - start:.4f} 秒")
            return result

        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"函数 '{func.__name__}' 耗时：{end - start:.4f} 秒")
            return result

        return sync_wrapper


# ================================== API 交互与异常处理 ==================================
@timetest
async def call_ollama_chat(system_prompt: str, user_prompt: str, retries: int = 3, timeout: int = 300) -> str:
    """
    调用 Ollama Chat Completion Ollama库[异步版：超时控制、网络波动重试与频率限制处理]
    """
    # 懒加载初始化信号量（确保在 asyncio 事件循环中创建）
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(MAX_API_CONCURRENCY)

    # 利用信号量限制并发，超过限制的协程会在这里等待（排队）
    async with _api_semaphore:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        model_options={
            #=================设置大模型的额外参数=================
            "num_ctx":MAX_CTX,
            "repeat_last_n": 64,
            'temperature': 0.5
        }
        print(f"\n{messages}\n")

        backoff = 2  # 初始退避时间

        print(f"协程获取到执行权，开始处理... (当前可用并发槽位: {_api_semaphore._value})")

        for attempt in range(retries):
            try:
                # 初始化异步客户端并设置超时时间 600 s
                client = AsyncClient(host=OLLAMA_API_URL, timeout=timeout)
                # 使用 await 发起非阻塞请求
                response = await client.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    stream=False,
                    options=model_options
                )

                print(f"\n《《《data》》》:\n{response}")
                return response.get('message', {}).get('content', '').strip()

        #==================================错误捕获区域==================================
            except ResponseError as e:
                # 专门捕获 Ollama 响应错误
                if e.status_code == 429:
                    print(f"  [警告] 触发 API 频率限制 (429)，{backoff}秒后重试...")
                else:
                    print(f"  [错误] Ollama API 错误: {e.error} (状态码: {e.status_code})。尝试 {attempt + 1}/{retries}...")

            except httpx.TimeoutException:
                # 超时抛出 httpx.TimeoutException
                print(f"[错误] API 请求超时 (Timeout)。尝试 {attempt + 1}/{retries}...")
            except httpx.ConnectError:
                print(f"  [错误] 网络连接失败，请检查 Ollama 服务({OLLAMA_API_URL})。尝试 {attempt + 1}/{retries}...")
            except Exception as e:
                print(f"  [错误] 未知调用异常: {e}。尝试 {attempt + 1}/{retries}...")

            # 触发异常后进行异步退避等待（绝对不能用 time.sleep，会阻塞整个事件循环）
            await asyncio.sleep(backoff)
            backoff *= 2

        return "【API 请求失败，无法生成结果。】"


# ================================== 上下文超长切片管理 ==================================
@timetest
def chunk_text(text: str) -> List[str]:
    """
    分段滚动处理 (Chunking & Sliding Window): 同步分词器，CPU密集
    """
    words = list(jieba.cut(text))
    total_tokens = len(words)

    if total_tokens <= CHUNK_MAX_TOKENS:
        return [text]

    #=================长文本=================
    print(f"  [信息] 文本总 token 估算为 {total_tokens}，超出单次处理限制，启动分段滚动处理策略...")
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + CHUNK_MAX_TOKENS, total_tokens)
        chunk = "".join(words[start:end])
        chunks.append(chunk)
        if end == total_tokens:
            break
        start += (CHUNK_MAX_TOKENS - CHUNK_OVERLAP)

    return chunks

# =================文本长度级别检测=================
# def chunk_length(text: str)-> int:
#     words = list(jieba.cut(text))
#     total_tokens = len(words)
#     ctx_sizes = [2048, 4096, 8192, 16384, 25000]
#     time_level = [300, 600, 1200, 2400, 3000, 3600]
#     for size in ctx_sizes:
#         if total_tokens <= size:
#             # 返回对应的级别（0=2048, 1=4096...）
#             return time_level[ctx_sizes.index(size)]
#     return time_level[5]


# ================================== 核心分析逻辑 ==================================
@timetest
async def extract_features(text: str) -> Dict[str, str]:
    """利用异步机制并发对单一片段提取三大基础特征"""

    sys_prompt = "你是一名专注于文本语义理解与结构化抽取的计算语言学家，擅长在少样本设定下完成摘要生成、情感极性判断与关键术语识别任务。"

    p_summary = f"""任务：请对以下由 ``` 包裹的文本进行信息浓缩，生成高保真度的核心摘要。

```
{text}
```
执行要求：
- 聚焦文本的主旨论点、关键事实与结论性陈述
- 保持原意完整性，避免主观推断或信息增补
- 输出应为连贯的陈述句，长度控制在原文15%-25%
直接输出摘要内容：："""

    p_sentiment = f"""任务：判断以下由 ``` 包裹的文本的情感极性并提供可解释依据。
```
{text}
```
执行要求：
- 从词汇情感、句法语气与语用意图三层面综合判断
- 输出格式：【情感倾向】+ 简短理由（不超过30字）
- 倾向类别限定：正面 / 负面 / 中性
直接输出分析结果："""

    p_keywords = f"""任务：从以下由 ``` 包裹的文本中抽取最具语义代表性的关键术语。

{text}
```
执行要求：
- 优先选择承载核心概念、领域术语或高频实体的词语
- 数量：5-10个，按语义重要性降序排列
- 使用英文逗号分隔，不包含序号或解释
直接输出关键词列表："""

    # 并发执行多个协程，等待全部完成
    f_sum, f_sen, f_kwd = await asyncio.gather(
        call_ollama_chat(sys_prompt, p_summary),
        call_ollama_chat(sys_prompt, p_sentiment),
        call_ollama_chat(sys_prompt, p_keywords)
    )

    return {
        "summary": f_sum,
        "sentiment": f_sen,
        "keywords": f_kwd
    }


@timetest
async def process_single_document(text: str, index: int) -> Dict[str, str]:
    """
    处理单个文档输入（集成超长文 Map-Reduce 合并逻辑）
    """
    print(f"[*] 开始分析文本档 {index}...")
    chunks = chunk_text(text)

    # 短文本直接处理
    if len(chunks) == 1:
        res = await extract_features(chunks[0])
        print(f"[+] 文本档 {index} 分析完成。")
        return res

    # 长文本 Map-Reduce 处理: 异步并发处理各片段
    print(f"  [信息] 文本档 {index} 被切分为 {len(chunks)} 个片段，正在并行处理各片段...")
    tasks = [extract_features(chunk) for chunk in chunks]
    chunk_results = await asyncio.gather(*tasks)

    print(f"  [信息] 文本档 {index} 各片段处理完毕，启动全局 Reduce 结果聚合...")
    sys_prompt = "你是一个专业的文本处理专家，负责融合并汇总局部信息。"

# 先将数据拼接好
    chunk_summaries = "\n---\n".join([r["summary"] for r in chunk_results])
    chunk_sentiments = "\n---\n".join([r["sentiment"] for r in chunk_results])
    chunk_keywords = "\n---\n".join([r["keywords"] for r in chunk_results])

# 1. 全局摘要汇总提示词
    agg_sum = f"""任务：对以下由 ``` 包裹的多个局部摘要进行语义融合，生成逻辑自洽的全局摘要。
    ```
    {chunk_summaries}
    ```
    执行三重融合操作：①信息去冗（识别并合并重复陈述）②逻辑重构（按"主旨-论据-结论"层级重组）③语义补全（填补片段间的推理间隙）
保持原文立场与事实准确性，避免引入新信息或主观推断
输出应为单一连贯段落，长度控制在输入总长度的30%-40%
直接输出全局摘要："""
# 2. 全局情感汇总提示词
    agg_sen = f"""任务：对以下由 ``` 包裹的同一文章不同段落的情感分析结果进行极性聚合与冲突消解。
    ```
    {chunk_sentiments}
    ```
    执行要求：
采用加权聚合策略：综合考量情感强度、段落位置（首尾段权重更高）与语义转折关系
若存在极性冲突，需识别主导情感并简要说明冲突片段的处理逻辑
输出格式：【全局情感倾向】+ 融合理由（不超过40字，聚焦聚合依据而非复述内容）
倾向类别限定：正面 / 负面 / 中性
直接输出分析结果："""
# 3. 全局关键词汇总提示词
    agg_kwd = f"""任务：从以下由 ``` 包裹的多个局部关键词列表中收敛出最具语义代表性的核心术语集。
    ```
    {chunk_keywords}
    ```
   执行要求：
执行三维筛选：①频次维度（跨片段共现优先）②分布维度（在文本不同区域均有出现）③概念层级维度（优先上位概念而非实例）
严格去重后按语义重要性降序排列，输出恰好10个关键词
使用中文逗号分隔，不包含序号、解释或额外标点
直接输出关键词列表："""

    # 并发执行最终的三大融合任务
    f_sum, f_sen, f_kwd = await asyncio.gather(
        call_ollama_chat(sys_prompt, agg_sum),
        call_ollama_chat(sys_prompt, agg_sen),
        call_ollama_chat(sys_prompt, agg_kwd)
    )

    res = {
        "summary": f_sum,
        "sentiment": f_sen,
        "keywords": f_kwd
    }

    print(f"[+] 文本档 {index} 分段汇总分析完成。")
    return res


@timetest
async def generate_comparison(results: List[Dict[str, str]]) -> str:
    """多文档对比分析"""
    print("[*] 正在执行多文本交叉对比分析...")
    sys_prompt = (
    "你是一名跨文档语义合成专家，擅长从多源异构文本中识别模式、冲突与共识。"
    "你的任务是基于提供的局部分析结果，构建一份具有洞察力的对比报告。"
    "请遵循“证据优先、逻辑闭环”原则，避免简单的信息堆砌。"
)
    
    # 先拼接整理需要传入的数据
    texts_data = ""
    for i, r in enumerate(results):
        texts_data += f"### 文本 {i + 1} 分析\n- **摘要**: {r['summary']}\n- **情感**: {r['sentiment']}\n- **关键词**: {r['keywords']}\n\n"

    # 构造提示词
    user_prompt = f"""以下是由 ``` 包裹的多个独立文本的分析结果集合：
    ```
    {texts_data}
    ```
    任务目标：基于上述数据，生成一份结构化对比报告。请严格执行以下三个模块的合成逻辑：
核心差异
聚焦维度：情感极性分歧、事实陈述矛盾、叙事视角差异。
要求：指出具体文本编号间的冲突点，而非泛泛而谈。
主题共性
聚焦维度：高频实体、共享论点、一致的情感倾向。
要求：提炼跨文本的最大公约数，说明共识的稳定性。
综合总结
聚焦维度：整体态势评估、潜在风险提示、信息完整性评价。
要求：基于差异与共性的加权分析，给出高层结论。
输出格式：标准 Markdown。
直接开始输出报告内容：
    """
    return await call_ollama_chat(sys_prompt, user_prompt, 3, 900)


# ================= 输入过滤与清理 =================
def sanitize_input(text: str) -> str:
    """过滤控制字符和非法输入"""
    if not text:
        return ""
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch in ['\n', '\r', '\t'])
    return cleaned.strip()


# ================= 业务流管理 ===================
async def create_report():
    print("\n" + "=" * 40)
    print("           [ 新建报告 ]")
    print("=" * 40)

    report_name = input("请输入报告名称: ").strip()
    if not report_name:
        print("[拦截] 报告名称不能为空！")
        return

    report_dir = BASE_DIR / report_name
    report_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    print("\n请提供要分析的资料内容（可多次输入）。完成所有输入后，请按 '3' 开始分析。")
    while True:
        print("\n选择输入源:  1. 纯文本  |  2. 文本文件路径  |  3. 结束输入，开始分析 ")
        choice = input("操作 >> ").strip()

        if choice == '1':
            text = input("请输入纯文本内容: ")
            text = sanitize_input(text)
            if text:
                inputs.append(text)
                print(f"[成功] 已添加文本。当前共 {len(inputs)} 份资料。")
            else:
                print("[拦截] 空输入或全为非法字符，已忽略。")

        elif choice == '2':
            print("当前路径为： " + str(Path.cwd()))
            path = input("请输入纯文本文件路径 (如 ./data.txt): ").strip()
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = sanitize_input(f.read())
                        if text:
                            inputs.append(text)
                            print(f"[成功] 已读取文件并添加。当前共 {len(inputs)} 份资料。")
                        else:
                            print("[拦截] 文件内容为空，已忽略。")
                except Exception as e:
                    print(f"[错误] 读取文件失败: {e}")
            else:
                print("[错误] 路径无效或文件不存在。")

        elif choice == '3':
            if not inputs:
                print("[错误] 没有有效的输入内容，无法生成报告。")
                return
            break
        else:
            print("[错误] 无效选项。")

    print(f"\n[*] 开始流水线作业，处理 {len(inputs)} 份资料 (异步并发模式)...")

    # 定义包含持久化的包装任务
    async def task_wrapper(index: int, doc_text: str):
        try:
            res = await process_single_document(doc_text, index)
            # =================单文件保存=================
            md_line = [
                f"# {report_name}的文档{index}智能分析报告",
                f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "\n---",
                f"\n### 📑  文本摘要\n{res['summary']}",
                f"\n### 🎭  情感倾向\n{res['sentiment']}",
                f"\n### 🔑  核心关键词\n{res['keywords']}",
                "\n---"
            ]
            single_report = "\n".join(md_line)
            file_path = report_dir / f"资料{index}报告.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(single_report)
            print(f"\n[✔️ ] {index}报告生成成功！\n保存位置: {file_path.absolute()}")
            return index, res

        except Exception as e:
            print(f"[致命异常] 处理文本档 {index} 时出错: {e}")
            return index, {"summary": "处理失败", "sentiment": "处理失败", "keywords": "处理失败"}

    results = [None] * len(inputs)
    tasks = [task_wrapper(i + 1, text) for i, text in enumerate(inputs)]

    for future in asyncio.as_completed(tasks):
        idx, res = await future
        results[idx - 1] = res


    # 触发对比分析进阶功能
    if len(inputs) >= 2:
        # =================构建汇总 Markdown=================
        md_lines = [
            f"# 智能分析报告：{report_name}",
            f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "\n---"
        ]

        # =================基础分析合并=================
        for i, res in enumerate(results):
            md_lines.extend([
                f"\n## 资料 {i + 1} 分析结果",
                f"\n### 📑  文本摘要\n{res['summary']}",
                f"\n### 🎭  情感倾向\n{res['sentiment']}",
                f"\n### 🔑  核心关键词\n{res['keywords']}",
                "\n---"
            ])

        md_lines.append(f"\n# ⚖️ {report_name}的多资料深度对比分析")
        comparison_res = await generate_comparison(results)
        md_lines.append(comparison_res)

        summary_report = "\n".join(md_lines)

        # =================汇总文件保存=================
        files_path = report_dir / f"{report_name}的汇总分析报告.md"
        try:
            with open(files_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            print(f"\n[✔️ ] 汇总及差异报告生成成功！\n保存位置: {files_path.absolute()}")
        except Exception as e:
            print(f"\n[❌ ] 保存汇总报告失败: {e}")


def view_history():
    print("\n" + "=" * 40)
    print("           [ 历史报告 ]")
    print("=" * 40)

    files = list(BASE_DIR.rglob("*.md"))
    if not files:
        print("📁 暂无任何历史报告。")
        return

    for i, f in enumerate(files):
        print(f" {i + 1}. {f.name} (位于 {f.parent.name}, 大小: {f.stat().st_size} 字节)")

    choice = input("\n请输入要查看的报告编号 (输入 0 取消): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            try:
                with open(files[idx], 'r', encoding='utf-8') as f:
                    print("\n\n" + "▼" * 50)
                    print(f.read())
                    print("▲" * 50 + "\n")
            except Exception as e:
                print(f"[错误] 读取文件失败: {e}")
        elif choice != '0':
            print("[错误] 编号不存在。")
    else:
        print("[错误] 输入无效。")


# ================= 异步主循环 =================
async def main_loop():
    while True:
        print("\n" + "#" * 45)
        print("文本智能分析与报告助手 (Ollama Python 异步版)")
        print("#" * 45)
        print("  1. 新建分析报告")
        print("  2. 查看历史报告")
        print("  3. 退出系统")
        print("-" * 45)

        choice = input("请选择您的操作 (1/2/3): ").strip()

        if choice == '1':
            await create_report()
        elif choice == '2':
            view_history()
        elif choice == '3':
            print("感谢使用，系统退出。")
            break
        else:
            print("[拦截] 无效输入，请重新选择。")


if __name__ == "__main__":
    # 使用 asyncio.run 运行异步主循环
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n[退出] 用户中断了程序执行。")