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
import json
from typing import Any
import re

# ================================== 配置区域 ==================================
OLLAMA_API_URL = "http://open-webui-ollama.open-webui:11434"
MODEL_NAME = "qwen3-coder:30b"
#MODEL_NAME = "gpt-oss:20b"

# Chunking (分段策略) 配置
MAX_CTX = 32768
# 单次切片最大上限为 8192 Token
CHUNK_MAX_TOKENS = 8192
CHUNK_OVERLAP = 300

# 全局 API 最大并发请求限制
# 根据显存大小和模型并发能力设置，30b 模型建议设置 2~5
MAX_API_CONCURRENCY = 3
_api_semaphore = None    # 信号量对象（必须在事件循环启动后初始化）

# 禁用 jieba 的默认日志输出，保持 CLI 清洁
jieba.setLogLevel(logging.INFO)

# 初始化报告存储目录
BASE_DIR = Path("./reports")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 新增：提示工程配置 ====================
ENABLE_COT = True  # 【交互开关】启用 Chain-of-Thought 推理（两步走：先 reasoning，再 answer）
FEW_SHOT_ENABLED = True  # 启用 Few-Shot 示例注入

# Few-Shot 示例池（按任务类型组织，便于扩展）
FEW_SHOT_EXAMPLES = {
    "analysis": [
        {
            "inputs": {"text": "Qwen3 在编码任务上表现卓越，但 30B 参数对显存提出挑战。"},
            "outputs": {
                "summary": "Qwen3 编码能力强但显存需求高。",
                "sentiment": "中性：肯定技术进步，指出资源约束。",
                "keywords": "Qwen3,编码能力,显存,参数规模,推理效率"
            }
        }
    ],
    "comparison": [
        {
            "inputs": {
                "doc1_summary": "文档1聚焦人类探索精神，从恐惧走向科学。",
                "doc2_summary": "文档2批判旧社会问题，通过阿Q形象反思人性。"
            },
            "outputs": {
                "differences": ["文档1基调积极，文档2基调批判", "文档1面向未来，文档2反思过去"],
                "commonalities": ["都涉及人类精神状态", "都使用文学手法表达思想"],
                "conclusion": "两篇文档从不同角度探讨人类处境：文档1歌颂探索，文档2批判社会。"
            }
        }
    ]
}

# ================================== 声明式签名系统 ==================================
class Field:
    def __init__(self, desc: str, type_: str = "string"):
        self.desc = desc
        self.type_ = type_  # "string" | "list" | "enum[正面,负面,中性]"

class Signature:
    def __init__(self, inputs: Dict[str, Field], outputs: Dict[str, Field], instruction: str):
        self.inputs = inputs
        self.outputs = outputs
        self.instruction = instruction

class Example:
    def __init__(self, inputs: Dict[str, str], outputs: Dict[str, Any]):
        self.inputs = inputs
        self.outputs = outputs

class AsyncPromptBuilder:
    """异步兼容的 Prompt 构建器：生成 (system_prompt, user_prompt) 二元组"""
    
    @staticmethod
    def _format_output_schema(outputs: Dict[str, Field], cot: bool) -> Dict:
        schema = {}
        if cot:
            schema["reasoning"] = "[string] 逐步推理过程，用中文简述关键逻辑链"
        for k, v in outputs.items():
            type_desc = {
                "string": "str",
                "list": "List[str]",
                "enum[正面,负面,中性]": "Literal['正面','负面','中性']"
            }.get(v.type_, "str")
            schema[k] = f"[{type_desc}] {v.desc}"
        return schema

    @staticmethod
    def build(signature: Signature, inputs: Dict[str, str], 
              few_shots: List[Example] = None, cot: bool = False) -> tuple[str, str]:
        # === System Prompt: 角色 + 约束 + JSON Schema ===
        output_schema = AsyncPromptBuilder._format_output_schema(signature.outputs, cot)
        
        system_prompt = (
            f"你是一名{signature.instruction}的专业引擎。\n"
            "【核心约束】\n"
            "1. 输出必须是严格合法的 JSON 对象，禁止任何额外文本、Markdown 标记或解释；\n"
            "2. 字段集合必须与签名完全一致，禁止增删、重命名或嵌套无关字段；\n"
            "3. 若启用 CoT，reasoning 字段必须先生成，answer 字段必须严格符合下方 Schema；\n"
            "4. 中文字符保留 UTF-8 编码，避免 Unicode 转义；\n"
            "5. 列表字段使用标准 JSON 数组格式，如 [\"kw1\", \"kw2\"]。\n"
            f"【输出 JSON Schema】\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
        )
        
        # === Few-Shot Examples ===
        user_prompt = ""
        if few_shots and FEW_SHOT_ENABLED:
            user_prompt += "【示例】\n"
            for i, ex in enumerate(few_shots, 1):
                user_prompt += f"示例{i}输入:\n{json.dumps(ex.inputs, ensure_ascii=False)}\n"
                user_prompt += f"示例{i}输出:\n{json.dumps(ex.outputs, ensure_ascii=False)}\n\n"
        
        # === 当前任务 ===
        user_prompt += f"【当前任务】\n输入:\n{json.dumps(inputs, ensure_ascii=False)}\n输出:"
        return system_prompt, user_prompt

class AsyncSignatureModule:
    """异步签名模块：封装 Prompt 构建 调用"""
    def __init__(self, signature: Signature, few_shots: List[Example] = None):
        self.signature = signature
        self.few_shots = few_shots or []
    
    async def __call__(self, inputs: Dict[str, str], cot: bool = False) -> Dict[str, Any]:
        system_prompt, user_prompt = AsyncPromptBuilder.build(
            self.signature, inputs, self.few_shots, cot
        )
        
        # 复用原有 call_ollama_chat（带重试/信号量/超时）
        raw = await call_ollama_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retries=3, timeout=120
        )
        
        # === 两步解析：CoT 模式 vs 直出模式 ===
        try:
            result = json.loads(raw)
            if cot and "reasoning" in result and "answer" in result:
                # CoT 模式：提取 answer 部分
                answer = result["answer"]
                # 可选：记录 reasoning 用于调试（不污染业务输出）
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"CoT 推理: {result['reasoning'][:300]}...")
                return answer
            elif cot and "reasoning" in result:
                # 降级：若 model 未严格遵循 schema，尝试从 reasoning 后提取 JSON
                match = re.search(r'\{.*\}', raw[raw.rfind("}")+1:], re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            return result
        except json.JSONDecodeError:
            # 【修复】在全文中搜索首个完整 JSON 块，而非切片后搜索
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    candidate = json.loads(match.group(0))
                    if cot and "answer" in candidate:
                        return candidate["answer"]
                    return candidate
                except:
                    pass
            return {"error": "JSON_PARSE_FAILED", "raw_preview": raw[:200]}

# ================================== 调试代码 (兼容同步与异步) ==================================
#
# def timetest(func):
#     """
#     智能装饰器，自动判断函数是同步还是异步，并计算耗时
#     """
#     if asyncio.iscoroutinefunction(func):
#         @functools.wraps(func)
#         async def async_wrapper(*args, **kwargs):
#             start = time.time()
#             result = await func(*args, **kwargs)
#             end = time.time()
#             print(f"函数 '{func.__name__}' 耗时：{end - start:.4f} 秒")
#             return result
#         return async_wrapper
#     else:
#         @functools.wraps(func)
#         def sync_wrapper(*args, **kwargs):
#             start = time.time()
#             result = func(*args, **kwargs)
#             end = time.time()
#             print(f"函数 '{func.__name__}' 耗时：{end - start:.4f} 秒")
#             return result
#         return sync_wrapper


# ================================== API 交互与异常处理 ==================================
# @timetest
async def call_ollama_chat(system_prompt: str, user_prompt: str, retries: int = 3, timeout: int = 100) -> str:
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
            "num_ctx": MAX_CTX,        # 最大token数，它会自适应的
            "temperature": 0.2,        # 保持严谨，降低创造力
            "top_p": 0.4,              # 提高输出内容的确定性
            "num_predict": -1,         # 允许生成较长的完整报告
            "repeat_penalty": 1.15,    # 略微提高重复惩罚，防止关键词重复
            "format": "json"
        }

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
# @timetest
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

# ==================== 任务签名定义 ====================
analysis_sig = Signature(
    inputs={"text": Field("待分析的文本片段")},
    outputs={
        "summary": Field("精炼的核心摘要，聚焦主旨论点与关键事实"),
        "sentiment": Field("情感极性判断", type_="enum[正面,负面,中性]"),
        "keywords": Field("5-10 个核心术语，按语义重要性降序排列", type_="list")
    },
    instruction="文本语义理解与结构化抽取，擅长在少样本设定下完成摘要生成、情感极性判断与关键术语识别"
)

comparison_sig = Signature(
    inputs={
        "doc1_summary": Field("文档 1 的摘要"),
        "doc2_summary": Field("文档 2 的摘要")
    },
    outputs={
        "differences": Field("文档差异列表", type_="list"),
        "commonalities": Field("共享主题列表", type_="list"),
        "conclusion": Field("整体对比结论，基于差异与共性的加权分析")
    },
    instruction="跨文档语义合成，擅长从多源异构文本中识别模式、冲突与共识"
)

# 初始化模块（注入 Few-Shot）
analysis_module = AsyncSignatureModule(
    analysis_sig, 
    few_shots=[Example(**ex) for ex in FEW_SHOT_EXAMPLES.get("analysis", [])]
)
comparison_module = AsyncSignatureModule(
    comparison_sig,
    few_shots=[Example(**ex) for ex in FEW_SHOT_EXAMPLES.get("comparison", [])]
)

# ================================== 核心分析逻辑 ==================================
# @timetest  # 可选：保留调试装饰器
async def extract_features(text: str) -> Dict[str, str]:
    """利用异步机制 + 声明式签名 + 可选 CoT 提取三大基础特征"""
    
    # 调用签名模块（CoT 开关由全局配置控制）
    result = await analysis_module(
        inputs={"text": text},
        cot=ENABLE_COT
    )
    
    # 后处理：确保字段存在
    return {
        "summary": result.get("summary", ""),
        "sentiment": result.get("sentiment", ""),
        "keywords": result.get("keywords", [])
    }


# @timetest
async def process_single_document(text: str, index: int) -> Dict[str, str]:
    """
    处理单个文档输入（集成超长文 Map-Reduce 合并逻辑）
    """
    print(f"[*] 开始分析文本档 {index}...")
    chunks = chunk_text(text)

    # 短文本直接处理
    if len(chunks) == 1:
        res = await analysis_module(inputs={"text": chunks[0]}, cot=ENABLE_COT)
        print(f"[+] 文本档 {index} 分析完成。")
        # 防御性：确保字段存在
        return {
            "summary": res.get("summary", ""),
            "sentiment": res.get("sentiment", ""),
            "keywords": res.get("keywords", [])
        }

    # 长文本 Map-Reduce 处理：异步并发处理各片段
    print(f"  [信息] 文本档 {index} 被切分为 {len(chunks)} 个片段，正在并行处理各片段...")

    tasks = [analysis_module(inputs={"text": chunk}, cot=ENABLE_COT) for chunk in chunks]
    chunk_results = await asyncio.gather(*tasks)

    print(f"  [信息] 文本档 {index} 各片段处理完毕，启动全局 Reduce 结果聚合...")
    
    # 数据拼接
    chunk_summaries = "\n---\n".join([r.get("summary", "") for r in chunk_results])
    chunk_sentiments = "\n---\n".join([r.get("sentiment", "") for r in chunk_results])
    chunk_keywords = "\n---\n".join([str(r.get("keywords", [])) for r in chunk_results])

    # 注意：此处复用 analysis_module，模型会根据输入内容自动适配（摘要输入->生成全局摘要）
    # 为提升精度，理论上可定义 agg_sig，但为保持架构轻量，暂复用 analysis_sig
    
    async def aggregate_task(input_text: str) -> Dict:
        return await analysis_module(inputs={"text": input_text}, cot=ENABLE_COT)

    # 并发执行三大聚合任务
    f_sum_res, f_sen_res, f_kwd_res = await asyncio.gather(
        aggregate_task(chunk_summaries),
        aggregate_task(chunk_sentiments),
        aggregate_task(chunk_keywords)
    )

    res = {
        "summary": f_sum_res.get("summary", ""),
        "sentiment": f_sen_res.get("sentiment", ""),
        "keywords": f_kwd_res.get("keywords", [])
    }

    print(f"[+] 文本档 {index} 分段汇总分析完成。")
    return res


# @timetest
async def generate_comparison(results: List[Dict[str, str]], text_name: List[str]) -> str:
    """多文档对比分析：基于声明式签名 + 可选 CoT"""
    print("[*] 正在执行多文本交叉对比分析...")
    
    # 构造输入
    compare_inputs = {
        "doc1_summary": results[0].get("summary", ""),
        "doc2_summary": results[1].get("summary", "")
    }
    
    # 调用对比签名模块
    compare_result = await comparison_module(
        inputs=compare_inputs,
        cot=ENABLE_COT
    )
    
    # 渲染 Markdown 报告（保持原有格式兼容）
    md = "# ⚖️ 多资料深度对比分析\n\n"
    
    # 解析 differences/commonalities（支持 list 或 string 兼容）
    def parse_list_field(val):
        if isinstance(val, list): return val
        if isinstance(val, str):
            try: return json.loads(val)
            except: return [val] if val.strip() else []
        return []
    
    diffs = parse_list_field(compare_result.get("differences", []))
    comms = parse_list_field(compare_result.get("commonalities", []))
    
    md += "## 🔍 共同点\n" + "\n".join(f"- {c}" for c in comms) + "\n\n"
    md += "## ⚠️ 差异点\n" + "\n".join(f"- {d}" for d in diffs) + "\n\n"
    md += f"## 💡 结论\n> {compare_result.get('conclusion', '')}\n"
    
    return md


# ================= 输入过滤与清理 =================
def sanitize_input(text: str) -> str:
    """过滤控制字符和非法输入"""
    if not text:
        return ""
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch in ['\n', '\r', '\t'])
    return cleaned.strip()


# ================= 核心业务操作流程管理 ===================
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
    text_name=[]
    print("\n请提供要分析的资料内容（可多次输入）。完成所有输入后，请按 '3' 开始分析。")
    while True:
        print("\n选择输入源:  1. 纯文本  |  2. 文本文件路径  |  3. 结束输入，开始分析 ")
        choice = input("操作 >> ").strip()

        if choice == '1':
            text = input("请输入纯文本内容: ")
            text = sanitize_input(text)
            if text:
                inputs.append(text)
                text_name.append(f"文本{len(inputs)}")
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
                            text_name.append(f.name)
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

    # 定义包含持久化、生成的包装任务
    async def task_wrapper(index: int, doc_text: str):
        try:
            res = await process_single_document(doc_text, index)
            # =================单文件保存=================
            md_line = [
                f"# {report_name}的文档{text_name[index-1]}智能分析报告",
                f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "\n---",
                f"\n### 📑  文本摘要\n{res['summary']}",
                f"\n### 🎭  情感倾向\n{res['sentiment']}",
                f"\n### 🔑  核心关键词\n{','.join(res['keywords'])}",
                "\n---"
            ]
            single_report = "\n".join(md_line)
            file_path = report_dir / f"文档{text_name[index-1]}报告.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(single_report)
            print(f"\n[✔️ ] {text_name[index-1]}报告生成成功！\n保存位置: {file_path.absolute()}")
            return index, res

        except Exception as e:
            print(f"[致命异常] 处理文本档 {index} 时出错: {e}")
            return index, {"summary": "处理失败", "sentiment": "处理失败", "keywords": "处理失败"}

    #执行任务
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
                f"\n## 资料 {text_name[i]} 分析结果",
                f"\n### 📑  文本摘要\n{res['summary']}",
                f"\n### 🎭  情感倾向\n{res['sentiment']}",
                f"\n### 🔑  核心关键词\n{','.join(res['keywords'])}",
                "\n---"
            ])

        md_lines.append(f"\n# ⚖️ {report_name}的多资料深度对比分析")
        comparison_res = await generate_comparison(results,text_name)
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