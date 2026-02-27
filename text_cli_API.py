import os
import time
import logging
import requests
import jieba
import concurrent.futures
from typing import List, Dict
from pathlib import Path
import functools

# ================= 配置区域 =================
# 使用 Chat Completion API 端点
OLLAMA_API_URL = "http://open-webui-ollama.open-webui:11434/api/chat"
MODEL_NAME = "qwen3-coder:30b"


# Chunking (分段策略) 配置
# MAX_CTX = 32000
# 为模型输出预留约 12000 Token，单次切片最大上限为 20000 Token
CHUNK_MAX_TOKENS = 4000
CHUNK_OVERLAP = 400

# 禁用 jieba 的默认日志输出，保持 CLI 清洁
jieba.setLogLevel(logging.INFO)

# 初始化报告存储目录
BASE_DIR = Path("./reports")
BASE_DIR.mkdir(parents=True, exist_ok=True)

#调试代码
def timetest(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"函数'{func.__name__}'耗时："+str(end-start))
        return result
    return wrapper

# ================= API 交互与异常处理 =================
@timetest
def call_ollama_chat(system_prompt: str, user_prompt: str, retries: int = 3, timeout:int =600) -> str:
    """
    调用 Ollama Chat Completion API [超时控制、网络波动重试与频率限制处理]
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    print("\n"+str(payload))
    backoff = 2  # 初始退避时间
    print("线程开始处理")

    for attempt in range(retries):
        try:
            # 大模型处理长文本耗时较长，Timeout 设置为 600 秒
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)

            # 频率限制 (Rate Limiting)
            if response.status_code == 429:
                print(f"  [警告] 触发 API 频率限制 (429)，{backoff}秒后重试...")
                time.sleep(backoff)
                backoff *= 2
                continue

            response.raise_for_status()
            data = response.json()

            # 解析 Chat Completion 的返回格式
            print(f"\n《《《data》》》:\n{data}")
            return data.get('message', {}).get('content', '').strip()

        except requests.exceptions.Timeout:
            print(f"  [错误] API 请求超时 (Timeout)。尝试 {attempt + 1}/{retries}...")
        except requests.exceptions.ConnectionError:
            print(f"  [错误] 网络连接失败，请检查 Ollama 服务。尝试 {attempt + 1}/{retries}...")
        except requests.exceptions.RequestException as e:
            print(f"  [错误] API 调用异常: {e}。尝试 {attempt + 1}/{retries}...")

        time.sleep(backoff)
        backoff *= 2

    return "【API 请求失败，无法生成结果。】"


# ================= 上下文超长切片管理 =================
@timetest
def chunk_text(text: str) -> List[str]:
    """
    分段滚动处理 (Chunking & Sliding Window):
    利用 jieba 分词估算 Token 数，超过限制则进行带重叠片段的切分。
    """
    # 词法切分估算 Token
    words = list(jieba.cut(text))
    total_tokens = len(words)

    if total_tokens <= CHUNK_MAX_TOKENS:
        return [text]

    print(f"  [信息] 文本总 token 估算为 {total_tokens}，超出单次处理限制，启动分段滚动处理策略...")
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + CHUNK_MAX_TOKENS, total_tokens)
        chunk = "".join(words[start:end])
        chunks.append(chunk)
        if end == total_tokens:
            break
        # 滑动窗口：向后退回 overlap 长度，保证段落上下文连贯性
        start += (CHUNK_MAX_TOKENS - CHUNK_OVERLAP)

    return chunks


# ================= 核心分析逻辑 =================
@timetest
def extract_features(text: str) -> Dict[str, str]:
    """多线程对单一片段并发提取三大基础特征"""
    sys_prompt = "你是一个专业的数据处理与文本智能分析专家。"

    p_summary = f"请对以下文本进行结构化的核心摘要提取，语言需精炼，只输出摘要，不要输出原文：{text}"
    p_sentiment = f"请分析以下文本的情感倾向（正面/负面/中性），并给出简明扼要的分析理由，只输出情感倾向及理由，不要输出原文：{text}"
    p_keywords = f"请提取以下文本中最重要的 5-10 个关键词，使用逗号分隔输出，只输出关键词，不要输出原文：{text}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_sum = executor.submit(call_ollama_chat, sys_prompt, p_summary)
        f_sen = executor.submit(call_ollama_chat, sys_prompt, p_sentiment)
        f_kwd = executor.submit(call_ollama_chat, sys_prompt, p_keywords)

        # 调试速度代码
        task1ing = 0
        task2ing = 0
        task3ing = 0
        while (1):
            if task1ing == 0 :
                if f_sum.done():
                    print("单一片段summary完成")
                    task1ing = 1
            if task2ing == 0:
                if f_sen.done():
                    print("单一片段sensitive完成")
                    task2ing = 1
            if task3ing == 0:
                if f_kwd.done():
                    print("单一片段keywords完成")
                    task3ing = 1
            if task1ing and task2ing and task3ing:
                break
        # 调试速度代码结束
        return {
            "summary": f_sum.result(),
            "sentiment": f_sen.result(),
            "keywords": f_kwd.result()
        }

@timetest
def process_single_document(text: str, index: int) -> Dict[str, str]:
    """
    处理单个文档输入（集成超长文 Map-Reduce 合并逻辑）
    """
    print(f"[*] 开始分析文本档 {index}...")
    chunks = chunk_text(text)

    # 短文本直接处理
    if len(chunks) == 1:
        res = extract_features(chunks[0])
        print(f"[+] 文本档 {index} 分析完成。")
        return res


    # 长文本 Map-Reduce 处理
    print(f"  [信息] 文本档 {index} 被切分为 {len(chunks)} 个片段，正在并行处理各片段...")
    chunk_results = []
    for i, chunk in enumerate(chunks):
        chunk_results.append(extract_features(chunk))

    print(f"  [信息] 文本档 {index} 各片段处理完毕，启动全局 Reduce 结果聚合...")
    sys_prompt = "你是一个专业的文本处理专家，负责融合并汇总局部信息。"

    agg_sum = "综合以下多个文本片段的摘要，生成一个连贯且完整的全局总摘要，只输出全局总摘要：" + "\n---\n".join(
        [r["summary"] for r in chunk_results])
    agg_sen = "综合以下对同一文章不同段落的情感分析，给出一个整体的全局情感倾向及总结理由，只输出全局情感倾向和理由：" + "\n---\n".join(
        [r["sentiment"] for r in chunk_results])
    agg_kwd = "综合以下关键词列表，去重并提取出最具代表性的 10 个核心关键词（仅用逗号分隔），只输出关键词：" + "\n---\n".join(
        [r["keywords"] for r in chunk_results])

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_sum = executor.submit(call_ollama_chat, sys_prompt, agg_sum)
        f_sen = executor.submit(call_ollama_chat, sys_prompt, agg_sen)
        f_kwd = executor.submit(call_ollama_chat, sys_prompt, agg_kwd)

        # 调试速度代码
        task1ing = 0
        task2ing = 0
        task3ing = 0
        while (1):
            if task1ing == 0:
                if f_sum.done():
                    print("长文本 Map-Reduce summary完成")
                    task1ing = 1
            if task2ing == 0:
                if f_sen.done():
                    print("长文本 Map-Reduce sensitive完成")
                    task2ing = 1
            if task3ing == 0:
                if f_kwd.done():
                    print("长文本 Map-Reduce keywords完成")
                    task3ing = 1
            if task1ing and task2ing and task3ing:
                break
        # 调试速度代码结束

        res = {
            "summary": f_sum.result(),
            "sentiment": f_sen.result(),
            "keywords": f_kwd.result()
        }
    print(f"[+] 文本档 {index} 分段汇总分析完成。")
    return res

@timetest
def generate_comparison(results: List[Dict[str, str]]) -> str:
    """多文档对比分析"""
    print("[*] 正在执行多文本交叉对比分析...")
    sys_prompt = "你是一个顶级数据分析专家。请生成包含'核心差异'、'主题共性'以及'综合总结'三个模块的结构化对比 Markdown 报告。"

    user_prompt = "以下是对多个独立文本的分析结果，请自动汇总这些文本的差异与共性，生成对比报告：\n"
    for i, r in enumerate(results):
        user_prompt += f"### 文本 {i + 1} 分析\n- **摘要**: {r['summary']}\n- **情感**: {r['sentiment']}\n- **关键词**: {r['keywords']}\n\n"

    return call_ollama_chat(sys_prompt, user_prompt,3,600)


# ================= 输入过滤与清理 =================
def sanitize_input(text: str) -> str:
    """过滤控制字符和非法输入"""
    if not text:
        return ""
    # 简单的非法字符过滤（去除无法打印的控制字符，保留换行）
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch in ['\n', '\r', '\t'])
    return cleaned.strip()


# ================= 业务流管理 =================
def create_report():
    print("\n" + "=" * 40)
    print("           [ 新建报告 ]")
    print("=" * 40)

    report_name = input("请输入报告名称: ").strip()
    if not report_name:
        print("[拦截] 报告名称不能为空！")
        return
    report_dir = Path(f"./reports/{report_name}")
    report_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    print("\n请提供要分析的资料内容（可多次输入）。完成所有输入后，请按 '3' 开始分析。")
    while True:
        print("\n选择输入源:  1. 纯文本  |  2. 文本文件路径  |  3. [结束输入，开始分析]")
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
            print("当前.路径为： "+str(Path.cwd()))
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

    print(f"\n[*] 开始流水线作业，处理 {len(inputs)} 份资料 (并发模式)...")

    # 并行处理所有文本
    results = [None] * len(inputs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(inputs))) as executor:
        future_to_idx = {
            executor.submit(process_single_document, text, i + 1): i for i, text in enumerate(inputs)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
                md_line = [
                    f"###{report_name}的文档{idx + 1}智能分析报告",
                    f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    "\n---"
                    f"\n## 📑  文本摘要\n{results[idx]['summary']}",
                    f"\n## 🎭  情感倾向\n{results[idx]['sentiment']}",
                    f"\n## 🔑  核心关键词\n{results[idx]['keywords']}",
                    "\n---"
                ]
                # 保存单个结果
                single_report = "\n".join(md_line)

                file_path = report_dir / f"资料{idx+1}报告.md"
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(single_report)
                    print(f"\n[✔️ ] {idx+1}报告生成成功！\n保存位置: {file_path.absolute()}")
                except Exception as e:
                    print(f"\n[❌ ] 保存{idx+1}报告失败: {e}")

            except Exception as e:
                print(f"[致命异常] 处理文本档 {idx + 1} 时出错: {e}")
                results[idx] = {"summary": "处理失败", "sentiment": "处理失败", "keywords": "处理失败"}



    # 如果具有2个及以上的独立输入，触发对比分析进阶功能
    if len(inputs) >= 2:
        # 构建 Markdown
        md_lines = [
            f"# 智能分析报告：{report_name}",
            f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "\n---"
        ]

        # 基础分析合并
        for i, res in enumerate(results):
            md_lines.extend([
                f"\n## 资料 {i + 1} 分析结果",
                f"\n# 📑  文本摘要\n{res['summary']}",
                f"\n# 🎭  情感倾向\n{res['sentiment']}",
                f"\n# 🔑  核心关键词\n{res['keywords']}",
                "\n---"
            ])
        md_lines.append(f"\n## ⚖️ {report_name}多资料深度对比分析")
        comparison_res = generate_comparison(results)
        md_lines.append(comparison_res)

        summary_report = "\n".join(md_lines)

        # 保存结果
        files_path = report_dir / f"{report_name}汇总报告.md"
        try:
            with open(files_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            print(f"\n[✔️ ] 汇总报告生成成功！\n保存位置: {files_path.absolute()}")
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
        print(f" {i + 1}. {f.stem} (大小: {f.stat().st_size} 字节)")

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


# ================= 程序入口 =================
#管理报告界面
def main():
    while True:
        print("\n" + "#" * 45)
        print(" 文本智能分析与报告助手 (Ollama API 版)")
        print("#" * 45)
        print("  1. 新建分析报告")
        print("  2. 查看历史报告")
        print("  3. 退出系统")
        print("-" * 45)

        choice = input("请选择您的操作 (1/2/3): ").strip()

        if choice == '1':
            create_report()
        elif choice == '2':
            view_history()
        elif choice == '3':
            print("感谢使用，系统退出。")
            break
        else:
            print("[拦截] 无效输入，请重新选择。")


if __name__ == "__main__":
    main()