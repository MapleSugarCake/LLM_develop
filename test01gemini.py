import os
import time
import logging
import requests
import jieba
import concurrent.futures
from typing import List, Dict
from pathlib import Path

# ================= 配置区域 =================
OLLAMA_API_URL = "http://open-webui-ollama.open-webui:11434/api/generate"
MODEL_NAME = "qwen3-coder:30b"
MAX_CTX = 32000
# 为输出预留空间，最大 Token 设置为 25000，重叠部分设置为 2000
CHUNK_MAX_TOKENS = 25000
CHUNK_OVERLAP = 2000
REPORTS_DIR = Path("./reports")

# 禁用 jieba 的默认日志输出
jieba.setLogLevel(logging.INFO)

# 初始化报告目录
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ================= API 交互与异常处理 =================
def call_ollama(prompt: str, system: str = "你是一个专业的数据处理与文本智能分析专家。", retries: int = 3) -> str:
    """
    调用 Ollama API，具备超时控制、网络波动重试与频率限制处理
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "num_ctx": MAX_CTX
        }
    }

    backoff = 2  # 初始退避时间
    for attempt in range(retries):
        try:
            # 考虑到大模型的分析时间，Timeout 设置为较为宽裕的 300 秒
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)

            # 频率限制 (Rate Limiting)
            if response.status_code == 429:
                print(f"  [警告] 触发 API 频率限制，{backoff}秒后重试...")
                time.sleep(backoff)
                backoff *= 2
                continue

            response.raise_for_status()
            data = response.json()
            return data.get('response', '').strip()

        except requests.exceptions.Timeout:
            print(f"  [错误] API 请求超时 (Timeout)。尝试 {attempt + 1}/{retries}...")
        except requests.exceptions.ConnectionError:
            print(f"  [错误] 网络连接失败，请检查 Ollama 服务({OLLAMA_API_URL})。尝试 {attempt + 1}/{retries}...")
        except requests.exceptions.RequestException as e:
            print(f"  [错误] API 调用异常: {e}。尝试 {attempt + 1}/{retries}...")

        time.sleep(backoff)
        backoff *= 2

    return "【API 请求失败，无法生成结果。】"


# ================= 上下文超长切片管理 =================
def chunk_text(text: str) -> List[str]:
    """
    分段滚动处理 (Chunking & Sliding Window):
    利用 jieba 分词估算 token 数，超过限制则进行带重叠片段的切分。
    """
    words = list(jieba.cut(text))
    total_tokens = len(words)

    if total_tokens <= CHUNK_MAX_TOKENS:
        return [text]

    print(f"  [信息] 文本总 token 估算为 {total_tokens}，超出单次处理限制，启动分段滚动处理策略。")
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + CHUNK_MAX_TOKENS, total_tokens)
        chunk = "".join(words[start:end])
        chunks.append(chunk)
        if end == total_tokens:
            break
        # 滑动窗口：向后退回 overlap 长度，保证段落上下文连贯
        start += (CHUNK_MAX_TOKENS - CHUNK_OVERLAP)

    return chunks


# ================= 核心分析逻辑 =================
def process_tasks_for_text(text: str) -> Dict[str, str]:
    """利用多线程对单一文本并行执行三大基础任务"""
    prompt_summary = f"请对以下文本进行结构化的核心摘要提取，语言需精炼：\n\n{text}"
    prompt_sentiment = f"请分析以下文本的情感倾向（正面/负面/中性），并给出简明扼要的分析理由：\n\n{text}"
    prompt_keywords = f"请提取以下文本中最重要的 5-10 个关键词，使用逗号分隔输出：\n\n{text}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_sum = executor.submit(call_ollama, prompt_summary)
        f_sen = executor.submit(call_ollama, prompt_sentiment)
        f_kwd = executor.submit(call_ollama, prompt_keywords)

        return {
            "summary": f_sum.result(),
            "sentiment": f_sen.result(),
            "keywords": f_kwd.result()
        }


def analyze_single_input(text: str, index: int) -> Dict[str, str]:
    """分析单个输入文本（自带超长文本校验和合并逻辑）"""
    print(f"[*] 开始分析文本 {index}...")
    chunks = chunk_text(text)

    # 场景1：无需分段，直接处理
    if len(chunks) == 1:
        result = process_tasks_for_text(chunks[0])
        print(f"[+] 文本 {index} 分析完成。")
        return result

    # 场景2：需要分段，处理后进行合并 (Map-Reduce 模式)
    print(f"  [信息] 文本 {index} 被分为 {len(chunks)} 个片段，正在分别处理...")
    chunk_results = []
    for i, chunk in enumerate(chunks):
        chunk_results.append(process_tasks_for_text(chunk))

    print(f"  [信息] 文本 {index} 各片段处理完毕，正在进行结果汇总聚合...")

    # 合并 Prompt 构造
    agg_sum_prompt = "请综合以下多个文本片段的摘要，生成一个连贯且完整的全局总摘要：\n" + "\n---\n".join(
        [r["summary"] for r in chunk_results])
    agg_sen_prompt = "请综合以下对同一文章不同段落的情感分析，给出一个整体的全局情感倾向及总结理由：\n" + "\n---\n".join(
        [r["sentiment"] for r in chunk_results])
    agg_kwd_prompt = "请综合以下关键词列表，去重并提取出最具代表性的 10 个核心关键词（逗号分隔）：\n" + "\n---\n".join(
        [r["keywords"] for r in chunk_results])

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_sum = executor.submit(call_ollama, agg_sum_prompt)
        f_sen = executor.submit(call_ollama, agg_sen_prompt)
        f_kwd = executor.submit(call_ollama, agg_kwd_prompt)

        result = {
            "summary": f_sum.result(),
            "sentiment": f_sen.result(),
            "keywords": f_kwd.result()
        }
    print(f"[+] 文本 {index} 分段汇总完成。")
    return result


def compare_texts(results: List[Dict[str, str]]) -> str:
    """进阶方向：多文本对比分析"""
    print("[*] 检测到多个输入文本，正在进行交叉对比分析...")
    prompt = "以下是对多个独立文本的分析结果，请你作为数据分析专家，自动汇总这些文本的差异与共性，并生成结构化的 Markdown 对比分析报告：\n\n"

    for i, r in enumerate(results):
        prompt += f"### 文本 {i + 1} 数据\n- **摘要**: {r['summary']}\n- **情感**: {r['sentiment']}\n- **关键词**: {r['keywords']}\n\n"

    system_prompt = "你是一个顶级数据分析专家。请生成包含'核心差异'、'主题共性'以及'综合总结'三个模块的结构化对比 Markdown 报告。"
    return call_ollama(prompt, system=system_prompt)


# ================= 业务流管理 =================
def create_report():
    print("\n--- 新建报告 ---")
    report_name = input("请输入报告名称: ").strip()
    if not report_name:
        print("[错误] 报告名称不能为空！")
        return

    inputs = []
    print("\n[输入阶段] 请输入要分析的内容（支持纯文本或文件路径）。输入完成后选择 '3' 结束添加。")
    while True:
        choice = input("\n请选择输入类型 (1.手动输入文本  2.输入文本文件路径  3.结束添加并开始分析): ").strip()
        if choice == '1':
            text = input("请输入文本内容: ").strip()
            if text:
                inputs.append(text)
            else:
                print("[拦截] 输入为空，已忽略。")

        elif choice == '2':
            path = input("请输入纯文本文件路径: ").strip()
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            inputs.append(text)
                        else:
                            print("[拦截] 文件内容为空，已忽略。")
                except Exception as e:
                    print(f"[错误] 读取文件失败: {e}")
            else:
                print("[错误] 文件路径不存在或非合法文件。")

        elif choice == '3':
            if not inputs:
                print("[错误] 没有有效的输入文本，无法生成报告。")
                return
            break
        else:
            print("[错误] 无效选项，请重新选择。")

    print(f"\n[*] 收集到 {len(inputs)} 个文本，启动批量处理流水线...")

    # 异步多线程批量处理所有输入的文本
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(inputs))) as executor:
        future_to_index = {executor.submit(analyze_single_input, txt, i + 1): i for i, txt in enumerate(inputs)}

        # 保持顺序结构，预先填充结果数组
        results = [None] * len(inputs)
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[致命错误] 处理文本 {idx + 1} 时发生异常: {e}")
                results[idx] = {"summary": "处理失败", "sentiment": "处理失败", "keywords": "处理失败"}

    # 构建 Markdown 报告
    markdown_lines = [f"# 文本智能分析报告：{report_name}\n"]
    markdown_lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for i, res in enumerate(results):
        markdown_lines.append(f"## 文本 {i + 1} 基础分析")
        markdown_lines.append(f"### 摘要\n{res['summary']}\n")
        markdown_lines.append(f"### 情感倾向\n{res['sentiment']}\n")
        markdown_lines.append(f"### 关键词\n{res['keywords']}\n")

    # 对比分析（大于等于2个文本时）
    if len(inputs) >= 2:
        markdown_lines.append("## 多文本对比分析报告")
        comparison = compare_texts(results)
        markdown_lines.append(comparison)

    final_report = "\n".join(markdown_lines)

    # 保存结果
    file_path = REPORTS_DIR / f"{report_name}.md"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        print(f"\n[成功] 报告已生成并保存在: {file_path.absolute()}")
    except Exception as e:
        print(f"\n[错误] 保存报告失败: {e}")


def view_history():
    print("\n--- 历史报告 ---")
    files = list(REPORTS_DIR.glob("*.md"))
    if not files:
        print("暂无历史报告。")
        return

    print("可用报告列表:")
    for i, f in enumerate(files):
        print(f"{i + 1}. {f.stem}")

    choice = input("请输入要查看的报告编号 (输入 0 返回主菜单): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            try:
                with open(files[idx], 'r', encoding='utf-8') as f:
                    print("\n" + "=" * 50)
                    print(f.read())
                    print("=" * 50 + "\n")
            except Exception as e:
                print(f"[错误] 读取文件失败: {e}")
        elif choice != '0':
            print("[错误] 无效编号。")
    else:
        print("[错误] 请输入数字。")


# ================= 主控制台 =================
def main():
    while True:
        print("\n================================")
        print("  文本智能分析与报告助手 (Ollama)")
        print("================================")
        print("1. 新建报告")
        print("2. 历史报告")
        print("3. 退出")
        print("================================")

        choice = input("请选择操作 (1-3): ").strip()

        if choice == '1':
            create_report()
        elif choice == '2':
            view_history()
        elif choice == '3':
            print("感谢使用，再见！")
            break
        else:
            print("[错误] 无效的输入，请重试。")


if __name__ == "__main__":
    main()