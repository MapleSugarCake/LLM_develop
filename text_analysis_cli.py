

# 配置常量
OLLAMA_API_URL = "http://open-webui-ollama.open-webui:11434/api/generate"
MODEL_NAME = "qwen3-coder:30b"
NUM_CTX = 32768
MAX_RETRIES = 3
SENSITIVE_WORDS = ["敏感词1", "敏感词2"]  # 基本敏感词列表，实际可扩展
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def is_sensitive(text):
    """基本敏感内容检测"""
    for word in SENSITIVE_WORDS:
        if word in text:
            return True
    return False

def chunk_text(text, max_chunk_size=28000, overlap=2800):
    """使用jieba分词进行文本切片（滑动窗口）"""
    if len(text) <= max_chunk_size:
        return [text]
    
    words = list(jieba.cut(text))
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > max_chunk_size and current_chunk:
            chunk_str = ''.join(current_chunk)
            chunks.append(chunk_str)
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_chunk):
                if overlap_len + len(w) > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w)
            current_chunk = overlap_words
            current_length = overlap_len
        
        current_chunk.append(word)
        current_length += len(word)
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks

def call_ollama_api(prompt, task_type, retries=MAX_RETRIES):
    """调用Ollama API，带指数退避重试"""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "options": {
            "num_ctx": NUM_CTX
        },
        "stream": False
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=data, headers=headers, timeout=60)
            if response.status_code == 200:
                result = response.json()
                try:
                    analysis_result = json.loads(result["response"])
                    return analysis_result
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON response", "raw_response": result["response"]}
            elif response.status_code in [429, 503, 504]:
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
                continue
            else:
                return {"error": f"HTTP {{response.status_code}}", "message": response.text}
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < retries - 1:
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
                continue
            else:
                return {"error": "Max retries exceeded", "exception": str(e)}
        except Exception as e:
            return {"error": "Unexpected error", "exception": str(e)}
    
    return {"error": "Max retries exceeded"}

def analyze_text(text, text_id):
    """对单个文本执行三种分析任务"""
    summary_prompt = f"""请对以下文本进行摘要，输出JSON格式：{{"summary": "摘要内容"}}。文本：{text}"""
    sentiment_prompt = f"""请对以下文本进行情感分析（正面/负面/中性），输出JSON格式：{{"sentiment": "情感标签"}}。文本：{text}"""
    keywords_prompt = f"""请提取以下文本的关键词（最多5个），输出JSON格式：{{"keywords": ["关键词1", "关键词2", ...]}}。文本：{text}"""
    
    tasks = [
        ("summary", summary_prompt),
        ("sentiment", sentiment_prompt),
        ("keywords", keywords_prompt)
    ]
    
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {
            executor.submit(call_ollama_api, prompt, task): task 
            for task, prompt in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results[task] = result
            except Exception as e:
                results[task] = {"error": str(e)}
    
    return {
        "text_id": text_id,
        "original_text": text[:100] + "..." if len(text) > 100 else text,
        "analysis": results
    }

def generate_comparison_report(analyses):
    """生成多文本对比分析的Markdown报告"""
    analysis_summary = ""
    for i, analysis in enumerate(analyses):
        summary = analysis["analysis"].get("summary", {}).get("summary", "N/A")
        sentiment = analysis["analysis"].get("sentiment", {}).get("sentiment", "N/A")
        keywords = analysis["analysis"].get("keywords", {}).get("keywords", [])
        analysis_summary += f"### 文本{i+1}\n- 摘要: {summary}\n- 情感: {sentiment}\n- 关键词: {', '.join(keywords)}\n\n"
    
    comparison_prompt = f"""请基于以下多个文本的分析结果，生成一份结构化的Markdown对比报告，突出差异和共同点：
{analysis_summary}
报告格式要求：
# 文本对比分析报告
## 共同点
- ...
## 差异点
- ...
## 总结
- ..."""
    
    result = call_ollama_api(comparison_prompt, "comparison")
    if "error" not in result:
        report_content = result.get("report", "未能生成对比报告")
    else:
        report_content = f"对比报告生成失败: {result.get('error', 'Unknown error')}"
    
    return report_content

def create_new_report():
    """新建报告功能"""
    report_name = input("请输入报告名称: ").strip()
    if not report_name:
        print("报告名称不能为空！")
        return
    
    user_input = input("请输入文本内容或文件路径（txt文件，多个文件用逗号分隔）: ").strip()
    if not user_input:
        print("输入不能为空！")
        return
    
    texts = []
    if ',' in user_input or '\n' in user_input:
        paths = [p.strip() for p in user_input.split(',')]
        for path in paths:
            if os.path.isfile(path) and path.endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                    else:
                        print(f"警告: 文件 {{path}} 为空，已跳过。")
            else:
                texts.append(user_input)
                break
    else:
        if os.path.isfile(user_input) and user_input.endswith('.txt'):
            with open(user_input, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
                else:
                    print("文件内容为空！")
                    return
        else:
            texts.append(user_input)
    
    if not texts:
        print("未获取到有效文本！")
        return
    
    for i, text in enumerate(texts):
        if is_sensitive(text):
            print(f"文本{{i+1}}包含敏感内容，已拒绝处理！")
            return
    
    processed_texts = []
    for text in texts:
        if len(text) > NUM_CTX * 2:
            print(f"警告: 文本长度超过上下文限制，将被截断。建议使用更短的文本。")
        processed_texts.append(text)
    
    analyses = []
    with ThreadPoolExecutor(max_workers=min(5, len(processed_texts))) as executor:
        future_to_index = {
            executor.submit(analyze_text, text, i): i 
            for i, text in enumerate(processed_texts)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                analyses.append(result)
            except Exception as e:
                analyses.append({"text_id": index, "error": str(e)})
    
    comparison_report = None
    if len(processed_texts) >= 2:
        print("正在生成对比分析报告...")
        comparison_report = generate_comparison_report(analyses)
    
    report_data = {
        "report_name": report_name,
        "texts": processed_texts,
        "analyses": analyses,
        "comparison_report": comparison_report
    }
    
    report_path = os.path.join(REPORTS_DIR, f"{{report_name}}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"报告已保存至: {{report_path}}")

def view_history_reports():
    """查看历史报告"""
    reports = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
    if not reports:
        print("暂无历史报告。")
        return
    
    print("历史报告列表:")
    for i, report in enumerate(reports, 1):
        print(f"{{i}}. {{report}}")
    
    choice = input("请选择报告编号: ").strip()
    if not choice.isdigit():
        print("无效输入！")
        return
    
    index = int(choice) - 1
    if index < 0 or index >= len(reports):
        print("编号超出范围！")
        return
    
    report_path = os.path.join(REPORTS_DIR, reports[index])
    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    print("\n=== 报告内容 ===")
    print(json.dumps(report_data, ensure_ascii=False, indent=2))

def main():
    """主菜单"""
    while True:
        print("\n=== 文本智能分析与报告助手 ===")
        print("1. 新建报告")
        print("2. 历史报告")
        print("3. 退出")
        choice = input("请选择操作: ").strip()
        
        if choice == '1':
            create_new_report()
        elif choice == '2':
            view_history_reports()
        elif choice == '3':
            print("退出程序。")
            break
        else:
            print("无效选择，请重新输入。")

if __name__ == "__main__":
    main()
