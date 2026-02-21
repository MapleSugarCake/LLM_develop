#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本智能分析与报告助手
支持文本摘要、情感分析、关键词提取和对比分析报告生成
"""
import json
import os
import requests
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

class ReportAssistant:
    def __init__(self):
        self.api_url = "http://open-webui-ollama.open-webui:11434/api/generate"
        self.model = "qwen3-coder:30b"
        self.num_ctx = 32000
        self.timeout = 30
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    def call_ollama_api(self, prompt: str, stream: bool = False) -> str:
        """调用Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "num_ctx": self.num_ctx
                }
            }
            
            response = requests.post(
                self.api_url, 
                json=payload, 
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                full_response += chunk["response"]
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                return full_response
            else:
                result = response.json()
                return result.get("response", "")
                
        except requests.exceptions.Timeout:
            raise Exception("API调用超时")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API调用失败: {str(e)}")
        except Exception as e:
            raise Exception(f"API调用异常: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本token数量（简单估算）"""
        # 简单估算：中文字符算2个token，英文单词算1个token
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_words = len(text.split())
        return chinese_chars + english_words
    
    def chunk_text(self, text: str, chunk_size: int = 28000, overlap: int = 2000) -> List[str]:
        """分段滚动处理长文本"""
        if self.estimate_tokens(text) <= chunk_size:
            return [text]
        
        chunks = []
        words = []
        current_word = ""  # 用于累积当前正在处理的英文单词
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 判断是否为中文字符
                # 如果之前有累积的英文单词，先将其加入列表
                if current_word:
                    words.append(current_word)
                    current_word = ""
                # 将当前中文字符作为一个元素加入列表
                words.append(char)
            elif char.isalpha() and char.isascii():  # 判断是否为英文字母 (isascii确保不是其他语言的字母)
                # 将字母添加到当前单词的末尾
                current_word += char
            else:  # 遇到了非字母、非中文的字符 (如空格、标点符号等)
                # 如果之前有累积的英文单词，先将其加入列表
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
        # 循环结束后，检查是否还有未处理的累积单词
        if current_word:
            words.append(current_word)

        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = 1
            if current_tokens + word_tokens > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # 计算重叠部分
                    overlap_words = []
                    overlap_tokens = 0
                    for w in reversed(current_chunk):
                        w_tokens = 1
                        if overlap_tokens + w_tokens <= overlap:
                            overlap_words.insert(0, w)
                            overlap_tokens += w_tokens
                        else:
                            break
                    current_chunk = overlap_words
                    current_tokens = overlap_tokens
            current_chunk.append(word)
            current_tokens += word_tokens
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        # 返回分段结果【划分好的文本段】
        return chunks

    
    def analyze_text_chunk(self, text_chunk: str, analysis_type: str) -> str:
        """分析文本片段"""
        prompts = {
            "summary": "请为以下文本生成简洁的摘要，控制在100字以内：\n\n{text}",
            "sentiment": "请分析以下文本的情感倾向（正面/负面/中性），并简要说明原因：\n\n{text}",
            "keywords": "请提取以下文本的关键词（5-10个），用逗号分隔：\n\n{text}"
        }
        prompt = prompts[analysis_type].format(text=text_chunk)
        return self.call_ollama_api(prompt)
    
    def analyze_text_comprehensive(self, text: str, text_name: str = "") -> Dict[str, str]:
        """对文本进行全面分析"""
        chunks = self.chunk_text(text)
        
        if len(chunks) == 1:
            # 短文本直接分析
            summary = self.analyze_text_chunk(chunks[0], "summary")
            sentiment = self.analyze_text_chunk(chunks[0], "sentiment")
            keywords = self.analyze_text_chunk(chunks[0], "keywords")
        else:
            # 长文本分段分析后汇总
            with ThreadPoolExecutor(max_workers=3) as executor:
                # 并行处理所有片段的三种分析
                futures = []
                for i, chunk in enumerate(chunks):
                    for analysis_type in ["summary", "sentiment", "keywords"]:
                        futures.append(
                            executor.submit(self.analyze_text_chunk, chunk, analysis_type)
                        )
                
                results = [future.result() for future in as_completed(futures)]
                
                # 汇总结果
                summaries = results[::3]  # 每3个结果的第一个是summary
                sentiments = results[1::3]  # 第二个是sentiment  
                keywords_list = results[2::3]  # 第三个是keywords
                
                # 再次调用API汇总
                summary_prompt = f"请将以下多个摘要合并为一个简洁的总体摘要：\n\n{'\n'.join(summaries)}"
                sentiment_prompt = f"请综合分析以下多个情感分析结果，给出整体情感倾向：\n\n{'\n'.join(sentiments)}"
                keywords_prompt = f"请合并以下多组关键词，去除重复，保留最重要的10个关键词：\n\n{', '.join(keywords_list)}"
                
                summary = self.call_ollama_api(summary_prompt)
                sentiment = self.call_ollama_api(sentiment_prompt)
                keywords = self.call_ollama_api(keywords_prompt)
        
        return {
            "name": text_name or "文本",
            "summary": summary.strip(),
            "sentiment": sentiment.strip(),
            "keywords": keywords.strip()
        }
    
    def generate_comparison_report(self, analyses: List[Dict[str, str]], report_name: str) -> str:
        """生成对比分析报告"""
        if len(analyses) == 1:
            # 单文本报告
            analysis = analyses[0]
            report = f"""# {report_name}

## 文本分析报告

### 文本名称
{analysis['name']}

### 摘要
{analysis['summary']}

### 情感分析
{analysis['sentiment']}

### 关键词
{analysis['keywords']}
"""
        else:
            # 多文本对比报告
            comparison_prompt = f"""请基于以下多个文本的分析结果，生成一份结构化的对比分析报告。

分析结果：
{json.dumps(analyses, ensure_ascii=False, indent=2)}

报告要求：
1. 包含总体概述
2. 逐个文本分析（摘要、情感、关键词）
3. 文本间异同点对比
4. 使用Markdown格式
"""
            report_content = self.call_ollama_api(comparison_prompt)
            report = f"# {report_name}\n\n{report_content}"
        
        return report
    
    def create_new_report(self):
        """创建新报告"""
        try:
            # 获取报告名称
            while True:
                report_name = input("请输入报告名称: ").strip()
                if report_name:
                    break
                print("报告名称不能为空，请重新输入。")
            
            # 获取文本输入
            texts = []
            print("请输入文本内容或文件路径（输入'done'结束）:")
            while True:
                user_input = input().strip()
                if user_input.lower() == 'done':
                    break
                if not user_input:
                    print("输入不能为空，请重新输入或输入'done'结束。")
                    continue
                
                # 检查是否为文件路径
                if os.path.isfile(user_input) and user_input.endswith('.txt'):
                    try:
                        with open(user_input, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                        texts.append((os.path.basename(user_input), text_content))
                    except Exception as e:
                        print(f"读取文件失败: {{e}}，将作为普通文本处理")
                        texts.append((f"文本_{{len(texts)+1}}", user_input))
                else:
                    texts.append((f"文本_{{len(texts)+1}}", user_input))
            
            if not texts:
                print("没有有效的文本输入，取消创建报告。")
                return
            
            # 并行分析所有文本
            print("正在分析文本...")
            analyses = []
            with ThreadPoolExecutor(max_workers=min(len(texts), 5)) as executor:
                futures = [
                    executor.submit(self.analyze_text_comprehensive, text, name)
                    for name, text in texts
                ]
                for future in as_completed(futures):
                    analyses.append(future.result())
            
            # 生成对比报告
            print("正在生成报告...")
            report_content = self.generate_comparison_report(analyses, report_name)
            
            # 保存报告
            report_path = self.reports_dir / f"{{report_name}}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"报告已成功保存到: {{report_path}}")
            
        except KeyboardInterrupt:
            print("\n操作已取消。")
        except Exception as e:
            print(f"创建报告失败: {{e}}")
    
    def view_history_reports(self):
        """查看历史报告"""
        report_files = list(self.reports_dir.glob("*.md"))
        if not report_files:
            print("暂无历史报告。")
            return
        
        print("历史报告列表:")
        for i, report_file in enumerate(report_files, 1):
            print(f"{{i}}. {{report_file.stem}}")
        
        try:
            choice = input("请选择要查看的报告编号（输入0返回主菜单）: ").strip()
            if choice == '0':
                return
            
            index = int(choice) - 1
            if 0 <= index < len(report_files):
                with open(report_files[index], 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"\n{{'='*50}}")
                print(content)
                print(f"{{'='*50}}")
            else:
                print("无效的编号。")
        except ValueError:
            print("请输入有效的数字。")
        except Exception as e:
            print(f"查看报告失败: {{e}}")
    
    def main_menu(self):
        """主菜单"""
        while True:
            print("\n" + "="*30)
            print("文本智能分析与报告助手")
            print("="*30)
            print("1. 新建报告")
            print("2. 历史报告")  
            print("3. 退出")
            print("-"*30)
            
            choice = input("请选择操作 (1-3): ").strip()
            
            if choice == '1':
                self.create_new_report()
            elif choice == '2':
                self.view_history_reports()
            elif choice == '3':
                print("感谢使用，再见！")
                break
            else:
                print("无效选择，请输入1-3。")


def main():
    """主函数"""
    assistant = ReportAssistant()
    assistant.main_menu()


if __name__ == "__main__":
    main()
