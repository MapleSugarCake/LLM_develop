import json
import requests
from typing import List, Dict, Any, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def _handle_stream_response(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """处理流式响应，逐行解析JSON并yield每个chunk"""
        try:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    # 检查流内错误
                    if chunk.get("status") == "error":
                        raise RuntimeError(f"Stream error: {chunk.get('error', 'Unknown error')}")
                    yield chunk
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode JSON from stream: {e}")
        finally:
            response.close()

    def generate(
        self,
        model: str,
        prompt: str,
        context: Optional[List[int]] = None,
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        调用 /api/generate 端点
        
        Returns:
            包含完整响应和上下文的字典
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        if context is not None:
            payload["context"] = context
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        try:
            response = self.session.post(url, json=payload, stream=stream)
            response.raise_for_status()
            
            full_response = ""
            final_context = None
            stats = {}
            
            if stream:
                for chunk in self._handle_stream_response(response):
                    if "response" in chunk:
                        full_response += chunk["response"]
                    if chunk.get("done"):
                        final_context = chunk.get("context")
                        stats = {k: v for k, v in chunk.items() if k not in ["model", "created_at", "response", "done", "context"]}
                return {
                    "response": full_response,
                    "context": final_context,
                    "stats": stats
                }
            else:
                result = response.json()
                return {
                    "response": result.get("response", ""),
                    "context": result.get("context"),
                    "stats": {k: v for k, v in result.items() if k not in ["model", "created_at", "response", "done", "context"]}
                }
                
        except requests.RequestException as e:
            raise RuntimeError(f"HTTP request failed: {e}")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        调用 /api/chat 端点
        
        Returns:
            包含完整响应和消息历史的字典
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        try:
            response = self.session.post(url, json=payload, stream=stream)
            response.raise_for_status()
            
            full_response = ""
            stats = {}
            
            if stream:
                for chunk in self._handle_stream_response(response):
                    if "message" in chunk and "content" in chunk["message"]:
                        full_response += chunk["message"]["content"]
                    if chunk.get("done"):
                        stats = {k: v for k, v in chunk.items() if k not in ["model", "created_at", "message", "done"]}
                # 构建完整消息历史
                updated_messages = messages + [{"role": "assistant", "content": full_response}]
                return {
                    "response": full_response,
                    "messages": updated_messages,
                    "stats": stats
                }
            else:
                result = response.json()
                assistant_message = result.get("message", {})
                updated_messages = messages + [{"role": "assistant", "content": assistant_message.get("content", "")}]
                return {
                    "response": assistant_message.get("content", ""),
                    "messages": updated_messages,
                    "stats": {k: v for k, v in result.items() if k not in ["model", "created_at", "message", "done"]}
                }
                
        except requests.RequestException as e:
            raise RuntimeError(f"HTTP request failed: {e}")

    def multi_request(
        self,
        requests_list: List[Dict[str, Any]],
        max_workers: int = 5
    ) -> List[Dict[str, Any]]:
        """
        并发执行多个请求
        
        Args:
            requests_list: 请求参数列表，每个元素包含 'type' ('generate' or 'chat') 和对应参数
            max_workers: 最大并发线程数
            
        Returns:
            结果列表
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for req in requests_list:
                req_type = req.pop('type')
                if req_type == 'generate':
                    futures.append(executor.submit(self.generate, **req))
                elif req_type == 'chat':
                    futures.append(executor.submit(self.chat, **req))
                else:
                    raise ValueError(f"Unknown request type: {req_type}")
            
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"error": str(e)})
                    
        return results


# 使用示例
if __name__ == "__main__":
    client = OllamaClient()
    
    # 测试 generate
    try:
        result = client.generate(
            model="llama3.2",
            prompt="Why is the sky blue?",
            options={"num_thread": 8}
        )
        print("Generate result:", result["response"][:100] + "...")
        print("Context length:", len(result["context"]) if result["context"] else 0)
    except Exception as e:
        print("Generate error:", e)
    
    # 测试 chat
    try:
        messages = [{"role": "user", "content": "Hello!"}]
        result = client.chat(
            model="llama3.2",
            messages=messages,
            options={"num_thread": 8}
        )
        print("Chat result:", result["response"][:100] + "...")
        print("Messages count:", len(result["messages"]))
    except Exception as e:
        print("Chat error:", e)