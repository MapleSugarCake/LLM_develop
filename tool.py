import requests

response = requests.get("http://open-webui-ollama.open-webui:11434/api/tags", timeout=3)
print(response.text)
print("\n")
print("*"*40)
print("\n")

for tag in ['qwen3-coder:30b','test-model:latest','llama3.2:latest']:
    payload = {
        "model":tag
    }
    response2 = requests.post("http://open-webui-ollama.open-webui:11434/api/show",json=payload,timeout=10)
    print("\n")
    print("*" * 40)
    print("\n")
    print(response2.text)
    print("\n")
    print("*" * 40)
    print("\n")