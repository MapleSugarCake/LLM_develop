import requests
import time

response2 = requests.post("http://open-webui-ollama.open-webui:11434/api/chat",timeout=300)
response2.raise_for_status()
data=response2.json()
end_time=time.time()
print("\n")
print("*" * 40)
print("\n")
print(data.get('message', {}).get('content', '').strip())
print("\n")
print("*" * 40)
print("\n")
print("time:"+str(end_time-start_time))