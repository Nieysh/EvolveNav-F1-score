import os
import dashscope

messages = [
    {
        "role": "user",
        "content": [
            {"image": "/mnt/data/Users/nieyunshuang/NaviLLM/1734422995658.png"},
            {"image": "/mnt/data/Users/nieyunshuang/NaviLLM/1734422995658.png"},
            {"text": "What is the difference between <img> and <img>?"}
        ]
    }
]

response = dashscope.MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # api_key=os.getenv('DASHSCOPE_API_KEY'),
    api_key = "sk-0b3f11b24c96444d840cf9ca7199d1ce",
    model='qwen-vl-max-latest',
    messages=messages
)

print(response.output.choices[0].message.content[0]["text"])