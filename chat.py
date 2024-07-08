from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# API Key 和目标 URL
API_KEY = "sk-18e59050d4ac46f2989afc944604c259"
TARGET_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'

@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    # 从请求中获取数据
    data = request.json

    # 设置请求头和请求体
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    body = {
        'model': 'qwen-turbo',
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": data['content']
                }
            ]
        },
        "parameters": {
            "result_format": "message"
        }
    }

    # 发送请求到目标 API
    response = requests.post(TARGET_URL, headers=headers, json=body)
    response_data = response.json()

    # 返回目标 API 的响应
    if 'output' in response_data and 'choices' in response_data['output']:
        return jsonify(response_data['output']['choices'][0]['message'])
    else:
        return jsonify({"error": "No valid response received."}), 400

if __name__ == '__main__':
    app.run(port=5001)
