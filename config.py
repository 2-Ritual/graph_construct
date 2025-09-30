import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "qwen-turbo"
EMBED_MODEL = "text-embedding-v4" # "text-embedding-async-v1"

if not DASHSCOPE_API_KEY:
    raise ValueError("错误：环境变量 DASHSCOPE_API_KEY 未设置。请设置您的API密钥。")