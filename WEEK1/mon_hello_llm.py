import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. Tải các biến môi trường từ file .env vào hệ thống
load_dotenv()

# 2. Lấy API Key ra bằng lệnh os.getenv
# Cách này giúp code của bạn sạch sẽ, không lộ mã trực tiếp
api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", # Thêm dòng này để trỏ sang Groq
    api_key=api_key         # Lấy đúng tên biến trong file .env
)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",                  # Đổi tên model của Groq
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=10
)

print(f"Groq OK rùi nha: {response.choices[0].message.content}")
