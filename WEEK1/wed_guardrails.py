from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Cấu hình Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)
MODEL = "llama-3.3-70b-versatile"

def guardrail_validator(output_text):
    """Kiểm tra định dạng JSON và các trường bắt buộc"""
    try:
        data = json.loads(output_text)
        required_keys = ["sentiment", "score"]
        if all(k in data for k in required_keys):
            return True, data
        return False, f"Thiếu trường: {set(required_keys) - set(data.keys())}"
    except Exception as e:
        return False, "Không phải JSON hợp lệ"

def pipeline_with_retry(user_input, max_retries=3):
    """
    Pipeline: Gọi LLM -> Kiểm tra -> Nếu lỗi thì bắt AI sửa lại (tối đa 3 lần)
    """
    attempt = 0
    messages = [
        {
            "role": "system", 
            "content": "Bạn là máy phân tích. Trả về JSON có key 'sentiment' đánh giá tích cực/tiêu cực và 'score' (0-1)."
        },
        {"role": "user", "content": user_input}
    ]

    while attempt < max_retries:
        print(f"--- Đang thử lần {attempt + 1} ---")
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.9, 
            response_format={"type": "json_object"}
        )
        
        raw_output = response.choices[0].message.content
        is_valid, result = guardrail_validator(raw_output)

        if is_valid:
            return result
        
        # Nếu lỗi, nạp lỗi vào hội thoại để AI tự sửa ở lần thử sau
        print(f"❌ Lỗi: {result}. Đang yêu cầu AI sửa...")
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({"role": "user", "content": f"Kết quả trước bị lỗi: {result}. Hãy sửa lại đúng định dạng JSON."})
        attempt += 1

    return "Thất bại sau nhiều lần thử."

# Chạy thử
print(pipeline_with_retry("Sản phẩm tuyệt vời nhưng giao hàng hơi lâu."))