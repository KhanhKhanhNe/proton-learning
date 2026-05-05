import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# 1. Cấu hình hệ thống
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Sử dụng model 70b để có chất lượng phản hồi tốt nhất
MODEL = "llama-3.3-70b-versatile"

# --- ĐỊNH NGHĨA 10 HÀM PROMPT ---

def p1_summarize(text):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Bạn là trợ lý điều hành cấp cao. Tóm tắt 3 gạch đầu dòng, mỗi dòng < 20 chữ."},
            {"role": "user", "content": text}
        ]
    )
    return res.choices[0].message.content

def p2_sentiment(text):
    res = client.chat.completions.create(
        model=MODEL, response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia UX. Trả về JSON: {sentiment, score, main_reason}"},
            {"role": "user", "content": text}
        ]
    )
    return res.choices[0].message.content

def p3_extract(text):
    res = client.chat.completions.create(
        model=MODEL, response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Bạn là máy trích xuất dữ liệu. Trả về JSON: {sender, deadline, location, actions}"},
            {"role": "user", "content": text}
        ]
    )
    return res.choices[0].message.content

def p4_translate(text):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia dịch thuật Marketing. Dịch thoát ý, tự nhiên, cung cấp 2 phương án."},
            {"role": "user", "content": text}
        ]
    )
    return res.choices[0].message.content

def p5_script(product):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Bạn là Creator triệu view. Viết kịch bản video 15s, chia 2 cột: Hình ảnh | Lời thoại."},
            {"role": "user", "content": product}
        ]
    )
    return res.choices[0].message.content

def p6_email(target):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Bạn là Sales chuyên nghiệp. Viết email < 120 chữ, súc tích, có CTA mạnh."},
            {"role": "user", "content": target}
        ]
    )
    return res.choices[0].message.content

def p7_convert(code):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Bạn là Software Architect. Chuyển code sang JavaScript, giữ nguyên logic, chú thích tiếng Việt."},
            {"role": "user", "content": code}
        ]
    )
    return res.choices[0].message.content

def p8_dummy():
    res = client.chat.completions.create(
        model=MODEL, response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Bạn là QA Engineer. Tạo mảng JSON 5 đối tượng người dùng: {id, name, email, job}"},
            {"role": "user", "content": "Tạo dữ liệu test."}
        ]
    )
    return res.choices[0].message.content

def p9_interview(jd):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Bạn là HR Manager. Soạn 3 câu hỏi phỏng vấn (Kỹ thuật, Tình huống, Văn hóa)."},
            {"role": "user", "content": jd}
        ]
    )
    return res.choices[0].message.content

def p10_eli5(concept):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Bạn là giáo viên dạy trẻ nhỏ. Giải thích khái niệm đơn giản, < 5 câu, dùng ví dụ so sánh."},
            {"role": "user", "content": concept}
        ]
    )
    return res.choices[0].message.content

# --- THỰC THI GỌI ĐỒNG LOẠT 10 PROMPTS ---

def run_all_tests():
    print("🚀 BẮT ĐẦU CHẠY HỆ THỐNG 10 PROMPTS TRÊN GROQ...\n")
    
    tasks = [
        ("Tóm tắt", p1_summarize, "Thị trường chứng khoán hôm nay biến động mạnh, chỉ số VN-Index giảm 10 điểm do áp lực chốt lời từ nhóm cổ phiếu ngân hàng, tuy nhiên khối ngoại vẫn mua ròng."),
        ("Cảm xúc", p2_sentiment, "Giao diện app mới hơi khó dùng, nhưng tính năng chuyển khoản nhanh thì rất tuyệt."),
        ("Trích xuất", p3_extract, "Gửi sếp, em hẹn đối tác tại cà phê Highland lúc 2h chiều mai để ký hợp đồng nhé."),
        ("Dịch thuật", p4_translate, "Think Different"),
        ("Kịch bản", p5_script, "Bàn phím cơ không dây có đèn LED cầu vồng"),
        ("Email Sales", p6_email, "Chào dịch vụ thiết kế website cho chuỗi cửa hàng trà sữa"),
        ("Chuyển Code", p7_convert, "def hello(): print('Chào thế giới')"),
        ("Dữ liệu mẫu", p8_dummy, ""),
        ("Phỏng vấn", p9_interview, "Tuyển lập trình viên ReactJS kinh nghiệm 2 năm"),
        ("Giải thích", p10_eli5, "Điện toán đám mây (Cloud Computing)")
    ]

    for i, (name, func, data) in enumerate(tasks, 1):
        print(f"--- [Prompt {i}] Tác vụ: {name} ---")
        try:
            # Nếu hàm chỉ nhận 1 tham số (hàm 8 không cần tham số nhưng ta để data rỗng)
            result = func(data) if data else func()
            print(result)
        except Exception as e:
            print(f"Lỗi: {e}")
        print("-" * 50 + "\n")
        time.sleep(1) # Nghỉ 1s giữa các lần gọi để tránh chạm giới hạn Rate Limit

if __name__ == "__main__":
    run_all_tests()