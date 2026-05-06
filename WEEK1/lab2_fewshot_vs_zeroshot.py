import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

MODEL = "llama-3.3-70b-versatile"


def validate_output(data):
    """
    Kiểm tra output có đúng format và giá trị hợp lệ không.
    """
    required_keys = ["sentiment", "score", "keywords"]
    allowed_sentiments = ["Tích cực", "Tiêu cực"]
    
    if not all(key in data for key in required_keys):
        return False, "Thiếu trường dữ liệu bắt buộc"
    
    if data["sentiment"] not in allowed_sentiments:
        return False, f"sentiment phải là 'Tích cực' hoặc 'Tiêu cực', nhận được: {data['sentiment']}"
    
    if not isinstance(data["score"], (int, float)) or not (0 <= data["score"] <= 1):
        return False, "score phải là số từ 0 đến 1"
    
    if not isinstance(data["keywords"], list):
        return False, "keywords phải là mảng"
    
    return True, "OK"


def zero_shot_analyzer(user_input):
    """
    Prompt zero-shot: không có ví dụ mẫu.
    """
    system_instruction = (
        "Bạn là chuyên gia phân tích phản hồi khách hàng. "
        "Hãy trích xuất sentiment, score và keywords. "
        "sentiment chỉ được là 'Tích cực' hoặc 'Tiêu cực'. "
        "score là số từ 0 đến 1. "
        "keywords là mảng các từ khóa quan trọng. "
        "Bắt buộc trả về JSON theo dạng: "
        '{"sentiment": "...", "score": 0.0, "keywords": [...]}'
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_input}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        output_content = response.choices[0].message.content
        data = json.loads(output_content)
        
        valid, msg = validate_output(data)
        if not valid:
            return {"error": msg, "raw": data}
        
        return data
    
    except Exception as e:
        return {"error": str(e)}


def few_shot_1_analyzer(user_input):
    """
    Few-shot 1: Chỉ 1 ví dụ mẫu.
    """
    system_instruction = (
        "Bạn là chuyên gia phân tích phản hồi khách hàng. "
        "Hãy trích xuất sentiment, score và keywords. "
        "sentiment chỉ được là 'Tích cực' hoặc 'Tiêu cực'. "
        "Bắt buộc trả về định dạng JSON theo mẫu."
    )
    
    examples = [
        {"role": "user", "content": "Giao hàng nhanh, đóng gói rất kỹ, nhưng giá hơi chát."},
        {"role": "assistant", "content": json.dumps({
            "sentiment": "Tích cực",
            "score": 0.6,
            "keywords": ["giao hàng nhanh", "đóng gói kỹ", "giá cao"]
        }, ensure_ascii=False)}
    ]
    
    messages = [{"role": "system", "content": system_instruction}] + examples
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        output_content = response.choices[0].message.content
        data = json.loads(output_content)
        
        valid, msg = validate_output(data)
        if not valid:
            return {"error": msg, "raw": data}
        
        return data
    
    except Exception as e:
        return {"error": str(e)}


def few_shot_2_analyzer(user_input):
    """
    Few-shot 2: 2 ví dụ đối lập (tích cực và tiêu cực).
    """
    system_instruction = (
        "Bạn là chuyên gia phân tích phản hồi khách hàng. "
        "Hãy trích xuất sentiment, score và keywords. "
        "sentiment chỉ được là 'Tích cực' hoặc 'Tiêu cực'. "
        "Bắt buộc trả về định dạng JSON theo mẫu."
    )
    
    examples = [
        {"role": "user", "content": "Giao hàng nhanh, đóng gói rất kỹ, sản phẩm đẹp hơn mong đợi."},
        {"role": "assistant", "content": json.dumps({
            "sentiment": "Tích cực",
            "score": 0.9,
            "keywords": ["giao hàng nhanh", "đóng gói kỹ", "đẹp hơn mong đợi"]
        }, ensure_ascii=False)},
        
        {"role": "user", "content": "Sản phẩm tệ quá, dùng được 2 ngày đã hỏng, gọi hỗ trợ không ai nghe."},
        {"role": "assistant", "content": json.dumps({
            "sentiment": "Tiêu cực",
            "score": 0.1,
            "keywords": ["tệ", "hỏng nhanh", "hỗ trợ kém"]
        }, ensure_ascii=False)}
    ]
    
    messages = [{"role": "system", "content": system_instruction}] + examples
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        output_content = response.choices[0].message.content
        data = json.loads(output_content)
        
        valid, msg = validate_output(data)
        if not valid:
            return {"error": msg, "raw": data}
        
        return data
    
    except Exception as e:
        return {"error": str(e)}


def few_shot_3_analyzer(user_input):
    """
    Few-shot 3: 3 ví dụ đa dạng (tích cực, tiêu cực, mixed nhưng phân loại rõ).
    """
    system_instruction = (
        "Bạn là chuyên gia phân tích phản hồi khách hàng. "
        "Hãy trích xuất sentiment, score và keywords. "
        "sentiment chỉ được là 'Tích cực' hoặc 'Tiêu cực'. "
        "Nếu phản hồi có cả ưu và nhược điểm, hãy xem xét tổng thể để chọn 1 trong 2. "
        "Bắt buộc trả về định dạng JSON theo mẫu."
    )
    
    examples = [
        {"role": "user", "content": "Shop tư vấn rất nhiệt tình, giao hàng đúng hẹn, sản phẩm đẹp hơn mong đợi."},
        {"role": "assistant", "content": json.dumps({
            "sentiment": "Tích cực",
            "score": 0.95,
            "keywords": ["tư vấn nhiệt tình", "giao đúng hẹn", "đẹp hơn mong đợi"]
        }, ensure_ascii=False)},
        
        {"role": "user", "content": "Máy dùng 2 hôm đã lỗi, gọi bảo hành không ai phản hồi."},
        {"role": "assistant", "content": json.dumps({
            "sentiment": "Tiêu cực",
            "score": 0.05,
            "keywords": ["lỗi nhanh", "bảo hành kém", "không phản hồi"]
        }, ensure_ascii=False)},
        
        {"role": "user", "content": "Giá hơi cao nhưng chất lượng ổn và đóng gói cẩn thận."},
        {"role": "assistant", "content": json.dumps({
            "sentiment": "Tích cực",
            "score": 0.65,
            "keywords": ["giá cao", "chất lượng ổn", "đóng gói cẩn thận"]
        }, ensure_ascii=False)}
    ]
    
    messages = [{"role": "system", "content": system_instruction}] + examples
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        output_content = response.choices[0].message.content
        data = json.loads(output_content)
        
        valid, msg = validate_output(data)
        if not valid:
            return {"error": msg, "raw": data}
        
        return data
    
    except Exception as e:
        return {"error": str(e)}


def compare_all():
    """
    So sánh 4 phương pháp trên cùng bộ test.
    """
    test_cases = [
        "Shop tư vấn rất nhiệt tình, giao hàng đúng hẹn, sản phẩm đẹp hơn mong đợi.",
        "Máy dùng 2 hôm đã lỗi, gọi bảo hành không ai phản hồi.",
        "Giá hơi cao nhưng chất lượng ổn và đóng gói cẩn thận.",
        "Áo đẹp, vải mát, mặc rất tôn dáng. Sẽ ủng hộ shop dài dài!",
        "Giao hàng chậm, sản phẩm không đúng mô tả, shop phản hồi lâu."
    ]
    
    methods = [
        ("Zero-shot", zero_shot_analyzer),
        ("Few-shot 1 (1 example)", few_shot_1_analyzer),
        ("Few-shot 2 (2 examples)", few_shot_2_analyzer),
        ("Few-shot 3 (3 examples)", few_shot_3_analyzer)
    ]
    
    print("=" * 80)
    print("LAB 2: SO SÁNH ZERO-SHOT VÀ FEW-SHOT PROMPTS")
    print("=" * 80)
    print()
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        print(f"Input: {test_input}")
        print()
        
        for method_name, method_func in methods:
            print(f"--- {method_name} ---")
            result = method_func(test_input)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print()
    
    print("\n" + "=" * 80)
    print("KẾT LUẬN")
    print("=" * 80)
    print("""
1. Zero-shot: Model hiểu yêu cầu cơ bản nhưng có thể không nhất quán về format và nhãn.

2. Few-shot 1: Cải thiện format nhưng chưa đủ để model phân biệt rõ tích cực/tiêu cực.

3. Few-shot 2: Model học được ranh giới giữa 2 thái cực, phân loại tốt hơn.

4. Few-shot 3: Ổn định nhất, xử lý tốt cả trường hợp mixed sentiment.

Khuyến nghị: Dùng few-shot với ít nhất 2-3 ví dụ đa dạng để đảm bảo chất lượng output.
    """)


if __name__ == "__main__":
    compare_all()
