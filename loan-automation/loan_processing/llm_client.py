import os
import requests
import base64
import io
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load the root .env file (one level above loan_processing)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")  # supply via .env; no hardcoded fallback

class LocalQwenLLM:
    """
    HTTP bridge to our fine-tuned Qwen2.5-VL running on port 8005 (or 8000).
    """
    def __init__(self, api_url=None):
        self.api_url = api_url or os.environ.get("QWEN_API_URL", "http://localhost:8005")

    def invoke(self, messages):
        try:
            image_datas = []
            prompt_text = ""
            
            # Parse Langchain messages to find image and text
            if isinstance(messages, list):
                for msg in messages:
                    content = getattr(msg, "content", msg)
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "image_url":
                                    url = part.get("image_url", {}).get("url", "")
                                    if url.startswith("data:image/"):
                                        image_datas.append(url.split(",")[1])
                                elif part.get("type") == "text":
                                    prompt_text += part.get("text", "")
                    elif isinstance(content, str):
                        prompt_text += content
            else:
                prompt_text = str(messages)

            # If there's an image, send to /extract
            if image_datas:
                combined_results = {}
                for idx, image_data in enumerate(image_datas):
                    image_bytes = base64.b64decode(image_data)
                    files = {'file': ('image.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
                    data = {}
                    
                    # Check specific document types first before falling back to ID card checks
                    dt_check = prompt_text.lower()
                    if 'خطاب' in prompt_text or 'مرتب' in prompt_text or 'hr_letter' in dt_check:
                        data['doc_type'] = 'hr_letter'
                    elif 'رخصة مزاولة' in prompt_text or 'practice_license' in dt_check:
                        data['doc_type'] = 'practice_license'
                    elif 'كارنيه النقابة' in prompt_text or 'syndicate_card' in dt_check:
                        data['doc_type'] = 'syndicate_card'
                    elif 'ترخيص العيادة' in prompt_text or 'المركز الطبي' in prompt_text or 'clinic_license' in dt_check:
                        data['doc_type'] = 'clinic_license'
                    elif 'فاتورة كهرباء' in prompt_text or 'electricity_bill' in dt_check:
                        data['doc_type'] = 'electricity_bill'
                    elif 'إيصال مرافق' in prompt_text or 'utility_bill' in dt_check or 'مرافق' in prompt_text:
                        data['doc_type'] = 'utility_bill'
                    elif 'عقد ملكية' in prompt_text or 'إيجار' in prompt_text or 'ownership_contract' in dt_check:
                        data['doc_type'] = 'ownership_contract'
                    elif 'البطاقة الضريبية' in prompt_text or 'tax_card' in dt_check:
                        data['doc_type'] = 'tax_card'
                    elif 'السجل التجاري' in prompt_text or 'commercial_register' in dt_check:
                        data['doc_type'] = 'commercial_register'
                    elif 'كشف حساب بنكي' in prompt_text or 'bank_statement' in dt_check:
                        data['doc_type'] = 'bank_statement'
                    elif 'تعهد بتحويل المعاش' in prompt_text or 'pension_transfer_pledge' in dt_check:
                        data['doc_type'] = 'pension_transfer_pledge'
                    elif 'تعهد بتحويل الراتب' in prompt_text or 'salary_transfer_pledge' in dt_check:
                        data['doc_type'] = 'salary_transfer_pledge'
                    elif len(image_datas) > 1 and ('رقم قومي' in prompt_text or 'بطاقة' in prompt_text):
                        data['doc_type'] = 'id_front' if idx == 0 else 'id_back'
                    elif 'بطاقة' in prompt_text and ('خلفي' in prompt_text or 'خلف' in prompt_text):
                        data['doc_type'] = 'id_back'
                    else:
                        data['doc_type'] = 'id_front'
                    
                    print(f"[LLM_CLIENT] -> POST {self.api_url}/extract  doc_type={data['doc_type']}", flush=True)
                    response = requests.post(f"{self.api_url}/extract", files=files, data=data, timeout=300)
                    res_json = response.json()
                    print(f"[LLM_CLIENT] /extract status={res_json.get('status')} doc_type={data['doc_type']} keys={list(res_json.get('data', {}).keys()) if res_json.get('status')=='success' else res_json.get('message','')[:80]}", flush=True)

                    if res_json.get("status") == "success":
                        extracted_data = res_json.get("data", {})
                        if extracted_data:  # Only update if we got real data
                            combined_results.update(extracted_data)
                    else:
                        # Model returned error - store raw message but keep document as uploaded
                        combined_results["_api_error"] = str(res_json.get("message", "unknown"))[:300]
                        combined_results["_raw"] = str(res_json.get("raw", ""))[:300]
                
                response_text = json.dumps(combined_results, ensure_ascii=False)
                    
            # If no image, send to /prompt
            else:
                payload = {"prompt": prompt_text[:6000]}
                response = requests.post(f"{self.api_url}/prompt", json=payload, timeout=300)
                res_json = response.json()
                response_text = res_json.get("data", "") if res_json.get("status") == "success" else res_json.get("message", "")
                
        except Exception as e:
            response_text = f'{{"error": "{e}"}}'
            
        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse(response_text)

def get_llm(loan_type=None):
    """
    Always returns the local fine-tuned Qwen model via HTTP.
    """
    return LocalQwenLLM()

def get_hf_client():
    """OpenAI-compatible Hugging Face router client."""
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )

HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def ask_qwen(prompt: str, temperature: float = 0) -> str:
    """Send a prompt to Llama-3.1-8B via HuggingFace Inference Router
    and return the response as a plain string.

    The function name is kept as ask_qwen to avoid breaking existing call sites.
    """
    client = get_hf_client()
    completion = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON-only responder. "
                    "Always respond with valid JSON and nothing else. "
                    "Do not add any explanation, markdown, or backticks."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=2000,
    )
    content = completion.choices[0].message.content
    return content if content else ""

class QwenReasoningWrapper:
    def invoke(self, prompt: str):
        response_text = ask_qwen(prompt)
        class MockResponse:
            content = response_text
        return MockResponse()

def get_reasoning_llm():
    """Reasoning via HuggingFace Llama-3.1-8B-Instruct."""
    return QwenReasoningWrapper()

def extract_json_from_text(text: str):
    import re
    if not text: return None
    try:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass
    return None