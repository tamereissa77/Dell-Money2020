"""
nodes/extract_documents.py
LLM Agent – extracts structured content from each uploaded document.

For each uploaded document the agent returns a JSON summary of
all key fields it can find (dates, names, IDs, amounts, etc.).
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _get_text(content) -> str:
    """Normalise LLM response content to a plain string.
    Older models return a str; newer ones return a list of content parts,
    where each part can be a str, a dict with 'text', or an object with .text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif hasattr(part, "text"):
                parts.append(part.text)
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)
try:
    from ..state import GraphState
    from ..config import DOC_DISPLAY_NAMES
    from ..llm_client import get_llm
except ImportError:
    from state import GraphState
    from config import DOC_DISPLAY_NAMES
    from llm_client import get_llm
from langchain_core.messages import HumanMessage


EXTRACTION_PROMPT = """أنت نظام استخراج بيانات متخصص في المستندات البنكية المصرية.

اسم المستند: {doc_name}
محتوى المستند:
\"\"\"
{doc_content}
\"\"\"

استخرج كل المعلومات المهمة من هذا المستند وأعدها كـ JSON فقط.
يجب أن يتضمن الـ JSON على الأقل الحقول التالية إذا كانت موجودة:
- document_type: نوع المستند
- holder_name: اسم صاحب المستند
- national_id_number: رقم الهوية إن وجد
- issue_date: تاريخ الإصدار (YYYY-MM-DD)
- expiry_date: تاريخ الانتهاء (YYYY-MM-DD)
- date_of_birth: تاريخ الميلاد (YYYY-MM-DD)
- age: العمر بالسنوات
- monthly_income: الدخل الشهري (رقم فقط)
- amount: أي مبلغ مالي مذكور (رقم فقط، بالجنيه المصري)
- activity_start_date: تاريخ بدء النشاط (YYYY-MM-DD)
- statement_from_date: بداية الكشف (YYYY-MM-DD)
- statement_to_date: نهاية الكشف (YYYY-MM-DD)
- hire_date: تاريخ التعيين (YYYY-MM-DD)
- company_name: اسم الشركة
- is_valid: هل المستند ساري؟ (true/false)
- both_sides_present: هل تم توفير صورة الوجهين (الأمامي والخلفي) للمستند؟ (true/false)
- notes: ملاحظات أخرى مهمة

أعد JSON فقط بدون أي نص إضافي أو backticks.
"""


def _extract_one(doc_key, doc_content, doc_display, llm):
    """Extract one document — runs in a thread."""
    t0 = time.time()
    print(f"[extract] ⏳ Starting extraction: {doc_display} ({doc_key})", flush=True)

    if not doc_content:
        return doc_key, {"error": "empty_document"}

    if isinstance(doc_content, list):
        prompt = EXTRACTION_PROMPT.format(doc_name=doc_display, doc_content="[صور المستند مرفقة]")
        content_parts = [{"type": "text", "text": prompt}]
        for part in doc_content:
            if isinstance(part, str) and part.startswith("data:image/"):
                content_parts.append({"type": "image_url", "image_url": {"url": part}})
            else:
                content_parts.append({"type": "text", "text": str(part)[:6000]})
        invoke_input = [HumanMessage(content=content_parts)]
    elif isinstance(doc_content, str) and doc_content.startswith("data:image/"):
        prompt = EXTRACTION_PROMPT.format(doc_name=doc_display, doc_content="[صورة المستند مرفقة]")
        invoke_input = [HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": doc_content}}
        ])]
    else:
        prompt = EXTRACTION_PROMPT.format(doc_name=doc_display, doc_content=str(doc_content)[:6000])
        invoke_input = prompt

    try:
        response = llm.invoke(invoke_input)
        raw = _get_text(response.content).strip()

        start_idx = raw.find('{')
        if start_idx != -1:
            try:
                decoder = json.JSONDecoder()
                parsed, _ = decoder.raw_decode(raw[start_idx:])
            except Exception:
                parsed = json.loads(raw)
        else:
            parsed = json.loads(raw)
        elapsed = round(time.time() - t0, 2)
        print(f"\n==================================================", flush=True)
        print(f"⏱ {elapsed}s | 📄 EXTRACTED: [{doc_display}] ({doc_key})", flush=True)
        print(f"--------------------------------------------------", flush=True)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                print(f"  • {k:<22}: {v}", flush=True)
        else:
            print(json.dumps(parsed, ensure_ascii=False, indent=2), flush=True)
        print(f"==================================================\n", flush=True)
        return doc_key, parsed

    except json.JSONDecodeError:
        elapsed = round(time.time() - t0, 2)
        print(f"\n==================================================", flush=True)
        print(f"⏱ {elapsed}s | ⚠ COULD NOT PARSE JSON FOR: [{doc_display}]", flush=True)
        print(f"Raw: {raw[:300]}", flush=True)
        print(f"==================================================\n", flush=True)
        return doc_key, {"raw_text": raw, "parse_error": True}
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        print(f"[extract] ✗ {doc_display} ({elapsed}s) → error: {e}", flush=True)
        return doc_key, {"error": str(e)}


def extract_documents(state: GraphState) -> GraphState:
    """Extract key fields from every uploaded document using parallel threads."""
    llm = get_llm()
    uploaded = state.get("uploaded_docs", {})
    extracted = {}

    total_docs = len([v for v in uploaded.values() if v])
    print(f"\n[extract] ⚡ Sending {total_docs} doc(s) to vision model IN PARALLEL...", flush=True)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max(total_docs, 1)) as executor:
        futures = {
            executor.submit(_extract_one, doc_key, doc_content,
                            DOC_DISPLAY_NAMES.get(doc_key, doc_key), llm): doc_key
            for doc_key, doc_content in uploaded.items()
        }
        for future in as_completed(futures):
            doc_key, result = future.result()
            extracted[doc_key] = result

    total_elapsed = round(time.time() - t_start, 2)
    print(f"[extract] ✅ All {total_docs} doc(s) extracted in {total_elapsed}s (parallel)", flush=True)

    return {**state, "extracted_docs": extracted}
