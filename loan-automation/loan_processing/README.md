# Loan Processing – LangGraph + Gemini

A complete LangGraph-based loan processing workflow using Google Gemini.

## Project Structure

```
loan_processing/
├── config.py              ← All loan types, required docs, validation rules
├── state.py               ← GraphState TypedDict
├── llm_client.py          ← Gemini LLM singleton  ← ADD YOUR API KEY HERE
├── graph.py               ← LangGraph StateGraph wiring
├── main.py                ← CLI runner + test scenarios
├── nodes/
│   ├── select_loan_type.py
│   ├── show_required_docs.py
│   ├── upload_docs_and_form.py
│   ├── extract_documents.py      ← LLM agent
│   ├── check_doc_presence.py     ← LLM agent
│   ├── validate_conditions.py    ← LLM agent
│   ├── score_threshold.py
│   └── final_nodes.py
└── edges/
    └── routers.py
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Open `loan_processing/llm_client.py` and either:

**Option A – Environment variable (recommended):**
```bash
export GOOGLE_API_KEY="your-key-here"
```

**Option B – Paste directly into the file:**
```python
GOOGLE_API_KEY = "your-key-here"    # line 17 in llm_client.py
```

Get your key at: https://aistudio.google.com/app/apikey

### 3. Run

```bash
# Default demo (personal_doctors loan)
python -m loan_processing.main

# Specific loan type
python -m loan_processing.main --loan car
python -m loan_processing.main --loan personal_freelance
python -m loan_processing.main --loan personal_salary_unsecured

# List all loan types
python -m loan_processing.main --list
```

## Loan Types

| Key | Arabic name |
|-----|-------------|
| `car` | قرض السيارة |
| `personal_default_income` | برنامج الدخل الافتراضي |
| `personal_doctors` | للأطباء وأصحاب العيادات |
| `personal_pension` | أصحاب المعاشات |
| `personal_freelance` | أصحاب الأعمال والمهن الحرة |
| `personal_salary_secured` | بضمان تحويل الراتب |
| `personal_salary_unsecured` | بدون ضمان تحويل الراتب |

## Decision Thresholds

| Score | Decision |
|-------|----------|
| ≥ 85% | ✅ Approve |
| 60–84% | ⚠️ Manual review |
| < 60% | ❌ Reject |

## Graph Flow

```
select_loan_type
      ↓
show_required_docs
      ↓
upload_docs_and_form  ←──────────────────┐
      ↓                                   │ missing docs
extract_documents                         │
      ↓                                   │
check_doc_presence ───────────────────────┘
      ↓ all present
validate_conditions
      ↓
score_threshold
      ↓
  ┌───┴───────┬──────────────┐
approve   manual_review   reject
  ↓            ↓              ↓
 END           END            END
```
