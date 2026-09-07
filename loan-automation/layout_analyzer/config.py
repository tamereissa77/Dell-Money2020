"""
Configuration for Document Layout Analysis.
Defines class labels, colors, and spatial zone heuristics for each document type.
"""

# ═══════════════════════════════════════════════════════════════
# CLASS DEFINITIONS & COLOR CODING
# ═══════════════════════════════════════════════════════════════
# Colors in BGR format for OpenCV drawing

# TASK 1: ID Front classes
ID_FRONT_CLASSES = {
    "Full_Name": {
        "color": (0, 200, 0),       # Green
        "label_en": "Full Name",
        "label_ar": "الاسم الكامل",
    },
    "Address": {
        "color": (255, 165, 0),     # Orange
        "label_en": "Address",
        "label_ar": "العنوان",
    },
    "National_ID_Front": {
        "color": (255, 0, 0),       # Blue
        "label_en": "National ID",
        "label_ar": "الرقم القومي",
    },
    "Birthdate": {
        "color": (0, 0, 255),       # Red
        "label_en": "Birth Date",
        "label_ar": "تاريخ الميلاد",
    },
}

# TASK 2: ID Back classes
ID_BACK_CLASSES = {
    "National_ID_Back": {
        "color": (255, 0, 0),       # Blue
        "label_en": "National ID",
        "label_ar": "الرقم القومي",
    },
    "Expiry_Date": {
        "color": (0, 200, 200),     # Yellow
        "label_en": "Expiry Date",
        "label_ar": "تاريخ الانتهاء",
    },
    "Job_Profession": {
        "color": (200, 0, 200),     # Magenta
        "label_en": "Profession",
        "label_ar": "المهنة",
    },
    "Gender": {
        "color": (0, 200, 0),       # Green
        "label_en": "Gender",
        "label_ar": "النوع",
    },
    "Marital_Status": {
        "color": (200, 200, 0),     # Cyan
        "label_en": "Marital Status",
        "label_ar": "الحالة الاجتماعية",
    },
}

# TASK 3: HR Letter classes
HR_LETTER_CLASSES = {
    "Company_Name": {
        "color": (0, 165, 255),     # Orange
        "label_en": "Company Name",
        "label_ar": "اسم الشركة",
    },
    "Employee_Name": {
        "color": (0, 200, 0),       # Green
        "label_en": "Employee Name",
        "label_ar": "اسم الموظف",
    },
    "HR_National_ID": {
        "color": (255, 0, 0),       # Blue
        "label_en": "National ID",
        "label_ar": "الرقم القومي",
    },
    "Salary_Amount": {
        "color": (0, 0, 255),       # Red
        "label_en": "Salary",
        "label_ar": "الراتب",
    },
    "Job_Title": {
        "color": (255, 128, 0),     # Orange-ish
        "label_en": "Job Title",
        "label_ar": "المسمى الوظيفي",
    },
    "Hire_Date": {
        "color": (200, 0, 200),     # Magenta
        "label_en": "Hire Date",
        "label_ar": "تاريخ التعيين",
    },
    "Issue_Date": {
        "color": (0, 200, 200),     # Yellow
        "label_en": "Issue Date",
        "label_ar": "تاريخ الإصدار",
    },
    "Signature_Stamp": {
        "color": (128, 0, 0),       # Dark Blue
        "label_en": "Signature/Stamp",
        "label_ar": "التوقيع/الختم",
    },
}

# Box drawing style
BOX_THICKNESS = 3
LABEL_FONT_SCALE = 0.7
LABEL_THICKNESS = 2
LABEL_BG_ALPHA = 0.7
