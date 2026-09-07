import os
import random
import cv2
import numpy as np
import sys
from faker import Faker
from jinja2 import Template
from html2image import Html2Image

# ========== Setup ==========
hti = Html2Image(output_path='output')
fake_ar = Faker(['ar_EG'])
fake_en = Faker(['en_US'])

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

STAMPS_IMAGE_PATH = r"D:\project\loan\confirm\ExtImage-2938597-435234432.jpg"
SIGNATURES_IMAGE_PATH = r"D:\project\loan\confirm\download.jfif"

# ========== Signatures Extraction ==========
def extract_signatures(sig_img_path):
    """Extract all 9 signatures from the 3x3 grid image."""
    if not os.path.exists(sig_img_path):
        print(f"Signature image not found: {sig_img_path}")
        return []
    
    img = cv2.imread(sig_img_path)
    h, w = img.shape[:2]
    cell_w = w // 3
    cell_h = h // 3
    sig_paths = []
    
    os.makedirs("temp_sigs", exist_ok=True)
    idx = 0
    for row in range(3):
        for col in range(3):
            x1 = col * cell_w
            y1 = row * cell_h
            cell = img[y1:y1+cell_h, x1:x1+cell_w]
            
            # Detect blue ink using optimized HSV
            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 60, 0])
            upper_blue = np.array([140, 255, 200])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Detect dark marks (black ink/dark blue)
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, mask_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            # Combine masks
            mask = cv2.bitwise_or(mask_blue, mask_dark)
            
            # Make the ink perfectly dark blue
            b = np.full(mask.shape, 150, dtype=np.uint8)
            g = np.full(mask.shape, 50, dtype=np.uint8)
            r = np.full(mask.shape, 20, dtype=np.uint8)
            rgba = cv2.merge([b, g, r, mask])
            
            # Crop 4 pixels to remove cell borders
            pad = 4
            rgba = rgba[pad:-pad, pad:-pad]
            
            path = os.path.abspath(f"temp_sigs/sig_{idx}.png").replace("\\", "/")
            cv2.imwrite(path, rgba)
            sig_paths.append(path)
            idx += 1
            
    return sig_paths

# ========== Single stamp extraction (Optimized) ==========
# Use ONE clean circular stamp from the center
STAMP_COORD = [220, 190, 220, 220]  # x, y, w, h - الختم الدائري الكبير بعد التعديل ليكون كامل

def get_stamp(stamps_img_path):
    """Extract one optimized clean stamp from the stamps sheet."""
    if not os.path.exists(stamps_img_path):
        print(f"Stamp image not found: {stamps_img_path}")
        return None
    
    img = cv2.imread(stamps_img_path)
    x, y, w, h = STAMP_COORD
    stamp = img[y:y+h, x:x+w]
    
    # Optimized extraction using HSV for blue color + dark thresholding
    hsv = cv2.cvtColor(stamp, cv2.COLOR_BGR2HSV)
    lower = np.array([85, 20, 20])
    upper = np.array([145, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    
    mask = cv2.bitwise_or(mask, dark)
    
    # Clean up small noise in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    b, g, r = cv2.split(stamp)
    rgba = cv2.merge([b, g, r, mask])
    
    stamp_path = os.path.abspath("temp_stamp.png").replace("\\", "/")
    cv2.imwrite(stamp_path, rgba)
    return stamp_path


# ========== Data Banks ==========
ARABIC_JOBS = [
    "محاسب", "مهندس مدني", "مهندس برمجيات", "مدير مبيعات",
    "أخصائي موارد بشرية", "محلل مالي", "مدير تسويق", "مطور ويب",
    "مدير مشروعات", "أخصائي تأمينات", "مراجع حسابات", "مدير عمليات",
    "مصمم جرافيك", "مهندس شبكات", "أخصائي مشتريات", "مدير إنتاج"
]

ARABIC_COMPANIES = [
    "شركة النيل للتجارة والاستيراد", "مجموعة الأهرام القابضة",
    "شركة مصر للتأمين", "شركة السلام للمقاولات",
    "مجموعة العربي للإلكترونيات", "شركة الدلتا للأغذية",
    "شركة النور للتطوير العقاري", "مجموعة المحروسة للاستثمار",
    "شركة الوادي للبترول", "شركة القاهرة للصناعات الدوائية",
    "شركة الإسكندرية للأسمنت", "مجموعة طلعت مصطفى",
    "شركة أوراسكوم للإنشاءات", "شركة المنصور للسيارات",
    "مجموعة السويدي إلكتريك", "شركة جهينة للصناعات الغذائية"
]

ENGLISH_COMPANIES = [
    "Orascom Construction Group", "El Sewedy Electric Co.",
    "Talaat Moustafa Group", "Juhayna Food Industries",
    "El Mansour Automotive", "Delta Foods International",
    "Cairo Pharmaceuticals Ltd.", "Alexandria Cement Co.",
    "Nile Trading & Import Co.", "Al Ahram Holding Group",
    "Arabian Electronics Group", "Valley Petroleum Co.",
    "Al Nour Real Estate Dev.", "El Mahrousa Investments",
    "Salam Contracting Co.", "Misr Insurance Group"
]

ENGLISH_JOBS = [
    "Accountant", "Civil Engineer", "Software Engineer", "Sales Manager",
    "HR Specialist", "Financial Analyst", "Marketing Manager", "Web Developer",
    "Project Manager", "Insurance Specialist", "Auditor", "Operations Manager",
    "Graphic Designer", "Network Engineer", "Procurement Specialist", "Production Manager"
]

ARABIC_NAMES = [
    "ابراهيم حسن ابراهيم حسن", "ابراهيم محمد علي يونس", "احمد شوقى عباس السيد", "ام ابراهيم السيد عبدالحميد صقر",
    "امانى سعد عبدالعزيز يرنس محمود", "امينه شعبان عبدالحافظ ابراهيم", "انيسه ابراهيم سعد المعناوى", "باسم محمد فتح الله كسر",
    "بشرى ابوشعيشع قاسم الجعفراوي", "حسام محمد عرجاوى عبدالغنى الطور", "حسن محمد احمد حسنين", "حلاوتهم سعيد مروان زامل",
    "دعاء مسعد سعد محمد سكوت", "رامز نصر محمد الرخاوى", "رضا واصف محمد على", "زينب غريب علي حسن غانم",
    "سحر محمد ابراهيم هلال", "سعد محمد محمد الشائلي", "سميره التميمي محمد حسن", "سهير احمد محمد عبدالغنى بتوت",
    "سهير عبدالعال عبدالوهاب ابوعجيله", "سينا ابراهيم سليمان الوكيل", "شاديه عبدالعال عبدالمالك عبدالعال", "شعبان عبدالله صالح عبدالله",
    "صباح فوزى عبدالعزيز سرحان", "صبحه محمد ابراهيم الدعمه", "عادل خليل خليل ابراهيم ابوطالب", "عاطف محمد الغريب محمد",
    "عاليه نبيل السيد اسماعيل", "عبدالحميد حميدو عبدالحميد الجباوي", "عبدالسلام خيرى عبدالسلام العلوانى", "علي جابر محمد الصاوي",
    "عماد عبدالرحمن محمد حلاوه", "فاديه احمد سعد حسن سليمان", "فرج احمد محمد الشرقاوى", "محمد احمد سعد حسن سليمان",
    "محمد سعد محمد عبدالعال", "محمد سعد محمود البلتاجى", "محمد صلاح عبدالغفار محمد بدر", "محمد محمود محمد محمد هلال",
    "محمد هبدالغفار عبدالهادي ابراهيم محجوب", "محمود محمود محمد ابوعيسى", "مديحة بسيونى نعمت الله حسن", "مصطفى ابراهيم ابراهيم متولي جبر",
    "مصطفى محمد محمد عبدالحليم", "منى صباح محمد الحنفي", "نورا محمد احمد حسن", "هانم محمد عبدالعزيز على",
    "هانى ابراهيم عبدالقادر إبراهيم"
]

ENGLISH_NAMES = [
    "Mohamed Abdel Rahman Mohamed Ali", "Ahmed Hussein Mahmoud Ibrahim", "Khaled El Sherif Abdullah Hassan", "Amr Ismail Mostafa Kamal",
    "Hassan Mahmoud Ahmed Saleh", "Ibrahim El Sayed Mohamed Youssef", "Mostafa Saleh Sherif Abdel Aziz", "Abdullah Ramadan Hassan Ali",
    "Youssef El Gendy Ibrahim El Sayed", "Omar Abdel Hamid Tarek El Badawy", "Tarek El Badawy Sami El Masry", "Sami El Masry Hesham Abu Zeid",
    "Hesham Abu Zeid Yasser El Feky", "Yasser El Feky Walid Osman", "Walid Osman Karim El Naggar", "Karim El Naggar Ramy El Hadidy",
    "Ramy El Hadidy Sherif Abdel Aziz", "Sherif Abdel Aziz Maged Hassan", "Maged Hassan Nabil Ali", "Nabil Ali Farouk Gamal",
    "Farouk Gamal Adel Said", "Adel Said Gamal Farouk", "Gamal Farouk Said Adel", "Said Adel Tamer Hosny",
    "Tamer Hosny Emad El Din", "Emad El Din Bahaa El Din", "Bahaa El Din Salah El Din", "Salah El Din Mohamed Ahmed"
]

# ========== HTML Templates (clean, no noise) ==========

TEMPLATE_ENGLISH = """
<style>
    body { font-family: 'Arial', sans-serif; padding: 50px; background: white; color: #222; }
    .document { border: 1px solid #999; padding: 60px; width: 700px; height: 950px; position: relative; box-sizing: border-box; }
    .header { border-bottom: 2px solid #333; padding-bottom: 15px; margin-bottom: 40px; display: flex; justify-content: space-between; align-items: flex-start; }
    .company-info h2 { color: #1a237e; margin: 0; }
    .company-info p { color: #555; margin: 5px 0 0 0; font-size: 14px; }
    .issue-date { font-size: 13px; color: #777; white-space: nowrap; }
    .body-text { font-size: 17px; line-height: 1.8; margin-top: 30px; }
    .sig-area { position: absolute; bottom: 80px; left: 80px; text-align: center; }
    .sig-img { width: 140px; opacity: 0.95; margin-bottom: -30px; display: block; margin-left: auto; margin-right: auto; }
    .stamp-img { width: 140px; transform: rotate({{ rotation }}deg); opacity: 0.9; position: absolute; left: 20px; top: -10px; z-index: -1; }
    .sig-title { font-size: 14px; color: #555; margin-top: 15px; border-top: 1px solid #ccc; padding-top: 5px; }
</style>
<div class="document">
    <div class="header">
        <div class="company-info">
            <h2>{{ company }}</h2>
            <p>Human Resources Department</p>
        </div>
        <div class="issue-date">Issue Date: {{ date }}</div>
    </div>
    <h3 style="text-align: center; margin-top: 40px; font-size: 20px;">To Whom It May Concern</h3>
    <div class="body-text">
        <p>This is to certify that <b>Mr. {{ name }}</b>, holder of National ID: <b>{{ national_id }}</b>,
        has been employed at <b>{{ company }}</b> since <b>{{ start_date }}</b>
        in the capacity of <b>{{ job }}</b>.</p>
        <p>His current monthly net salary is <b>{{ salary }} EGP</b> (Egyptian Pounds).</p>
        <p style="margin-top: 30px;">This certificate is issued upon his request without any liability on the company.</p>
    </div>
    <div class="sig-area">
        <div style="position: relative;">
            {% if stamp_path %}<img src="file:///{{ stamp_path }}" class="stamp-img">{% endif %}
            <img src="file:///{{ sig_path }}" class="sig-img">
        </div>
        <p class="sig-title"><b>HR Manager</b></p>
    </div>
</div>
"""

TEMPLATE_ARABIC = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    body { font-family: 'Cairo', 'Arial', sans-serif; direction: rtl; padding: 40px; background: white; }
    .document { border: 2px solid #333; padding: 50px; width: 700px; height: 950px; position: relative; }
    .header { border-bottom: 3px double #333; padding-bottom: 15px; margin-bottom: 30px; text-align: center; }
    .header h2 { color: #1a237e; margin: 0; font-size: 24px; }
    .header p { color: #555; margin: 5px 0 0 0; font-size: 14px; }
    .body-text { font-size: 18px; line-height: 2.2; margin-top: 20px; }
    .sig-area { position: absolute; bottom: 80px; left: 80px; text-align: center; }
    .sig-img { width: 140px; opacity: 0.95; margin-bottom: -30px; display: block; margin-left: auto; margin-right: auto; }
    .stamp-img { width: 140px; transform: rotate({{ rotation }}deg); opacity: 0.9; position: absolute; right: 20px; top: -10px; z-index: -1; }
    .sig-title { font-size: 14px; color: #555; margin-top: 15px; border-top: 1px solid #ccc; padding-top: 5px; }
</style>
<div class="document">
    <div class="header">
        <h2>{{ company }}</h2>
        <p>إدارة الموارد البشرية</p>
    </div>
    <div style="text-align: left; font-size: 14px; color: #555;">القاهرة في: {{ date }}</div>
    <h3 style="text-align: center; margin-top: 30px; font-size: 22px; text-decoration: underline;">شهادة راتب</h3>
    <div class="body-text">
        <p>تشهد <b>{{ company }}</b> بأن السيد/ <b>{{ name }}</b>
        والذي يحمل رقم قومي <b>{{ national_id }}</b>
        يعمل لدينا منذ تاريخ <b>{{ start_date }}</b>
        بوظيفة <b>{{ job }}</b>.</p>
        <p>وأن صافي راتبه الشهري هو ( <b>{{ salary }} جنيه مصري</b> ).</p>
        <p style="margin-top: 20px;">أُعطيت هذه الشهادة بناءً على طلبه دون أدنى مسؤولية على الشركة.</p>
    </div>
    <div class="sig-area">
        <div style="position: relative;">
            {% if stamp_path %}<img src="file:///{{ stamp_path }}" class="stamp-img">{% endif %}
            <img src="file:///{{ sig_path }}" class="sig-img">
        </div>
        <p class="sig-title"><b>مدير الموارد البشرية</b></p>
    </div>
</div>
"""

# ========== Factory ==========

def build_factory(num=50, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract the single optimized stamp once
    # Fix paths to be relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stamps_path = os.path.join(base_dir, "confirm", "ExtImage-2938597-435234432.jpg")
    sigs_path = os.path.join(base_dir, "confirm", "download.jfif")
    
    stamp_p = get_stamp(stamps_path)
    
    # Extract signatures
    sig_paths = extract_signatures(sigs_path)
    if not sig_paths:
        print("Warning: No signatures extracted.")
    
    # Need to update Html2Image initialization to use the new output dir
    global hti
    hti = Html2Image(output_path=output_dir)
    
    for i in range(num):
        is_arabic = (i % 2 == 0)  # alternate: even=arabic, odd=english
        
        if is_arabic:
            name = random.choice(ARABIC_NAMES)
            company = random.choice(ARABIC_COMPANIES)
            job = random.choice(ARABIC_JOBS)
            template_str = TEMPLATE_ARABIC
        else:
            name = random.choice(ENGLISH_NAMES)
            company = random.choice(ENGLISH_COMPANIES)
            job = random.choice(ENGLISH_JOBS)
            template_str = TEMPLATE_ENGLISH
            
        current_sig_path = random.choice(sig_paths) if sig_paths else ""
        
        data = {
            "company": company,
            "name": name,
            "national_id": random.choice(["26", "27", "28", "29", "30"]) + fake_ar.numerify("############"),
            "job": job,
            "salary": f"{random.randint(8000, 60000):,}",
            "start_date": fake_ar.date_between(start_date='-10y').strftime('%Y-%m-%d'),
            "date": fake_ar.date_this_year().strftime('%Y/%m/%d'),
            "sig_path": current_sig_path,
            "stamp_path": stamp_p,
            "rotation": random.randint(-15, 15)
        }

        html_content = Template(template_str).render(data)
        
        img_name = f"hr_doc_{i}.png"
        img_path = os.path.join(output_dir, img_name)

        try:
            hti.screenshot(html_str=html_content, save_as=img_name)
            
            if os.path.exists(img_path):
                print(f"[{i+1}/{num}] Generated: {img_path}")
            else:
                # Html2Image might save it in the current working directory instead of output_path sometimes
                if os.path.exists(img_name):
                    os.replace(img_name, img_path)
                    print(f"[{i+1}/{num}] Generated: {img_path}")
                else:
                    print(f"[{i+1}/{num}] Warning: could not find output for image {i}")
        except Exception as e:
            print(f"[{i+1}/{num}] Error: {e}")

if __name__ == "__main__":
    print("Starting HR Document Factory...")
    build_factory(1000)
    print("Done! Check the 'output' folder.")