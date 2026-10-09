# NOVA - OCR Module (Text Reading)
# الهدف: نقرا أي نص موجود في صورة (فاتورة، ورقة، لافتة) بالعربي أو الإنجليزي،
# ونستخرج منه معلومات مفيدة زي المبلغ، رقم التليفون، التاريخ، إلخ.
#
# pip install easyocr

import re
import easyocr

# ------------------------------------------------------------------
# أنماط البحث (Regex Patterns): كل نمط بيوصف "شكل" المعلومة اللي بندور عليها
# ------------------------------------------------------------------
PATTERNS = {
    "money":  r'\d+(?:[.,]\d+)?\s*(?:جنيه|EGP|LE|USD|\$|€|dollar|pound)',
    "phone":  r'(?:\+20|0)1[0125]\d{8}',
    "date":   r'\d{1,4}[/-]\d{1,2}[/-]\d{1,4}',
    "email":  r'[\w.-]+@[\w.-]+\.\w+',
    "link":   r'https?://\S+',
}

# تحويل الأرقام العربية (٠١٢٣...) للأرقام الإنجليزية (0123...)
ARABIC_TO_ENGLISH_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


class OCRReader:
    """بتقرا النص من صورة وتستخرج منه معلومات مفيدة (فلوس، تواريخ، تليفونات...)."""

    def __init__(self, languages=None):
        # أول مرة بتتنشأ، هتنزل موديلات اللغة تلقائيًا من الإنترنت (مرة واحدة بس)
        self.reader = easyocr.Reader(languages or ['ar', 'en'])

    def read_text(self, image_path):
        """بترجع كل النص اللي لقته في الصورة كسطر واحد طويل."""
        raw_lines = self.reader.readtext(image_path, detail=0)
        text = "\n".join(raw_lines)
        return text.translate(ARABIC_TO_ENGLISH_DIGITS)

    def extract_info(self, text):
        """بتدور في النص عن الأنماط المعروفة (فلوس، تليفونات، إلخ) وترجعهم كـ dictionary."""
        found_info = {}
        for name, pattern in PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found_info[name] = matches
        return found_info

    def read_and_analyze(self, image_path):
        """بتعمل الحاجتين مع بعض: تقرا الصورة وترجع النص + المعلومات المستخرجة."""
        text = self.read_text(image_path)
        info = self.extract_info(text)
        return text, info


if __name__ == "__main__":
    # اختبار: حط صورة اسمها test_document.jpg في نفس المجلد وشغل الملف ده لوحده
    ocr = OCRReader()

    image_file = "test_document.jpg"
    text, info = ocr.read_and_analyze(image_file)

    print("النص الكامل:\n", text, "\n")

    if info:
        for name, matches in info.items():
            print(f"{name}: {matches}")
    else:
        print("مفيش معلومات معروفة (فلوس/تليفون/تاريخ) اتلقت في النص.")
