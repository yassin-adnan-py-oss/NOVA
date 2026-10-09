# NOVA - Memory Module
# الهدف: نسجل الأجسام المهمة اللي شافها NOVA (زي مفاتيح، محفظة) مع الوقت والمكان،
# وبعدين نقدر نرد على أسئلة زي "فين مفاتيحي؟" بالرجوع لآخر مرة اتسجلت فيها.

import json
import os
import time
from datetime import datetime

# الأجسام اللي يستاهل نسجلها في الذاكرة (مش كل حاجة، بس الحاجات اللي المستخدم ممكن يدور عليها)
MEMORABLE_OBJECTS = {
    "cell phone", "keys", "key", "wallet", "backpack", "handbag",
    "book", "cup", "bottle", "laptop", "remote", "scissors"
}

MEMORY_FILE = "nova_memory.json"


class Memory:
    """بتسجل الأجسام المهمة وتدور عليها لاحقًا بالاسم."""

    def __init__(self, file_path=MEMORY_FILE):
        self.file_path = file_path
        self.records = self._load()

    # ----------------------------------------------------------------
    # تحميل وحفظ البيانات من/على القرص (عشان الذاكرة تفضل موجودة حتى لو قفلنا البرنامج)
    # ----------------------------------------------------------------
    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------
    # تسجيل جسم شفناه (Remember)
    # ----------------------------------------------------------------
    def remember(self, label, direction, color=""):
        """
        بتسجل جسم مهم شفناه دلوقتي، وتستبدل أي تسجيل قديم لنفس الجسم
        (يعني بنحتفظ بآخر مكان شفنا فيه الجسم ده بس، مش كل مرة).
        """
        if label not in MEMORABLE_OBJECTS:
            return  # مش من الأجسام اللي يستاهل نسجلها

        self.records[label] = {
            "direction": direction,
            "color": color,
            "timestamp": time.time(),
            "readable_time": datetime.now().strftime("%I:%M %p"),
        }
        self._save()

    # ----------------------------------------------------------------
    # البحث عن جسم (Recall)
    # ----------------------------------------------------------------
    def recall(self, query_text):
        """
        بتدور في الذاكرة عن أي جسم اسمه موجود في نص السؤال.
        مثال: query_text = "where are my keys" -> بيدور عن "key" أو "keys"
        """
        query_lower = query_text.lower()

        for label, info in self.records.items():
            # بندور هل اسم الجسم (أو جزء منه) موجود في السؤال
            if label in query_lower or label.rstrip("s") in query_lower:
                minutes_ago = int((time.time() - info["timestamp"]) / 60)
                time_phrase = "just now" if minutes_ago < 1 else f"{minutes_ago} minutes ago"

                color_phrase = f"{info['color']} " if info["color"] else ""
                return (
                    f"I last saw your {color_phrase}{label} {time_phrase}, "
                    f"on your {info['direction']}, around {info['readable_time']}."
                )

        return "I don't remember seeing that recently."

    def forget_all(self):
        """بتمسح كل الذاكرة (مفيدة لو عايز تبدأ من جديد وقت التجربة)."""
        self.records = {}
        self._save()


if __name__ == "__main__":
    # اختبار بسيط للتأكد إن الكود شغال قبل ما ندمجه مع nova_core.py
    memory = Memory("test_memory.json")

    memory.remember("keys", "right", "silver")
    print(memory.recall("where are my keys?"))

    print(memory.recall("have you seen my wallet?"))

    memory.forget_all()
    os.remove("test_memory.json")
