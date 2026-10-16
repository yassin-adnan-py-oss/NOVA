# NOVA - Agent Module (Hybrid: reliable keyword routing + local Llama for open questions)
#
# ليه الطريقة دي مش Tool Calling مباشر؟
# جربنا نخلي Llama 3.2 (1B) يقرر بنفسه (عبر Tool Calling) أي أداة يستخدم،
# لكن النسخة الصغيرة دي بتتلخبط أحيانًا وترفض أسئلة عادية جدًا (زي "فين مفاتيحي؟")
# بشكل غلط تمامًا. الحل العملي: الكلمات المفتاحية (موثوقة 100%) بتتعرف
# على أسئلة NOVA الأساسية وبترد عليها مباشرة، وأي سؤال عام تاني بيروح
# لـ Llama يرد عليه بحرية (زي أي شات بوت عادي).
#
# pip install ollama
# لازم Ollama يكون شغال على الجهاز (ollama run llama3.2:1b جربناه قبل كده)

import ollama
import json

MODEL_NAME = "llama3.2:3b"

# تعريف بسيط بالاسم بس - من غير تقييد المواضيع، عشان NOVA يفضل قادر يجاوب
# على أي سؤال عام (زي Llama الأصلي)
SYSTEM_PROMPT = "Your name is NOVA. You are a helpful AI assistant."

# ------------------------------------------------------------------
# كلمات بتدل على إن السؤال خاص بميزات NOVA نفسها (مش سؤال عام)
# لاحظ: دول نفس الكلمات اللي في nova_agent.py (النسخة البسيطة)
# ------------------------------------------------------------------
FIND_OBJECT_KEYWORDS = ["where", "find", "لقيت", "فين"]
DESCRIBE_SCENE_KEYWORDS = [
    "what do you see", "what's in front", "what is in front", "in front of me",
    "what's around", "what is around", "around me", "surroundings", "describe",
    "ايه قدامي", "شايف ايه", "حواليا",
]
CLOSE_KEYWORDS = ["close", "near", "careful", "danger", "قريب"]
DIRECTION_PHRASE = {"left": "on your left", "center": "straight ahead", "right": "on your right"}


class Agent:
    """Agent حقيقي بيستخدم Llama (محليًا) عشان يقرر أي أداة يستخدم."""

    def __init__(self, memory, get_current_objects):
        self.memory = memory
        self.get_current_objects = get_current_objects

    # ----------------------------------------------------------------
    # الأدوات الموثوقة (Rule-Based) - بترد مباشرة من غير ما تدخل Llama في القصة
    # ----------------------------------------------------------------
    def _handle_find_object(self, query):
        return self.memory.recall(query)

    def _handle_describe_scene(self):
        objects = self.get_current_objects()
        if not objects:
            return "Nothing is clearly visible right now."
        parts = [f"a {o['color']} {o['label']} on your {o['direction']}".strip() for o in objects]
        return "I see " + ", ".join(parts)

    def _handle_close_objects(self):
        close = [o for o in self.get_current_objects()
                 if o.get("distance") in ("Close", "Very close")]
        if not close:
            return "Nothing is close to you right now."
        parts = [f"{o['label']} {DIRECTION_PHRASE.get(o['direction'], '')}".strip() for o in close]
        return "Close: " + ", ".join(parts)

    # ----------------------------------------------------------------
    # نقطة الدخول الرئيسية
    # ----------------------------------------------------------------
    def handle(self, query):
        query_lower = query.lower()

        # الأولوية للكلمات المفتاحية الموثوقة - دي أسئلة خاصة بميزات NOVA
        # بترد عليها مباشرة من غير ما نخاطر بلخبطة Llama مع الـ Tool Calling
        if any(kw in query_lower for kw in FIND_OBJECT_KEYWORDS):
            return self._handle_find_object(query)

        if any(kw in query_lower for kw in CLOSE_KEYWORDS):
            return self._handle_close_objects()

        if any(kw in query_lower for kw in DESCRIBE_SCENE_KEYWORDS):
            return self._handle_describe_scene()

        # أي سؤال عام تاني (مش عن ميزات NOVA) بيروح لـ Llama بحرية كاملة
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return response["message"]["content"]


if __name__ == "__main__":
    from nova_memory import Memory

    memory = Memory("test_agent_memory.json")
    memory.remember("keys", "left", "red")

    def fake_current_objects():
        return [{"label": "laptop", "color": "silver", "direction": "center"}]

    agent = Agent(memory, fake_current_objects)

    print(agent.handle("where are my keys?"))
    print(agent.handle("what do you see right now?"))
    print(agent.handle("tell me a fun fact about space"))  # سؤال عام - المفروض يروح لـ Llama

    import os
    memory.forget_all()
    os.remove("test_agent_memory.json")
