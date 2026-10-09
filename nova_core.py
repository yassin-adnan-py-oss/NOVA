# NOVA - Core Vision Pipeline (Phase 1)
# كاميرا + YOLO Detection/Tracking + Direction + Approach Status + Color + Memory + Voice (TTS)
#
# pip install opencv-python ultralytics pyttsx3 torch torchvision timm

import cv2
import time
import threading
from ultralytics import YOLO
from nova_color_organizer import ColorOrganizer
from nova_memory import Memory

# ---------------------------------------------------------
# الإعدادات (Config)
# ---------------------------------------------------------
MODEL_PATH = 'yolov8s.pt'
SPEAK_COOLDOWN = 2.5          # ثانية بين كل جملة صوتية والتانية
SIZE_CHANGE_THRESHOLD = 0.03  # نسبة التغير في المساحة عشان نعتبره "بيقرب/بيبعد" فعلاً
VOICE_ENABLED = True          # خليها False لو عايز توقف الصوت مؤقتًا

# ---------------------------------------------------------
# التهيئة (Setup)
# ---------------------------------------------------------
model = YOLO(MODEL_PATH)
color_organizer = ColorOrganizer()
memory = Memory()

# CAP_DSHOW instead of the default backend: on some Windows laptops, the
# default (Media Foundation) backend opens the camera but fails to read
# frames from it. DirectShow works reliably in that case.
#
# مؤقتًا بنستخدم فيديو مسجل بدل الكاميرا الحية، لحد ما نحل مشكلة الـ Driver.
# لما تحل المشكلة، رجع السطر ده بدل السطر اللي تحته:
# webcamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
VIDEO_SOURCE = "test_video.mp4"  # حط اسم الفيديو بتاعك هنا (لازم يكون في نفس المجلد)
webcamera = cv2.VideoCapture(VIDEO_SOURCE)

# الإعدادات دي بتتجاهل تلقائيًا لو المصدر فيديو مسجل (مش كاميرا حية)
webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
webcamera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

previous_sizes = {}     # آخر مساحة معروفة لكل جسم (بالـ track_id)
last_spoken_text = ""   # آخر جملة اتقالت بصوت
last_spoken_time = 0    # آخر توقيت اتكلمنا فيه

tts_engine = None
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except ImportError:
    print("pyttsx3 مش متثبتة - الصوت هيبقى متوقف. شغل: pip install pyttsx3")


# ---------------------------------------------------------
# الدوال المساعدة (Helper Functions)
# ---------------------------------------------------------
def speak_async(text):
    """بتشغل الصوت في thread منفصل عشان متوقفش الفيديو."""
    if not VOICE_ENABLED or tts_engine is None:
        return

    def run():
        tts_engine.say(text)
        tts_engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()


def get_direction(center_x, frame_width):
    """بتحدد هل الجسم على اليمين، الشمال، ولا في النص."""
    left_boundary = frame_width / 3
    right_boundary = frame_width * 2 / 3

    if center_x < left_boundary:
        return "left"
    elif center_x > right_boundary:
        return "right"
    else:
        return "center"


def get_approach_status(track_id, area, previous_sizes, threshold):
    """بتحدد هل الجسم بيقرب، بيبعد، ولا ثابت - بمقارنة حجم الـ box عبر الوقت."""
    status = "stable"
    if track_id in previous_sizes:
        old_area = previous_sizes[track_id]
        diff = area - old_area
        if diff > old_area * threshold:
            status = "approaching"
        elif diff < -old_area * threshold:
            status = "moving away"
    previous_sizes[track_id] = area
    return status


# ---------------------------------------------------------
# الحلقة الرئيسية (Main Loop)
# ---------------------------------------------------------
def main():
    global last_spoken_text, last_spoken_time

    while True:
        success, frame = webcamera.read()
        if not success:
            # لو المصدر فيديو مسجل (مش كاميرا حية)، لما يخلص نرجعه لأول لقطة تاني
            # بدل ما نوقف البرنامج، عشان نقدر نتفرج عليه أكتر من مرة أثناء التجربة
            if isinstance(VIDEO_SOURCE, str):
                webcamera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("failed to capture frame")
            break

        frame_width = frame.shape[1]
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):
                x1, y1, x2, y2 = box.tolist()
                area = (x2 - x1) * (y2 - y1)
                track_id = int(track_id.item())
                center_x = (x1 + x2) / 2

                direction = get_direction(center_x, frame_width)
                label = model.names[int(cls)]
                status = get_approach_status(track_id, area, previous_sizes, SIZE_CHANGE_THRESHOLD)

                # كشف الألوان: بناخد نفس bounding box بتاع الجسم ونحلل ألوانه
                colors = color_organizer.get_dominant_colors(frame, bbox=(x1, y1, x2, y2), top_n=2)
                color_name = colors[0]["color"] if colors else ""

                # تسجيل الجسم في الذاكرة لو كان من النوع اللي يستاهل نتذكره
                memory.remember(label, direction, color_name)

                # النص المكتوب على الشاشة (بيتحدث كل فريم بدون مشكلة)
                display_text = f"{color_name} {label} ({direction}) - {status}".strip()
                cv2.putText(annotated_frame, display_text,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # الجملة اللي هتتقال بصوت (بس لو جديدة ومختلفة عن آخر جملة)
                spoken_label = f"{color_name} {label}".strip()
                spoken_text = f"{spoken_label} on your {direction}, {status}"
                current_time = time.time()

                is_new_sentence = (spoken_text != last_spoken_text)
                cooldown_passed = (current_time - last_spoken_time) > SPEAK_COOLDOWN

                if is_new_sentence and cooldown_passed:
                    speak_async(spoken_text)
                    last_spoken_text = spoken_text
                    last_spoken_time = current_time

        cv2.imshow("NOVA - Live Camera", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("m"):
            # اضغط "m" في أي وقت عشان تسأل النظام عن حاجة ضاعت (اختبار مؤقت بالكتابة)
            query = input("Ask NOVA (e.g. 'where are my keys?'): ")
            print(memory.recall(query))

    webcamera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
