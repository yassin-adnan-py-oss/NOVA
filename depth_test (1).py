# NOVA - Depth Estimation Test (Standalone)
# الهدف: نجرب موديل MiDaS لوحده، نتأكد إنه شغال كويس قبل ما ندمجه مع nova_core.py
#
# pip install torch torchvision timm opencv-python

import cv2
import torch
import numpy as np

# ---------------------------------------------------------
# تحميل الموديل (أول مرة هيتنزل تلقائيًا من الإنترنت)
# ---------------------------------------------------------
# MiDaS_small = أخف نسخة، مناسبة للابتوب عادي بدون GPU قوي
model_type = "MiDaS_small"

print("جاري تحميل موديل MiDaS... (أول مرة بس، ممكن تاخد شوية وقت)")
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# تحديد الجهاز اللي هيشتغل عليه الموديل (GPU لو موجود، وإلا CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()  # وضع "التقييم" - يعني الموديل هيشتغل للتنبؤ بس، مش للتدريب

print(f"الموديل شغال على: {device}")

# ---------------------------------------------------------
# تحميل أدوات تجهيز الصورة (Transforms) الخاصة بالموديل
# ---------------------------------------------------------
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # نفس النسخة الخفيفة اللي اخترناها فوق

# ---------------------------------------------------------
# الكاميرا
# ---------------------------------------------------------
webcamera = cv2.VideoCapture(0)

while True:
    success, frame = webcamera.read()
    if not success:
        print("failed to capture frame")
        break

    # MiDaS محتاج الصورة بصيغة RGB، لكن OpenCV بيدينا BGR بشكل افتراضي
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # تجهيز الصورة بالشكل اللي الموديل يفهمه، وإرسالها لنفس جهاز الموديل (GPU/CPU)
    input_batch = transform(img_rgb).to(device)

    # التنبؤ (Inference) - هنا بيحصل الحساب الفعلي لخريطة العمق
    with torch.no_grad():  # يمنع حساب gradients (مش محتاجينها، بنستخدم الموديل بس مش بندربه)
        prediction = midas(input_batch)

        # الموديل بيرجع النتيجة بحجم مختلف عن الصورة الأصلية، فبنعيد ضبط الحجم (resize)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # تحويل النتيجة من "تنسور" (Tensor) بايتورش لمصفوفة نمباي عادية
    depth_map = prediction.cpu().numpy()

    # تطبيع القيم (Normalization) عشان تتحول لصورة تقدر تتعرض (من 0 إلى 255)
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # تلوين خريطة العمق عشان تبقى مقروءة بصريًا (أحمر = قريب، أزرق = بعيد -- حسب الخريطة اللونية)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    # عرض الصورة الأصلية وخريطة العمق جنب بعض
    cv2.imshow("Original", frame)
    cv2.imshow("Depth Map", depth_colored)

    if cv2.waitKey(1) == ord("q"):
        break

webcamera.release()
cv2.destroyAllWindows()
