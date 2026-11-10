import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import io
import os
import time
from google import genai
from google.genai import types

# --- إعداد Gemini ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

# نماذج حقيقية من جوجل — الأسرع أولاً
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
]

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

translations = {
    "person": "شخص", "chair": "كرسي", "cup": "كوباية",
    "cell phone": "موبايل", "book": "كتاب", "bottle": "إزازة",
    "laptop": "لابتوب", "car": "عربية", "dog": "كلب", "cat": "قطة"
}

def describe_with_gemini(image, yolo_description):
    """تحسين وصف YOLO باستخدام Gemini، مع خطة بديلة لو فشل."""
    if not GEMINI_API_KEY:
        return yolo_description  # لو مفيش مفتاح، استخدم YOLO بس

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""أنت مساعد بصري ذكي لمستخدم كفيف.
وصف مبدئي من نظام كشف الأشياء: "{yolo_description}"
حلل الصورة دي واديني وصف نهائي بالعربي، دقيق ومفيد.
لو الوصف المبدئي فيه أخطاء (زي إنه قال ريموت بدل تليفون)، صححها.
وصف المشهد ككل، مش بس الأشياء."""

    for model_name in GEMINI_MODELS:
        for attempt in range(2):
            try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                except Exception:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image]
                    )
                return response.text  # نجح
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str:
                    time.sleep(1)  # انتظار قصير
                    continue
                else:
                    break

    # ⬅️ الخطة البديلة: نرجع وصف YOLO بس بدل ما نعلّق المستخدم
    return yolo_description + " (الوصف التفصيلي غير متاح حاليًا)"


# --- واجهة Streamlit ---
st.set_page_config(page_title="بصيرة", page_icon="👁️")
st.title("👁️ بصيرة — المساعد البصري الذكي")
st.write("صور أي حاجة بكاميرا تليفونك، ارفع الصورة هنا، وهقولك هي إيه")

uploaded_file = st.file_uploader("ارفع صورة من معرض الصور", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة اللي رفعتها", use_column_width=True)

    with st.spinner("بصيرة بيشوف الصورة..."):
        # YOLO على الصورة الأصلية
        results = model(image, verbose=False)[0]
        names = [model.names[int(c)] for c in results.boxes.cls]
        unique = list(dict.fromkeys(names))

        if not unique:
            yolo_description = "مش شايف حاجة واضحة."
        else:
            arabic = [translations.get(n, n) for n in unique]
            yolo_description = "أنا شايف: " + "، ".join(arabic)

        st.info("جاري تحسين الوصف بالذكاء الاصطناعي...")
        final_description = describe_with_gemini(image, yolo_description)

        st.success(final_description)

        # تحويل النص لصوت
        tts = gTTS(text=final_description, lang='ar', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes, format='audio/mp3')
