import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import io
import os
import time
from google import genai

# --- إعداد Gemini ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

# قائمة النماذج: الأساسي، ثم الاحتياطي
GEMINI_MODELS = [
    "gemini-3.8-flash",          # الأساسي
    "gemini-2.0-flash-lite",     # احتياطي 1 (أخف وأسرع، حصة منفصلة)
    "gemini-1.5-flash",          # احتياطي 2 (مستقر وقديم)
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

def describe_with_gemini(image: Image.Image, yolo_description: str) -> str:
    """تستخدم Gemini لتحسين وصف YOLO مع إعادة محاولة ونماذج احتياطية."""
    if not GEMINI_API_KEY:
        return "خطأ: مفتاح Gemini API غير موجود. تأكد من إعداده في Secrets."
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    أنت مساعد بصري ذكي لمستخدم كفيف.
    وصف مبدئي من نظام كشف الأشياء: "{yolo_description}"
    حلل الصورة دي واديني وصف نهائي بالعربي، دقيق ومفيد.
    لو الوصف المبدئي فيه أخطاء (زي إنه قال ريموت بدل تليفون)، صححها.
    وصف المشهد ككل، مش بس الأشياء.
    """

    last_error = None
    # نجرب كل نموذج في القائمة
    for model_name in GEMINI_MODELS:
        # لكل نموذج، نحاول 3 مرات كحد أقصى
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image]
                )
                return response.text  # نجح! نرجع النتيجة
            except Exception as e:
                last_error = e
                error_str = str(e)
                # لو الخطأ 503 أو 429 (مؤقت)، نستنى ونحاول تاني
                if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str:
                    wait_time = 2 ** attempt  # 1، 2، 4 ثواني
                    time.sleep(wait_time)
                    continue  # حاول تاني بنفس النموذج
                else:
                    break  # خطأ تاني (زي مفتاح غلط)، نجرب النموذج اللي بعده
    
    # لو كل المحاولات فشلت
    return f"عذرًا، خدمة التحسين مشغولة حاليًا. جرب تاني بعد شوية. (تفاصيل: {last_error})"

# --- واجهة Streamlit (زي ما هي) ---
st.set_page_config(page_title="بصيرة", page_icon="👁️")
st.title("👁️ بصيرة — المساعد البصري الذكي")
st.write("صور أي حاجة بكاميرا تليفونك، ارفع الصورة هنا، وهقولك هي إيه")

uploaded_file = st.file_uploader("ارفع صورة من معرض الصور", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة اللي رفعتها", use_column_width=True)

    with st.spinner("بصيرة بيشوف الصورة..."):
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

        tts = gTTS(text=final_description, lang='ar', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes, format='audio/mp3')
