import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import io
import os
from google import genai

# --- إعداد Gemini ---
# يقرأ المفتاح من Secrets في Streamlit Cloud
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# قاموس الترجمة (زي ما هو)
translations = {
    "person": "شخص", "chair": "كرسي", "cup": "كوباية",
    "cell phone": "موبايل", "book": "كتاب", "bottle": "إزازة",
    "laptop": "لابتوب", "car": "عربية", "dog": "كلب", "cat": "قطة"
}

def describe_with_gemini(image: Image.Image, yolo_description: str) -> str:
    """تستخدم Gemini لتحسين وصف YOLO."""
    if not GEMINI_API_KEY:
        return "خطأ: مفتاح Gemini API غير موجود. تأكد من إعداده في Secrets."
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        # نستخدم النموذج السريع والمجاني
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                f"""
                أنت مساعد بصري ذكي لمستخدم كفيف.
                وصف مبدئي من نظام كشف الأشياء: "{yolo_description}"
                حلل الصورة دي واديني وصف نهائي بالعربي، دقيق ومفيد.
                لو الوصف المبدئي فيه أخطاء (زي إنه قال ريموت بدل تليفون)، صححها.
                وصف المشهد ككل، مش بس الأشياء.
                """,
                image
            ]
        )
        return response.text
    except Exception as e:
        return f"عذرًا، حدث خطأ أثناء استخدام Gemini: {e}"

# --- واجهة Streamlit ---
st.set_page_config(page_title="بصيرة", page_icon="👁️")
st.title("👁️ بصيرة — المساعد البصري الذكي")
st.write("صور أي حاجة بكاميرا تليفونك، ارفع الصورة هنا، وهقولك هي إيه")

uploaded_file = st.file_uploader("ارفع صورة من معرض الصور", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة اللي رفعتها", use_column_width=True)

    with st.spinner("بصيرة بيشوف الصورة..."):
        # 1. كشف الأشياء بـ YOLO
        results = model(image, verbose=False)[0]
        names = [model.names[int(c)] for c in results.boxes.cls]
        unique = list(dict.fromkeys(names))

        if not unique:
            yolo_description = "مش شايف حاجة واضحة."
        else:
            arabic = [translations.get(n, n) for n in unique]
            yolo_description = "أنا شايف: " + "، ".join(arabic)

        # 2. تحسين الوصف بـ Gemini
        st.info("جاري تحسين الوصف بالذكاء الاصطناعي...")
        final_description = describe_with_gemini(image, yolo_description)

        # 3. عرض النتيجة والصوت
        st.success(final_description)

        tts = gTTS(text=final_description, lang='ar', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes, format='audio/mp3')
